#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import dataclasses
import gc
import json
import random
import statistics
import time
import types
from pathlib import Path
from typing import Any

import torch

import flashinfer
from flashinfer.autotune_cache import MeasurementPolicy, autotune_v2
from flashinfer.autotuner import AutoTuner
from flashinfer.fp8_quantization import mxfp8_quantize
from flashinfer.gemm.gemm_base import (
    DEFAULT_WORKSPACE_SIZE,
    _MM_MXFP8_DYNAMIC_QUANT_TUNING_CONFIG,
    _TrtllmDynamicQuantMxfp8Runner,
)

Candidate = tuple[bool, int] | int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=20260825)
    parser.add_argument("--threshold-pct", type=float, default=-1.0)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--refinement-rounds", type=int, default=3)
    parser.add_argument("--evaluation-rounds", type=int, default=5)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def parse_candidate(value: str) -> Candidate:
    layout, tactic = value.split(":", maxsplit=1)
    return layout == "8x4", int(tactic)


def format_candidate(candidate: Candidate) -> str:
    if isinstance(candidate, int):
        return str(candidate)
    layout, tactic = candidate
    return f"{'8x4' if layout else '128x4'}:{tactic}"


def load_shapes(path: Path, threshold_pct: float) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if float(row["online_graph_regret_pct"]) > threshold_pct
        ]
    return sorted(rows, key=lambda row: tuple(int(row[key]) for key in ("m", "n", "k")))


def prepare_problem(m: int, n: int, k: int, seed: int):
    torch.manual_seed(seed)
    device = torch.device("cuda")
    activation = torch.randn((m, k), dtype=torch.bfloat16, device=device)
    weight = torch.randn((n, k), dtype=torch.bfloat16, device=device)
    weight_q, weight_scale = mxfp8_quantize(
        weight,
        sf_swizzle_layout=flashinfer.SfLayout.layout_linear,
    )
    prepared_weight, prepared_scale = flashinfer.prepare_mxfp8_trtllm_weights(
        weight_q,
        weight_scale,
    )
    output = torch.empty((m, n), dtype=torch.bfloat16, device=device)
    workspace = torch.empty(DEFAULT_WORKSPACE_SIZE, dtype=torch.uint8, device=device)
    runner = _TrtllmDynamicQuantMxfp8Runner(device)
    inputs: list[Any] = [
        activation,
        prepared_weight,
        prepared_scale,
        torch.bfloat16,
        output,
        workspace,
    ]
    runner(inputs, tactic=-1, do_preparation=True)
    torch.cuda.synchronize()
    del weight, weight_q, weight_scale
    gc.collect()
    return runner, inputs


def randomize_candidate_order(runner: Any, seed: int) -> None:
    original = runner.get_valid_tactics

    def shuffled(_self: Any, inputs: list[Any], profile: Any) -> list[Candidate]:
        candidates = list(original(inputs, profile))
        random.Random(seed).shuffle(candidates)
        return candidates

    runner.get_valid_tactics = types.MethodType(shuffled, runner)


def select_with_policy(
    tuner: AutoTuner,
    runner: Any,
    inputs: list[Any],
    tuning_config: Any,
    custom_op: str,
    policy: MeasurementPolicy,
) -> tuple[Candidate, float, list[dict[str, Any]]]:
    profile_calls: list[dict[str, Any]] = []
    original = tuner._profile_single_kernel

    def traced_profile(
        _self: AutoTuner,
        selected_runner: Any,
        selected_inputs: list[Any],
        tactic: Candidate,
        selected_config: Any,
        **kwargs: Any,
    ) -> float:
        elapsed = original(
            selected_runner,
            selected_inputs,
            tactic,
            selected_config,
            **kwargs,
        )
        profile_calls.append(
            {"tactic": format_candidate(tactic), "elapsed_ms": float(elapsed)}
        )
        return elapsed

    tuner._profile_single_kernel = types.MethodType(traced_profile, tuner)
    try:
        started = time.perf_counter()
        with autotune_v2(
            persistent_cache=False,
            measurement_policy=policy,
        ):
            _, tactic = tuner.choose_one(
                custom_op,
                [runner],
                tuning_config,
                inputs,
            )
        torch.cuda.synchronize()
        wall_s = time.perf_counter() - started
    finally:
        tuner._profile_single_kernel = original
    return tactic, wall_s, profile_calls


def measure_interleaved(
    tuner: AutoTuner,
    runner: Any,
    inputs: list[Any],
    tuning_config: Any,
    candidates: list[Candidate],
    rounds: int,
    seed: int,
) -> dict[str, list[float]]:
    config, input_batches = tuner.prepare_tactic_profile(inputs, tuning_config)
    measurements = {format_candidate(candidate): [] for candidate in candidates}
    rng = random.Random(seed)
    for _ in range(rounds):
        order = list(candidates)
        rng.shuffle(order)
        for candidate in order:
            elapsed_ms = tuner.profile_tactic(
                runner,
                inputs,
                candidate,
                config,
                input_batches,
            )
            measurements[format_candidate(candidate)].append(float(elapsed_ms))
    return measurements


def main() -> None:
    args = parse_args()
    shapes = load_shapes(args.shape_summary, args.threshold_pct)
    if args.limit is not None:
        shapes = sorted(
            shapes,
            key=lambda row: float(row["online_graph_regret_pct"]),
            reverse=True,
        )[: args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    exact_shape_config = dataclasses.replace(
        _MM_MXFP8_DYNAMIC_QUANT_TUNING_CONFIG,
        dynamic_tensor_specs=(),
    )
    policies = {
        "one_pass": MeasurementPolicy(execution_mode="cuda_graph", cold_l2=True),
        "top3x3": MeasurementPolicy(
            execution_mode="cuda_graph",
            cold_l2=True,
            refinement_top_k=args.top_k,
            refinement_rounds=args.refinement_rounds,
        ),
    }

    with args.output.open("w", buffering=1) as handle:
        for repetition in range(1, args.repetitions + 1):
            ordered_shapes = list(shapes)
            random.Random(args.base_seed + repetition).shuffle(ordered_shapes)
            for shape_index, shape in enumerate(ordered_shapes):
                m, n, k = (int(shape[key]) for key in ("m", "n", "k"))
                seed = args.base_seed + repetition * 10_000 + shape_index
                runner, inputs = prepare_problem(m, n, k, seed)
                randomize_candidate_order(runner, seed)
                tuner = AutoTuner.get()

                order = list(policies)
                if seed % 2:
                    order.reverse()
                selections: dict[str, Candidate] = {}
                walls: dict[str, float] = {}
                traces: dict[str, list[dict[str, Any]]] = {}
                for name in order:
                    tactic, wall_s, calls = select_with_policy(
                        tuner,
                        runner,
                        inputs,
                        exact_shape_config,
                        f"mxfp8_policy_{name}_r{repetition}_{m}_{n}_{k}",
                        policies[name],
                    )
                    selections[name] = tactic
                    walls[name] = wall_s
                    traces[name] = calls

                oracle = parse_candidate(shape["graph_cold_oracle"])
                evaluation_candidates = list(
                    dict.fromkeys(
                        (selections["one_pass"], selections["top3x3"], oracle)
                    )
                )
                evaluation = measure_interleaved(
                    tuner,
                    runner,
                    inputs,
                    exact_shape_config,
                    evaluation_candidates,
                    args.evaluation_rounds,
                    seed + 1,
                )
                medians = {
                    candidate: statistics.median(values)
                    for candidate, values in evaluation.items()
                }
                local_best_ms = min(medians.values())
                one_pass_ms = medians[format_candidate(selections["one_pass"])]
                top3x3_ms = medians[format_candidate(selections["top3x3"])]
                row = {
                    "repetition": repetition,
                    "m": m,
                    "n": n,
                    "k": k,
                    "one_pass": format_candidate(selections["one_pass"]),
                    "top3x3": format_candidate(selections["top3x3"]),
                    "reference_oracle": format_candidate(oracle),
                    "one_pass_wall_s": walls["one_pass"],
                    "top3x3_wall_s": walls["top3x3"],
                    "refinement_overhead_pct": (
                        walls["top3x3"] / walls["one_pass"] - 1.0
                    )
                    * 100.0,
                    "one_pass_profile_calls": len(traces["one_pass"]),
                    "top3x3_profile_calls": len(traces["top3x3"]),
                    "one_pass_trace": traces["one_pass"],
                    "top3x3_trace": traces["top3x3"],
                    "evaluation_ms": evaluation,
                    "one_pass_cold_ms": one_pass_ms,
                    "top3x3_cold_ms": top3x3_ms,
                    "local_best_cold_ms": local_best_ms,
                    "one_pass_regret_pct": (one_pass_ms / local_best_ms - 1.0) * 100.0,
                    "top3x3_regret_pct": (top3x3_ms / local_best_ms - 1.0) * 100.0,
                }
                handle.write(json.dumps(row, sort_keys=True) + "\n")
                del runner, inputs
                gc.collect()
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
