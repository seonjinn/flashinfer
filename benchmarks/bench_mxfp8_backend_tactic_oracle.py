#!/usr/bin/env python3
"""Compare serving-selected MXFP8 tactics with a CUDA Graph oracle.

For every observed serving shape, this benchmark loads the tactic recorded by
the serving process, enumerates every valid tactic exposed by the same backend,
and measures all candidates under CUDA Graph replay with cold L2 inputs.
Candidate order is shuffled between rounds to reduce thermal and ordering bias.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import statistics
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from benchmarks.bench_cutedsl_mxfp8_serving_shapes import (
    Shape,
    _configure_source_jit_paths,
    _cosine_similarity,
    _jsonable,
    _make_call,
    _make_problem,
    _validate_backend_layout,
    group_shapes,
    load_shapes,
)


def _tactic_key(tactic: Any) -> str:
    return json.dumps(_normalize_tactic(tactic), separators=(",", ":"))


def _normalize_tactic(tactic: Any) -> Any:
    if isinstance(tactic, (list, tuple)):
        return [_normalize_tactic(value) for value in tactic]
    return _jsonable(tactic)


def _restore_tactic(tactic: Any) -> Any:
    if isinstance(tactic, list):
        return tuple(_restore_tactic(value) for value in tactic)
    return tactic


def _load_selected_tactics(path: Path) -> dict[tuple[Shape, str], Any]:
    selected: dict[tuple[Shape, str], Any] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            shape = Shape(int(row["m"]), int(row["n"]), int(row["k"]))
            key = (shape, row["runner"])
            tactic = _restore_tactic(json.loads(row["selected_tactic"]))
            if key in selected and selected[key] != tactic:
                raise ValueError(
                    f"conflicting selected tactics for {key}: "
                    f"{selected[key]} vs {tactic}"
                )
            selected[key] = tactic
    if not selected:
        raise ValueError(f"selected tactic CSV is empty: {path}")
    return selected


def _deduplicate_tactics(tactics: list[Any]) -> list[Any]:
    unique: list[Any] = []
    seen: set[str] = set()
    for tactic in tactics:
        key = _tactic_key(tactic)
        if key not in seen:
            seen.add(key)
            unique.append(tactic)
    return unique


def _candidate_tactics(valid_tactics: list[Any], *, selected_tactic: Any) -> list[Any]:
    candidates = _deduplicate_tactics(valid_tactics)
    candidate_keys = {_tactic_key(tactic) for tactic in candidates}
    if _tactic_key(selected_tactic) in candidate_keys:
        return candidates
    if selected_tactic == -1:
        return [selected_tactic, *candidates]
    raise ValueError(
        f"selected tactic is not valid: {_normalize_tactic(selected_tactic)}"
    )


def _capture_selected_output(runner: Any, inputs: list[Any], tactic: Any) -> Any:
    return runner(inputs, tactic=tactic).detach().clone()


def _profile_from_shapes(
    input_shapes: tuple[tuple[int, ...], ...],
    *,
    profile_type: Callable[..., Any],
    static_dim_type: Callable[[int], Any],
) -> Any:
    shapes = [
        [static_dim_type(dimension) for dimension in shape] for shape in input_shapes
    ]
    return profile_type(shapes, [None] * len(shapes))


def _make_concrete_profile(inputs: list[Any]) -> Any:
    from flashinfer.autotuner import AutoTuner, OptimizationProfile, StaticDim

    input_shapes = AutoTuner.get()._get_input_sizes(inputs)  # noqa: SLF001
    return _profile_from_shapes(
        input_shapes,
        profile_type=OptimizationProfile,
        static_dim_type=StaticDim,
    )


def _capture_invocation(call: Callable[[], Any]) -> tuple[Any, dict[str, Any]]:
    from flashinfer.autotuner import AutoTuner

    captured: dict[str, Any] = {}
    original = AutoTuner.choose_one

    def spy(self, custom_op, runners, tuning_config, inputs, **kwargs):
        runner, tactic = original(
            self, custom_op, runners, tuning_config, inputs, **kwargs
        )
        if custom_op == "mxfp8_gemm":
            captured.update(
                {
                    "runner": runner,
                    "selected_tactic": tactic,
                    "tuning_config": tuning_config,
                    "inputs": inputs,
                }
            )
        return runner, tactic

    AutoTuner.choose_one = spy
    try:
        output = call()
    finally:
        AutoTuner.choose_one = original
    if not captured:
        raise RuntimeError("mm_mxfp8 invocation was not captured")
    return output, captured


def _measure_candidate(
    *,
    runner: Any,
    inputs: list[Any],
    tactic: Any,
    reference: Any,
    selected_output: Any,
    dry_run_iters: int,
    repeat_iters: int,
    graph_calls: int,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    import numpy as np
    import torch

    from flashinfer.testing.utils import bench_gpu_time

    def run(candidate_inputs: list[Any]) -> None:
        runner(candidate_inputs, tactic=tactic)

    output = runner(inputs, tactic=tactic)
    torch.cuda.synchronize()
    cosine = _cosine_similarity(reference, output)
    output_float = output.float()
    selected_float = selected_output.float()
    finite = bool(torch.isfinite(output_float).all().item())
    max_abs_diff = float((output_float - selected_float).abs().max().item())
    matches_selected = finite and torch.allclose(
        output_float, selected_float, rtol=rtol, atol=atol
    )
    samples = bench_gpu_time(
        fn=run,
        input_args=(inputs,),
        dry_run_iters=dry_run_iters,
        repeat_iters=repeat_iters,
        sleep_after_run=False,
        use_cuda_graph=True,
        num_iters_within_graph=graph_calls,
        cold_l2_cache=True,
    )
    samples_ms = [float(sample) for sample in samples]
    return {
        "tactic": _normalize_tactic(tactic),
        "median_ms": float(np.median(samples_ms)),
        "p10_ms": float(np.percentile(samples_ms, 10)),
        "p90_ms": float(np.percentile(samples_ms, 90)),
        "samples_ms": samples_ms,
        "cosine_similarity": cosine,
        "finite": finite,
        "matches_selected": matches_selected,
        "max_abs_diff_from_selected": max_abs_diff,
    }


def _aggregate_candidate_rounds(
    *, tactic: Any, rows: list[dict[str, Any]]
) -> dict[str, Any]:
    import numpy as np

    samples = [sample for row in rows for sample in row["samples_ms"]]
    return {
        "tactic": _normalize_tactic(tactic),
        "median_ms": float(np.median(samples)),
        "p10_ms": float(np.percentile(samples, 10)),
        "p90_ms": float(np.percentile(samples, 90)),
        "samples_ms": samples,
        "cosine_similarity": min(row["cosine_similarity"] for row in rows),
        "finite": all(row["finite"] for row in rows),
        "matches_selected": all(row["matches_selected"] for row in rows),
        "max_abs_diff_from_selected": max(
            row["max_abs_diff_from_selected"] for row in rows
        ),
    }


def _summarize_shape(
    *,
    shape: Shape,
    runner_name: str,
    selected_tactic: Any,
    candidates: list[dict[str, Any]],
    min_cosine: float,
) -> dict[str, Any]:
    selected_key = _tactic_key(selected_tactic)
    selected = next(
        (row for row in candidates if _tactic_key(row["tactic"]) == selected_key),
        None,
    )
    if selected is None:
        raise RuntimeError(
            f"Selected tactic was not measured for {shape}: "
            f"{_normalize_tactic(selected_tactic)}"
        )
    correct = [
        row
        for row in candidates
        if row["cosine_similarity"] >= min_cosine
        and row.get("finite") is True
        and row.get("matches_selected") is True
    ]
    if not correct:
        raise RuntimeError(f"No correct tactic was measured for {shape}")
    oracle = min(correct, key=lambda row: row["median_ms"])
    speedup = selected["median_ms"] / oracle["median_ms"]
    return {
        **asdict(shape),
        "runner": runner_name,
        "candidate_count": len(candidates),
        "correct_candidate_count": len(correct),
        "selected_tactic": _normalize_tactic(selected_tactic),
        "selected_ms": selected["median_ms"],
        "selected_cosine_similarity": selected["cosine_similarity"],
        "oracle_tactic": oracle["tactic"],
        "oracle_ms": oracle["median_ms"],
        "oracle_cosine_similarity": oracle["cosine_similarity"],
        "oracle_finite": oracle["finite"],
        "oracle_matches_selected": oracle["matches_selected"],
        "oracle_max_abs_diff_from_selected": oracle.get(
            "max_abs_diff_from_selected", 0.0
        ),
        "speedup": speedup,
        "regret_pct": 100.0 * (speedup - 1.0),
        "same_tactic": selected_key == _tactic_key(oracle["tactic"]),
        "candidates": candidates,
    }


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.write_text(json.dumps(report, indent=2) + "\n")


def _geomean(values: list[float]) -> float:
    return math.exp(statistics.fmean(math.log(value) for value in values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shapes", type=Path, required=True)
    parser.add_argument("--selected-tactics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--backend", choices=("cute-dsl", "cutlass", "trtllm"), required=True
    )
    parser.add_argument("--scale-layout", choices=("8x4", "128x4"), required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--dry-run-iters", type=int, default=10)
    parser.add_argument("--repeat-iters", type=int, default=30)
    parser.add_argument("--graph-calls", type=int, default=10)
    parser.add_argument("--min-cosine", type=float, default=0.98)
    parser.add_argument("--rtol", type=float, default=0.02)
    parser.add_argument("--atol", type=float, default=0.1)
    args = parser.parse_args()

    if args.rounds <= 0:
        parser.error("--rounds must be positive")
    if args.dry_run_iters < 0:
        parser.error("--dry-run-iters must be non-negative")
    if args.repeat_iters <= 0:
        parser.error("--repeat-iters must be positive")
    if args.graph_calls <= 0:
        parser.error("--graph-calls must be positive")
    try:
        _validate_backend_layout(args.backend, args.scale_layout)
    except ValueError as error:
        parser.error(str(error))

    import torch
    import flashinfer

    from flashinfer.autotuner import AutoTuner

    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires a CUDA GPU")
    _configure_source_jit_paths()

    shapes = load_shapes(args.shapes)
    groups = group_shapes(shapes)
    selected_tactics = _load_selected_tactics(args.selected_tactics)
    selected_shapes = {shape for shape, _ in selected_tactics}
    missing_selected = sorted(set(shapes) - selected_shapes)
    if missing_selected:
        raise ValueError(
            f"missing serving-selected tactics for {len(missing_selected)} shapes: "
            f"{missing_selected[:8]}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "report.json"
    shape_results: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "backend": args.backend,
        "scale_layout": args.scale_layout,
        "flashinfer_commit": os.getenv("FLASHINFER_COMMIT"),
        "flashinfer_version": flashinfer.__version__,
        "flashinfer_file": flashinfer.__file__,
        "container_sha256": os.getenv("CONTAINER_SHA256"),
        "selected_tactics_file": str(args.selected_tactics),
        "shapes_file": str(args.shapes),
        "timing": {
            "cuda_graph": True,
            "cold_l2_cache": True,
            "rounds": args.rounds,
            "dry_run_iters": args.dry_run_iters,
            "repeat_iters": args.repeat_iters,
            "calls_per_graph": args.graph_calls,
        },
        "correctness": {
            "minimum_cosine_similarity": args.min_cosine,
            "rtol": args.rtol,
            "atol": args.atol,
        },
        "shapes": shape_results,
    }

    tuner = AutoTuner.get()
    profiling_started = time.perf_counter()
    for (n, k), m_values in groups.items():
        for m in m_values:
            shape = Shape(m, n, k)
            problem = _make_problem(shape, args.seed, args.backend, args.scale_layout)
            call = _make_call(problem, args.backend, args.scale_layout)
            tuner.clear_cache()
            output, invocation = _capture_invocation(call)
            runner = invocation["runner"]
            inputs = invocation["inputs"]
            runner_name = runner.__class__.__name__
            execution_key = (shape, runner_name)
            if execution_key not in selected_tactics:
                recorded_runners = sorted(
                    candidate_runner
                    for candidate_shape, candidate_runner in selected_tactics
                    if candidate_shape == shape
                )
                raise RuntimeError(
                    f"Serving runner mismatch for {shape}: "
                    f"recorded={recorded_runners}, oracle={runner_name}"
                )
            selected_tactic = selected_tactics[execution_key]
            selected_output = _capture_selected_output(runner, inputs, selected_tactic)
            if not torch.isfinite(selected_output).all():
                raise RuntimeError(
                    f"Serving-selected tactic produced non-finite output for {shape}"
                )
            try:
                candidates = _candidate_tactics(
                    list(
                        runner.get_valid_tactics(inputs, _make_concrete_profile(inputs))
                    ),
                    selected_tactic=selected_tactic,
                )
            except ValueError as error:
                raise RuntimeError(
                    f"Selected tactic is not valid for {shape}: "
                    f"{_normalize_tactic(selected_tactic)}"
                ) from error
            del output

            rows_by_tactic: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
            tactic_by_key = {_tactic_key(tactic): tactic for tactic in candidates}
            for round_index in range(args.rounds):
                order = list(candidates)
                random.Random(
                    args.seed
                    + round_index * 1_000_003
                    + shape.m * 10_007
                    + shape.n
                    + shape.k
                ).shuffle(order)
                for tactic in order:
                    row = _measure_candidate(
                        runner=runner,
                        inputs=inputs,
                        tactic=tactic,
                        reference=problem[-1],
                        selected_output=selected_output,
                        dry_run_iters=args.dry_run_iters,
                        repeat_iters=args.repeat_iters,
                        graph_calls=args.graph_calls,
                        rtol=args.rtol,
                        atol=args.atol,
                    )
                    rows_by_tactic[_tactic_key(tactic)].append(row)

            aggregated = [
                _aggregate_candidate_rounds(
                    tactic=tactic_by_key[key], rows=rows_by_tactic[key]
                )
                for key in tactic_by_key
            ]
            shape_result = _summarize_shape(
                shape=shape,
                runner_name=runner_name,
                selected_tactic=selected_tactic,
                candidates=aggregated,
                min_cosine=args.min_cosine,
            )
            shape_results.append(shape_result)
            report["geomean_speedup"] = _geomean(
                [row["speedup"] for row in shape_results]
            )
            report["max_regret_pct"] = max(row["regret_pct"] for row in shape_results)
            report["same_tactic_count"] = sum(
                row["same_tactic"] for row in shape_results
            )
            report["profiling_wall_s"] = time.perf_counter() - profiling_started
            report["measured_candidate_gpu_s"] = sum(
                sample / 1000.0
                for shape_row in shape_results
                for candidate in shape_row["candidates"]
                for sample in candidate["samples_ms"]
            )
            _write_report(report_path, report)
            print(
                f"{shape.m},{shape.n},{shape.k},"
                f"selected={shape_result['selected_ms']:.6f}ms,"
                f"oracle={shape_result['oracle_ms']:.6f}ms,"
                f"speedup={shape_result['speedup']:.4f},"
                f"candidates={shape_result['candidate_count']}"
            )
            del problem, runner, inputs
            torch.cuda.empty_cache()

    print(f"Geomean speedup: {report['geomean_speedup']:.6f}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
