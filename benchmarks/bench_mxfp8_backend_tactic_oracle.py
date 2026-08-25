#!/usr/bin/env python3
"""Compare cached MXFP8 tactics with a CUDA Graph exhaustive oracle.

For every observed serving shape, this benchmark loads the exact-M AutoTuner
selection, enumerates every valid tactic exposed by the selected backend, and
measures all candidates under CUDA Graph replay with cold L2 inputs. Candidate
order is shuffled between rounds to reduce thermal and ordering bias.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from benchmarks.bench_cutedsl_mxfp8_serving_shapes import (
    Shape,
    _cache_path,
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


def _deduplicate_tactics(tactics: list[Any]) -> list[Any]:
    unique: list[Any] = []
    seen: set[str] = set()
    for tactic in tactics:
        key = _tactic_key(tactic)
        if key not in seen:
            seen.add(key)
            unique.append(tactic)
    return unique


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
    dry_run_iters: int,
    repeat_iters: int,
    graph_calls: int,
) -> dict[str, Any]:
    import numpy as np
    import torch

    from flashinfer.testing.utils import bench_gpu_time

    def run(candidate_inputs: list[Any]) -> None:
        runner(candidate_inputs, tactic=tactic)

    output = runner(inputs, tactic=tactic)
    torch.cuda.synchronize()
    cosine = _cosine_similarity(reference, output)
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
    }


def _summarize_shape(
    *,
    shape: Shape,
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
    correct = [row for row in candidates if row["cosine_similarity"] >= min_cosine]
    if not correct:
        raise RuntimeError(f"No correct tactic was measured for {shape}")
    oracle = min(correct, key=lambda row: row["median_ms"])
    speedup = selected["median_ms"] / oracle["median_ms"]
    return {
        **asdict(shape),
        "candidate_count": len(candidates),
        "correct_candidate_count": len(correct),
        "selected_tactic": _normalize_tactic(selected_tactic),
        "selected_ms": selected["median_ms"],
        "selected_cosine_similarity": selected["cosine_similarity"],
        "oracle_tactic": oracle["tactic"],
        "oracle_ms": oracle["median_ms"],
        "oracle_cosine_similarity": oracle["cosine_similarity"],
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
    parser.add_argument("--selected-cache-dir", type=Path, required=True)
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

    from flashinfer import autotune
    from flashinfer.autotuner import AutoTuner

    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires a CUDA GPU")
    _configure_source_jit_paths()

    shapes = load_shapes(args.shapes)
    groups = group_shapes(shapes)
    cache_path = _cache_path(
        args.selected_cache_dir, args.backend, args.scale_layout, "exact"
    )
    if not cache_path.is_file():
        raise FileNotFoundError(f"Missing exact-M AutoTuner cache: {cache_path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "report.json"
    shape_results: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "backend": args.backend,
        "scale_layout": args.scale_layout,
        "source_cache": str(cache_path),
        "shapes_file": str(args.shapes),
        "timing": {
            "cuda_graph": True,
            "cold_l2_cache": True,
            "rounds": args.rounds,
            "dry_run_iters": args.dry_run_iters,
            "repeat_iters": args.repeat_iters,
            "calls_per_graph": args.graph_calls,
        },
        "shapes": shape_results,
    }

    tuner = AutoTuner.get()
    for (n, k), m_values in groups.items():
        for m in m_values:
            shape = Shape(m, n, k)
            problem = _make_problem(shape, args.seed, args.backend, args.scale_layout)
            call = _make_call(problem, args.backend, args.scale_layout)
            tuner.clear_cache()
            with autotune(
                False,
                cache=str(cache_path),
                tuning_buckets=m_values,
            ):
                output, invocation = _capture_invocation(call)
            selected_tactic = invocation["selected_tactic"]
            runner = invocation["runner"]
            inputs = invocation["inputs"]
            candidates = _deduplicate_tactics(
                list(runner.get_valid_tactics(inputs, None))
            )
            if _tactic_key(selected_tactic) not in {
                _tactic_key(tactic) for tactic in candidates
            }:
                raise RuntimeError(
                    f"Selected tactic is not valid for {shape}: "
                    f"{_normalize_tactic(selected_tactic)}"
                )
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
                        dry_run_iters=args.dry_run_iters,
                        repeat_iters=args.repeat_iters,
                        graph_calls=args.graph_calls,
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
