#!/usr/bin/env python3
"""Compare CuTeDSL MXFP8 hybrid buckets with exact serving-shape tuning.

The default FlashInfer tuning config maps the dynamic GEMM M dimension to a
hybrid bucket. This benchmark measures the transfer regret from that bucket's
winner at the original serving M, then compares it with a winner tuned at the
exact M. Both paths use the public autotune cache API and CUDA Graph replay.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True, order=True)
class Shape:
    m: int
    n: int
    k: int

    def validate(self) -> None:
        if self.m <= 0 or self.n <= 0 or self.k <= 0:
            raise ValueError(f"M, N, and K must be positive, got {self}")
        if self.k % 32 != 0:
            raise ValueError(f"K must be divisible by 32 for MXFP8, got {self.k}")


def load_shapes(path: Path) -> list[Shape]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Shape CSV has no header: {path}")
        names = {name.strip().lower(): name for name in reader.fieldnames}
        missing = {"m", "n", "k"} - names.keys()
        if missing:
            raise ValueError(f"Shape CSV is missing columns {sorted(missing)}: {path}")
        shapes = {
            Shape(
                m=int(row[names["m"]]),
                n=int(row[names["n"]]),
                k=int(row[names["k"]]),
            )
            for row in reader
        }
    for shape in shapes:
        shape.validate()
    if not shapes:
        raise ValueError(f"Shape CSV is empty: {path}")
    return sorted(shapes)


def group_shapes(shapes: list[Shape]) -> dict[tuple[int, int], tuple[int, ...]]:
    grouped: defaultdict[tuple[int, int], set[int]] = defaultdict(set)
    for shape in shapes:
        grouped[(shape.n, shape.k)].add(shape.m)
    return {key: tuple(sorted(grouped[key])) for key in sorted(grouped)}


def _jsonable(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _capture_selection(call: Callable[[], Any]) -> tuple[Any, dict[str, Any]]:
    from flashinfer.autotuner import AutoTuner

    selected: dict[str, Any] = {}
    original = AutoTuner.choose_one

    def spy(self, custom_op, runners, tuning_config, inputs, **kwargs):
        runner, tactic = original(
            self, custom_op, runners, tuning_config, inputs, **kwargs
        )
        selected[custom_op] = {
            "runner": runner.__class__.__name__,
            "tactic": _jsonable(tactic),
        }
        return runner, tactic

    AutoTuner.choose_one = spy
    try:
        output = call()
    finally:
        AutoTuner.choose_one = original
    if "mxfp8_gemm" not in selected:
        raise RuntimeError(f"mm_mxfp8 selection was not captured: {selected}")
    return output, selected["mxfp8_gemm"]


def _make_problem(shape: Shape, seed: int):
    import torch

    from flashinfer import SfLayout
    from flashinfer.fp8_quantization import mxfp8_quantize

    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed + shape.m * 1_000_003 + shape.n * 101 + shape.k)
    a = torch.randn(
        (shape.m, shape.k),
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    weight = torch.randn(
        (shape.n, shape.k),
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    a_q, a_scale = mxfp8_quantize(a, sf_swizzle_layout=SfLayout.layout_128x4)
    weight_q, weight_scale = mxfp8_quantize(
        weight, sf_swizzle_layout=SfLayout.layout_128x4
    )
    out = torch.empty((shape.m, shape.n), device="cuda", dtype=torch.bfloat16)
    reference = torch.mm(a, weight.T)
    return a_q, weight_q.T, a_scale, weight_scale, out, reference


def _make_call(problem):
    import torch

    from flashinfer import mm_mxfp8

    a_q, weight_q_t, a_scale, weight_scale, out, _ = problem

    def call():
        return mm_mxfp8(
            a_q,
            weight_q_t,
            a_scale,
            weight_scale,
            out=out,
            out_dtype=torch.bfloat16,
            backend="cute-dsl",
            use_8x4_sf_layout=False,
        )

    return call


def _cosine_similarity(reference, output) -> float:
    import torch.nn.functional as F

    return F.cosine_similarity(
        reference.float().reshape(-1), output.float().reshape(-1), dim=0
    ).item()


def _tune_group(
    *,
    n: int,
    k: int,
    m_values: tuple[int, ...],
    exact: bool,
    cache_path: Path,
    reset_cache: bool,
    seed: int,
) -> float:
    import torch

    from flashinfer import autotune
    from flashinfer.autotuner import AutoTuner

    tuner = AutoTuner.get()
    tuner.clear_cache()
    torch.cuda.empty_cache()
    if reset_cache:
        cache_path.unlink(missing_ok=True)
    max_shape = Shape(max(m_values), n, k)
    problem = _make_problem(max_shape, seed)
    call = _make_call(problem)
    buckets = m_values if exact else None
    torch.cuda.synchronize()
    start = time.perf_counter()
    with autotune(True, cache=str(cache_path), tuning_buckets=buckets):
        call()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    del problem
    torch.cuda.empty_cache()
    return elapsed


def _benchmark_shape(
    *,
    shape: Shape,
    exact_m_values: tuple[int, ...] | None,
    cache_path: Path,
    seed: int,
    dry_run_iters: int,
    repeat_iters: int,
) -> dict[str, Any]:
    import numpy as np
    import torch

    from flashinfer import autotune
    from flashinfer.testing.utils import bench_gpu_time

    problem = _make_problem(shape, seed)
    call = _make_call(problem)
    with autotune(
        False,
        cache=str(cache_path),
        tuning_buckets=exact_m_values,
    ):
        output, selection = _capture_selection(call)
        cosine = _cosine_similarity(problem[-1], output)
        samples_ms = bench_gpu_time(
            call,
            dry_run_iters=dry_run_iters,
            repeat_iters=repeat_iters,
            enable_cupti=True,
            use_cuda_graph=True,
            cold_l2_cache=True,
        )
    row = {
        **asdict(shape),
        "median_ms": float(np.median(samples_ms)),
        "p10_ms": float(np.percentile(samples_ms, 10)),
        "p90_ms": float(np.percentile(samples_ms, 90)),
        "cosine_similarity": cosine,
        **selection,
    }
    del problem, output
    torch.cuda.empty_cache()
    return row


def _run_mode(
    *,
    mode: str,
    groups: dict[tuple[int, int], tuple[int, ...]],
    output_dir: Path,
    seed: int,
    dry_run_iters: int,
    repeat_iters: int,
) -> dict[str, Any]:
    exact = mode == "exact"
    rows = []
    tuning_time_s = 0.0
    cache_path = output_dir / f"{mode}_cache.json"
    for group_index, ((n, k), m_values) in enumerate(groups.items()):
        tuning_time_s += _tune_group(
            n=n,
            k=k,
            m_values=m_values,
            exact=exact,
            cache_path=cache_path,
            reset_cache=group_index == 0,
            seed=seed,
        )
        for m in m_values:
            rows.append(
                _benchmark_shape(
                    shape=Shape(m, n, k),
                    exact_m_values=m_values if exact else None,
                    cache_path=cache_path,
                    seed=seed,
                    dry_run_iters=dry_run_iters,
                    repeat_iters=repeat_iters,
                )
            )
    return {
        "mode": mode,
        "cache_path": str(cache_path),
        "tuning_time_s": tuning_time_s,
        "shapes": rows,
    }


def _summarize(baseline: dict[str, Any], exact: dict[str, Any]) -> list[dict[str, Any]]:
    baseline_by_shape = {
        (row["m"], row["n"], row["k"]): row for row in baseline["shapes"]
    }
    summary = []
    for row in exact["shapes"]:
        key = (row["m"], row["n"], row["k"])
        base = baseline_by_shape[key]
        summary.append(
            {
                "m": row["m"],
                "n": row["n"],
                "k": row["k"],
                "hybrid_ms": base["median_ms"],
                "exact_ms": row["median_ms"],
                "speedup": base["median_ms"] / row["median_ms"],
                "regret_pct": 100.0
                * (base["median_ms"] - row["median_ms"])
                / row["median_ms"],
                "same_tactic": base["tactic"] == row["tactic"],
                "hybrid_tactic": base["tactic"],
                "exact_tactic": row["tactic"],
                "cosine_similarity": row["cosine_similarity"],
            }
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shapes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--dry-run-iters", type=int, default=10)
    parser.add_argument("--repeat-iters", type=int, default=30)
    args = parser.parse_args()

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires a CUDA GPU")

    shapes = load_shapes(args.shapes)
    groups = group_shapes(shapes)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline = _run_mode(
        mode="hybrid",
        groups=groups,
        output_dir=args.output_dir,
        seed=args.seed,
        dry_run_iters=args.dry_run_iters,
        repeat_iters=args.repeat_iters,
    )
    exact = _run_mode(
        mode="exact",
        groups=groups,
        output_dir=args.output_dir,
        seed=args.seed,
        dry_run_iters=args.dry_run_iters,
        repeat_iters=args.repeat_iters,
    )
    summary = _summarize(baseline, exact)
    report = {
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "shapes_file": str(args.shapes),
        "baseline": baseline,
        "exact": exact,
        "summary": summary,
        "median_speedup": statistics.median(row["speedup"] for row in summary),
        "max_regret_pct": max(row["regret_pct"] for row in summary),
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")

    print("M,N,K,hybrid_ms,exact_ms,speedup,regret_pct,same_tactic")
    for row in summary:
        print(
            f"{row['m']},{row['n']},{row['k']},"
            f"{row['hybrid_ms']:.6f},{row['exact_ms']:.6f},"
            f"{row['speedup']:.4f},{row['regret_pct']:.2f},"
            f"{row['same_tactic']}"
        )
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
