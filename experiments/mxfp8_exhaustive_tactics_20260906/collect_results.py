#!/usr/bin/env python3

import ast
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def geometric_mean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def read_latencies(root: Path) -> dict[tuple[str, str], float]:
    values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for path in sorted((root / "raw").glob("repetition-*.csv")):
        with path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                tags = dict(item.split("=", 1) for item in row["case_tag"].split(";"))
                values[(tags["mode"], tags["shape"])].append(float(row["median_time"]))
    return {key: statistics.median(samples) for key, samples in values.items()}


def read_selected_tactics(root: Path) -> dict[str, tuple[str, int]]:
    votes: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for path in sorted((root / "cache").glob("autotune-*.json")):
        payload: dict[str, Any] = json.loads(path.read_text())
        for key, value in payload.items():
            if key == "_metadata" or "mxfp8_dynamic_quant_gemm" not in key:
                continue
            parsed = ast.literal_eval(key)
            input_shapes = parsed[2]
            m, k = input_shapes[0]
            _, n = input_shapes[1]
            layout, tactic = value[1]
            votes[f"{m}x{n}x{k}"].append(("8x4" if layout else "128x4", int(tactic)))
    return {shape: statistics.mode(shape_votes) for shape, shape_votes in votes.items()}


def main() -> None:
    root = Path(sys.argv[1])
    control = read_latencies(root / "pruned")
    exhaustive = read_latencies(root / "exhaustive")
    control_tactics = read_selected_tactics(root / "pruned")
    exhaustive_tactics = read_selected_tactics(root / "exhaustive")
    shapes = sorted(shape for mode, shape in control if mode == "adaptive")

    rows = []
    for shape in shapes:
        pruned_ms = control[("adaptive", shape)]
        exhaustive_ms = exhaustive[("adaptive", shape)]
        fixed_8x4_ms = exhaustive[("fixed-8x4", shape)]
        fixed_128x4_ms = exhaustive[("fixed-128x4", shape)]
        rows.append(
            (
                shape,
                pruned_ms,
                exhaustive_ms,
                pruned_ms / exhaustive_ms,
                fixed_8x4_ms,
                fixed_128x4_ms,
                min(fixed_8x4_ms, fixed_128x4_ms) / exhaustive_ms,
                str(control_tactics.get(shape, ("missing", -1))),
                str(exhaustive_tactics.get(shape, ("missing", -1))),
            )
        )

    with (root / "per_shape.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "shape",
                "pruned_adaptive_ms",
                "exhaustive_adaptive_ms",
                "exhaustive_speedup_x",
                "fixed_8x4_ms",
                "fixed_128x4_ms",
                "exhaustive_vs_best_fixed_x",
                "pruned_selected",
                "exhaustive_selected",
            )
        )
        writer.writerows(rows)

    exhaustive_speedup = geometric_mean([float(row[3]) for row in rows])
    oracle_ratio = geometric_mean([float(row[6]) for row in rows])
    improved = sum(float(row[3]) > 1.0 for row in rows)
    changed = sum(row[7] != row[8] for row in rows)
    timing = {}
    for label in ("pruned", "exhaustive"):
        samples = [
            float(path.read_text())
            for path in sorted((root / label / "logs").glob("timing-*.txt"))
        ]
        timing[label] = statistics.median(samples)

    with (root / "summary.md").open("w") as stream:
        stream.write("# MXFP8 exhaustive tactic validation\n\n")
        stream.write(f"Shapes: {len(rows)}; repetitions: 4; GPU: GB200.\n\n")
        stream.write(
            f"Exhaustive versus pruned adaptive geomean speedup: {exhaustive_speedup:.4f}x "
            f"({(exhaustive_speedup - 1.0) * 100.0:+.2f}%).\n\n"
        )
        stream.write(
            f"Exhaustive versus per-shape best fixed layout: {oracle_ratio:.4f}x "
            f"({(oracle_ratio - 1.0) * 100.0:+.2f}%).\n\n"
        )
        stream.write(f"Improved shapes: {improved}/{len(rows)}.\n\n")
        stream.write(f"Selected layout/tactic changed: {changed}/{len(rows)}.\n\n")
        stream.write(
            f"Median sweep wall time: pruned {timing['pruned']:.1f}s; "
            f"exhaustive {timing['exhaustive']:.1f}s "
            f"({timing['exhaustive'] / timing['pruned']:.2f}x).\n"
        )


if __name__ == "__main__":
    main()
