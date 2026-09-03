#!/usr/bin/env python3

import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

Key = tuple[str, str, str, str, str]


def read_source(root: Path) -> dict[Key, float]:
    values: dict[Key, list[float]] = defaultdict(list)
    for path in sorted((root / "raw").glob("repetition-*.csv")):
        with path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                tags = dict(item.split("=", 1) for item in row["case_tag"].split(";"))
                key = (
                    tags["model"],
                    tags["topology"],
                    tags["projection"],
                    tags["mode"],
                    tags["shape"],
                )
                values[key].append(float(row["median_time"]))
    return {key: statistics.median(samples) for key, samples in values.items()}


def geomean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def summarize_model(
    model: str,
    topology: str,
    pr4657: dict[Key, float],
    combined: dict[Key, float],
) -> list[tuple[str, str, str, int, float, float, float]]:
    coordinates = sorted(
        (projection, shape)
        for key_model, key_topology, projection, mode, shape in pr4657
        if key_model == model and key_topology == topology and mode == "fixed-8x4"
    )
    arms = {
        "baseline": [
            pr4657[(model, topology, projection, "fixed-8x4", shape)]
            for projection, shape in coordinates
        ],
        "pr4657": [
            pr4657[(model, topology, projection, "adaptive", shape)]
            for projection, shape in coordinates
        ],
        "pr4684": [
            combined[(model, topology, projection, "fixed-8x4", shape)]
            for projection, shape in coordinates
        ],
        "combined": [
            combined[(model, topology, projection, "adaptive", shape)]
            for projection, shape in coordinates
        ],
    }
    baseline = arms["baseline"]
    rows = []
    for arm, latencies in arms.items():
        speedup = geomean([base / value for base, value in zip(baseline, latencies)])
        rows.append(
            (
                model,
                topology,
                arm,
                len(coordinates),
                geomean(latencies),
                speedup,
                (speedup - 1.0) * 100.0,
            )
        )
    return rows


def main() -> None:
    root = Path(sys.argv[1])
    pr4657 = read_source(root / "pr4657")
    combined = read_source(root / "combined")
    models = sorted({(model, topology) for model, topology, _, _, _ in pr4657})
    rows = [
        row
        for model, topology in models
        for row in summarize_model(model, topology, pr4657, combined)
    ]

    with (root / "summary.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "model",
                "topology",
                "arm",
                "model_shape_count",
                "geomean_latency_ms",
                "speedup_x",
                "throughput_change_pct",
            )
        )
        writer.writerows(rows)

    with (root / "per_shape.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "model",
                "topology",
                "projection",
                "shape",
                "baseline_ms",
                "pr4657_ms",
                "pr4684_ms",
                "combined_ms",
                "pr4657_speedup_x",
                "pr4684_speedup_x",
                "combined_speedup_x",
            )
        )
        coordinates = sorted(
            (model, topology, projection, shape)
            for model, topology, projection, mode, shape in pr4657
            if mode == "fixed-8x4"
        )
        for model, topology, projection, shape in coordinates:
            baseline = pr4657[(model, topology, projection, "fixed-8x4", shape)]
            pr4657_value = pr4657[(model, topology, projection, "adaptive", shape)]
            pr4684_value = combined[(model, topology, projection, "fixed-8x4", shape)]
            combined_value = combined[(model, topology, projection, "adaptive", shape)]
            writer.writerow(
                (
                    model,
                    topology,
                    projection,
                    shape,
                    baseline,
                    pr4657_value,
                    pr4684_value,
                    combined_value,
                    baseline / pr4657_value,
                    baseline / pr4684_value,
                    baseline / combined_value,
                )
            )

    with (root / "summary.md").open("w") as stream:
        stream.write("# PR 4657 + PR 4684 model-shape generalization\n\n")
        stream.write(
            "GB200 kernel sweep over TP4 dense-linear projection shapes. Each row "
            "uses M=1..16384 and four independent repetitions. Routed-expert GEMMs "
            "are excluded because they use the MoE backend rather than this API.\n\n"
        )
        stream.write(
            "| Model | Topology | Arm | Shape count | Geomean latency (ms) | "
            "Speedup vs baseline | Throughput change |\n"
        )
        stream.write("|---|---|---|---:|---:|---:|---:|\n")
        for model, topology, arm, count, latency, speedup, change in rows:
            stream.write(
                f"| {model} | {topology} | {arm} | {count} | {latency:.6f} | "
                f"{speedup:.4f}x | {change:+.2f}% |\n"
            )
        stream.write(
            "\nThese are equal-weight kernel results across model projection/M pairs, not "
            "model-level E2E throughput claims.\n"
        )


if __name__ == "__main__":
    main()
