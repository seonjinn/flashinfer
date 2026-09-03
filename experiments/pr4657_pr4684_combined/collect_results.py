#!/usr/bin/env python3

import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def read_source(root: Path) -> dict[tuple[str, str], float]:
    values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for path in sorted((root / "raw").glob("repetition-*.csv")):
        with path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                tags = dict(item.split("=", 1) for item in row["case_tag"].split(";"))
                values[(tags["mode"], tags["shape"])].append(
                    float(row["median_time"])
                )
    return {key: statistics.median(samples) for key, samples in values.items()}


def geomean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def main() -> None:
    root = Path(sys.argv[1])
    pr4657 = read_source(root / "pr4657")
    combined = read_source(root / "combined")
    shapes = sorted(shape for mode, shape in pr4657 if mode == "fixed-8x4")
    arms = {
        "baseline": [pr4657[("fixed-8x4", shape)] for shape in shapes],
        "pr4657": [pr4657[("adaptive", shape)] for shape in shapes],
        "pr4684": [combined[("fixed-8x4", shape)] for shape in shapes],
        "combined": [combined[("adaptive", shape)] for shape in shapes],
    }
    baseline = arms["baseline"]
    rows = []
    for arm, latencies in arms.items():
        speedup = geomean([base / value for base, value in zip(baseline, latencies)])
        rows.append((arm, geomean(latencies), speedup, (speedup - 1.0) * 100.0))

    with (root / "summary.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("arm", "geomean_latency_ms", "speedup_x", "throughput_change_pct"))
        writer.writerows(rows)

    incremental_4657 = geomean(
        [
            combined[("fixed-8x4", shape)] / combined[("adaptive", shape)]
            for shape in shapes
        ]
    )
    combined_vs_4657 = geomean(
        [
            pr4657[("adaptive", shape)] / combined[("adaptive", shape)]
            for shape in shapes
        ]
    )
    with (root / "summary.md").open("w") as stream:
        stream.write("# PR 4657 + PR 4684 combined kernel sweep\n\n")
        stream.write(f"Validated shapes: {len(shapes)}; four repetitions per shape.\n\n")
        stream.write("| Arm | Geomean latency (ms) | Speedup vs baseline | Throughput change |\n")
        stream.write("|---|---:|---:|---:|\n")
        for arm, latency, speedup, change in rows:
            stream.write(f"| {arm} | {latency:.6f} | {speedup:.4f}x | {change:+.2f}% |\n")
        stream.write("\n")
        stream.write(
            f"Incremental dynamic-layout gain after PR 4684: {incremental_4657:.4f}x "
            f"({(incremental_4657 - 1.0) * 100.0:+.2f}%).\n\n"
        )
        stream.write(
            f"Combined dynamic path versus PR 4657-only dynamic path: "
            f"{combined_vs_4657:.4f}x ({(combined_vs_4657 - 1.0) * 100.0:+.2f}%).\n"
        )


if __name__ == "__main__":
    main()
