#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1)
    return ordered[index]


def aggregate(regrets: list[float]) -> dict[str, Any]:
    return {
        "p95_regret_pct": percentile(regrets, 0.95),
        "max_regret_pct": max(regrets),
        "shapes_over_5pct": sum(value > 5.0 for value in regrets),
        "shapes_within_1pct": sum(value <= 1.0 for value in regrets),
        "shape_count": len(regrets),
    }


def main() -> None:
    args = parse_args()
    rows = []
    for path in sorted(args.input_dir.glob("*.jsonl")):
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line)
    if not rows:
        raise RuntimeError(f"no validation rows found under {args.input_dir}")

    by_shape: dict[tuple[int, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_shape[(row["m"], row["n"], row["k"])].append(row)

    one_pass_regrets = []
    top3x3_regrets = []
    for shape_rows in by_shape.values():
        one_pass_regrets.append(
            statistics.median(row["one_pass_regret_pct"] for row in shape_rows)
        )
        top3x3_regrets.append(
            statistics.median(row["top3x3_regret_pct"] for row in shape_rows)
        )

    summary = {
        "row_count": len(rows),
        "shape_count": len(by_shape),
        "one_pass": aggregate(one_pass_regrets),
        "top3x3": aggregate(top3x3_regrets),
        "timing": {
            "median_one_pass_wall_s": statistics.median(
                row["one_pass_wall_s"] for row in rows
            ),
            "median_top3x3_wall_s": statistics.median(
                row["top3x3_wall_s"] for row in rows
            ),
            "median_refinement_overhead_pct": statistics.median(
                row["refinement_overhead_pct"] for row in rows
            ),
            "median_one_pass_profile_calls": statistics.median(
                row["one_pass_profile_calls"] for row in rows
            ),
            "median_top3x3_profile_calls": statistics.median(
                row["top3x3_profile_calls"] for row in rows
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
