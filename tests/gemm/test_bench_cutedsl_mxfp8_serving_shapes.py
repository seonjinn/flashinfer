import csv

import pytest

from benchmarks.bench_cutedsl_mxfp8_serving_shapes import (
    Shape,
    _aggregate_rounds,
    group_shapes,
    load_shapes,
)


def test_load_shapes_normalizes_headers_and_deduplicates(tmp_path):
    path = tmp_path / "shapes.csv"
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["M", "N", "K"])
        writer.writerow([128, 2304, 8192])
        writer.writerow([16, 2304, 8192])
        writer.writerow([128, 2304, 8192])

    assert load_shapes(path) == [
        Shape(m=16, n=2304, k=8192),
        Shape(m=128, n=2304, k=8192),
    ]


def test_load_shapes_rejects_invalid_mxfp8_dimensions(tmp_path):
    path = tmp_path / "shapes.csv"
    path.write_text("M,N,K\n1,2304,8191\n")

    with pytest.raises(ValueError, match="K must be divisible by 32"):
        load_shapes(path)


def test_group_shapes_collects_exact_m_values_per_projection():
    shapes = [
        Shape(m=128, n=8192, k=2560),
        Shape(m=16, n=2304, k=8192),
        Shape(m=128, n=2304, k=8192),
    ]

    assert group_shapes(shapes) == {
        (2304, 8192): (16, 128),
        (8192, 2560): (128,),
    }


def test_aggregate_rounds_combines_samples_and_keeps_worst_cosine():
    shape = Shape(m=33, n=2304, k=8192)
    rows = [
        {
            "samples_ms": [1.0, 2.0],
            "cosine_similarity": 0.999,
            "runner": "CuteDSLMxfp8GemmRunner",
            "tactic": [[128, 32], [1, 1], True, False, 2],
        },
        {
            "samples_ms": [3.0, 4.0],
            "cosine_similarity": 0.998,
            "runner": "CuteDSLMxfp8GemmRunner",
            "tactic": [[128, 32], [1, 1], True, False, 2],
        },
    ]

    result = _aggregate_rounds(shape, rows)

    assert result["median_ms"] == 2.5
    assert result["samples_ms"] == [1.0, 2.0, 3.0, 4.0]
    assert result["cosine_similarity"] == 0.998


def test_aggregate_rounds_rejects_unstable_tactic_selection():
    shape = Shape(m=33, n=2304, k=8192)
    rows = [
        {
            "samples_ms": [1.0],
            "cosine_similarity": 0.999,
            "runner": "CuteDSLMxfp8GemmRunner",
            "tactic": [[128, 32], [1, 1], True, False, 2],
        },
        {
            "samples_ms": [1.0],
            "cosine_similarity": 0.999,
            "runner": "CuteDSLMxfp8GemmRunner",
            "tactic": [[128, 64], [1, 1], True, False, 1],
        },
    ]

    with pytest.raises(RuntimeError, match="Selected tactic changed"):
        _aggregate_rounds(shape, rows)
