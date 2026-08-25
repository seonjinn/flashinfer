import csv

import pytest

from benchmarks.bench_cutedsl_mxfp8_serving_shapes import (
    Shape,
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
