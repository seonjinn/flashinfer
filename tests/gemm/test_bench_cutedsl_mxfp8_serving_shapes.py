import csv

import pytest

from benchmarks.bench_cutedsl_mxfp8_serving_shapes import (
    Shape,
    _aggregate_rounds,
    _cache_path,
    _find_jit_dependency_root,
    _reuse_mode,
    _validate_backend_layout,
    group_shapes,
    load_shapes,
)


def test_find_jit_dependency_root_prefers_initialized_source_tree(tmp_path):
    source_root = tmp_path / "source"
    for path in (
        "cutlass/include",
        "cutlass/tools/util/include",
        "spdlog/include",
        "cccl/cub",
        "cccl/libcudacxx/include",
        "cccl/thrust",
    ):
        (source_root / "3rdparty" / path).mkdir(parents=True)
    packaged_root = tmp_path / "site" / "flashinfer" / "data"
    for path in (
        "cutlass/include",
        "cutlass/tools/util/include",
        "spdlog/include",
        "cccl/cub",
        "cccl/libcudacxx/include",
        "cccl/thrust",
    ):
        (packaged_root / path).mkdir(parents=True)

    assert _find_jit_dependency_root(source_root, [tmp_path / "site"]) == (
        source_root / "3rdparty"
    )


def test_find_jit_dependency_root_falls_back_to_installed_package(tmp_path):
    source_root = tmp_path / "source"
    packaged_root = tmp_path / "site" / "flashinfer" / "data"
    for path in (
        "cutlass/include",
        "cutlass/tools/util/include",
        "spdlog/include",
        "cccl/cub",
        "cccl/libcudacxx/include",
        "cccl/thrust",
    ):
        (packaged_root / path).mkdir(parents=True)

    assert _find_jit_dependency_root(source_root, [tmp_path / "site"]) == packaged_root


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


def test_validate_backend_layout_accepts_supported_pairs():
    _validate_backend_layout("cute-dsl", "128x4")
    _validate_backend_layout("cutlass", "128x4")
    _validate_backend_layout("trtllm", "8x4")
    _validate_backend_layout("trtllm", "128x4")


def test_validate_backend_layout_rejects_cute_dsl_8x4():
    with pytest.raises(ValueError, match="CuTeDSL requires 128x4"):
        _validate_backend_layout("cute-dsl", "8x4")


def test_validate_backend_layout_rejects_cutlass_8x4():
    with pytest.raises(ValueError, match="CUTLASS requires 128x4"):
        _validate_backend_layout("cutlass", "8x4")


def test_cache_path_separates_backend_and_layout(tmp_path):
    assert _cache_path(tmp_path, "trtllm", "8x4", "exact") == (
        tmp_path / "trtllm_8x4_exact_cache.json"
    )
    assert _cache_path(tmp_path, "cutlass", "128x4", "exact") == (
        tmp_path / "cutlass_128x4_exact_cache.json"
    )


def test_reuse_mode_requires_and_selects_native_cache(tmp_path):
    with pytest.raises(FileNotFoundError, match="Missing exact autotuner cache"):
        _reuse_mode("exact", tmp_path, "cute-dsl", "128x4")

    cache_path = tmp_path / "cute-dsl_128x4_exact_cache.json"
    cache_path.write_text("{}\n")

    assert _reuse_mode("exact", tmp_path, "cute-dsl", "128x4") == {
        "mode": "exact",
        "cache_path": str(cache_path),
        "cache_source": "reused",
        "tuning_time_s": 0.0,
        "shapes": [],
    }
