import pytest

from benchmarks.bench_mxfp8_backend_tactic_oracle import (
    _aggregate_candidate_rounds,
    _deduplicate_tactics,
    _summarize_shape,
)
from benchmarks.bench_cutedsl_mxfp8_serving_shapes import Shape


def test_deduplicate_tactics_preserves_order_for_nested_tactics():
    assert _deduplicate_tactics(
        [
            ((128, 32), (1, 1), True, False, 1),
            ((128, 32), (1, 1), True, False, 1),
            ((128, 64), (1, 1), False, False, 1),
        ]
    ) == [
        ((128, 32), (1, 1), True, False, 1),
        ((128, 64), (1, 1), False, False, 1),
    ]


def test_aggregate_candidate_rounds_combines_samples_and_worst_cosine():
    result = _aggregate_candidate_rounds(
        tactic=7,
        rows=[
            {"samples_ms": [2.0, 1.0], "cosine_similarity": 0.999},
            {"samples_ms": [4.0, 3.0], "cosine_similarity": 0.998},
        ],
    )

    assert result["tactic"] == 7
    assert result["median_ms"] == 2.5
    assert result["samples_ms"] == [2.0, 1.0, 4.0, 3.0]
    assert result["cosine_similarity"] == 0.998


def test_summarize_shape_reports_oracle_speedup_over_selected():
    result = _summarize_shape(
        shape=Shape(128, 2304, 8192),
        selected_tactic=11,
        candidates=[
            {"tactic": 11, "median_ms": 12.0, "cosine_similarity": 0.999},
            {"tactic": 12, "median_ms": 10.0, "cosine_similarity": 0.998},
        ],
        min_cosine=0.98,
    )

    assert result["selected_ms"] == 12.0
    assert result["oracle_ms"] == 10.0
    assert result["oracle_tactic"] == 12
    assert result["speedup"] == pytest.approx(1.2)
    assert result["regret_pct"] == pytest.approx(20.0)


def test_summarize_shape_rejects_missing_selected_tactic():
    with pytest.raises(RuntimeError, match="Selected tactic was not measured"):
        _summarize_shape(
            shape=Shape(128, 2304, 8192),
            selected_tactic=11,
            candidates=[{"tactic": 12, "median_ms": 10.0, "cosine_similarity": 0.999}],
            min_cosine=0.98,
        )


def test_summarize_shape_ignores_incorrect_candidate():
    result = _summarize_shape(
        shape=Shape(128, 2304, 8192),
        selected_tactic=((128, 32), (1, 1), True, False, 1),
        candidates=[
            {
                "tactic": [[128, 32], [1, 1], True, False, 1],
                "median_ms": 12.0,
                "cosine_similarity": 0.999,
            },
            {
                "tactic": [[128, 64], [1, 1], False, False, 1],
                "median_ms": 8.0,
                "cosine_similarity": 0.50,
            },
        ],
        min_cosine=0.98,
    )

    assert result["oracle_ms"] == 12.0
    assert result["speedup"] == pytest.approx(1.0)
