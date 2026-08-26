import pytest

from benchmarks.bench_mxfp8_backend_tactic_oracle import (
    _aggregate_candidate_rounds,
    _deduplicate_tactics,
    _load_selected_tactics,
    _profile_from_shapes,
    _summarize_shape,
)
from benchmarks.bench_cutedsl_mxfp8_serving_shapes import Shape


def test_load_selected_tactics_preserves_serving_runner_and_nested_tactic(tmp_path):
    path = tmp_path / "observed.csv"
    path.write_text(
        "m,n,k,runner,selected_tactic\n"
        '128,2304,8192,CuteRunner,"[[128,32],[1,1],true,false,1]"\n'
    )

    selected = _load_selected_tactics(path)

    assert selected[Shape(128, 2304, 8192)] == {
        "runner": "CuteRunner",
        "tactic": ((128, 32), (1, 1), True, False, 1),
    }


def test_load_selected_tactics_rejects_conflicting_execution_signatures(tmp_path):
    path = tmp_path / "observed.csv"
    path.write_text(
        "m,n,k,runner,selected_tactic\n"
        "128,2304,8192,CuteRunner,7\n"
        "128,2304,8192,OtherRunner,9\n"
    )

    with pytest.raises(ValueError, match="conflicting selected tactics"):
        _load_selected_tactics(path)


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
            {
                "samples_ms": [2.0, 1.0],
                "cosine_similarity": 0.999,
                "finite": True,
                "matches_selected": True,
                "max_abs_diff_from_selected": 0.01,
            },
            {
                "samples_ms": [4.0, 3.0],
                "cosine_similarity": 0.998,
                "finite": True,
                "matches_selected": True,
                "max_abs_diff_from_selected": 0.02,
            },
        ],
    )

    assert result["tactic"] == 7
    assert result["median_ms"] == 2.5
    assert result["samples_ms"] == [2.0, 1.0, 4.0, 3.0]
    assert result["cosine_similarity"] == 0.998
    assert result["matches_selected"] is True
    assert result["max_abs_diff_from_selected"] == 0.02


def test_profile_from_shapes_builds_static_concrete_profile():
    class FakeStaticDim:
        def __init__(self, value):
            self.val = value

    class FakeProfile:
        def __init__(self, shapes, tensor_initializers):
            self.shapes = shapes
            self.tensor_initializers = tensor_initializers

    profile = _profile_from_shapes(
        ((3, 8192), (8192, 2304), (0,)),
        profile_type=FakeProfile,
        static_dim_type=FakeStaticDim,
    )

    assert [[dim.val for dim in shape] for shape in profile.shapes] == [
        [3, 8192],
        [8192, 2304],
        [0],
    ]
    assert profile.tensor_initializers == [None, None, None]


def test_summarize_shape_reports_oracle_speedup_over_selected():
    result = _summarize_shape(
        shape=Shape(128, 2304, 8192),
        runner_name="CuteRunner",
        selected_tactic=11,
        candidates=[
            {
                "tactic": 11,
                "median_ms": 12.0,
                "cosine_similarity": 0.999,
                "finite": True,
                "matches_selected": True,
            },
            {
                "tactic": 12,
                "median_ms": 10.0,
                "cosine_similarity": 0.998,
                "finite": True,
                "matches_selected": True,
            },
        ],
        min_cosine=0.98,
    )

    assert result["selected_ms"] == 12.0
    assert result["runner"] == "CuteRunner"
    assert result["oracle_ms"] == 10.0
    assert result["oracle_tactic"] == 12
    assert result["speedup"] == pytest.approx(1.2)
    assert result["regret_pct"] == pytest.approx(20.0)


def test_summarize_shape_rejects_missing_selected_tactic():
    with pytest.raises(RuntimeError, match="Selected tactic was not measured"):
        _summarize_shape(
            shape=Shape(128, 2304, 8192),
            runner_name="CuteRunner",
            selected_tactic=11,
            candidates=[{"tactic": 12, "median_ms": 10.0, "cosine_similarity": 0.999}],
            min_cosine=0.98,
        )


def test_summarize_shape_ignores_incorrect_candidate():
    result = _summarize_shape(
        shape=Shape(128, 2304, 8192),
        runner_name="CuteRunner",
        selected_tactic=((128, 32), (1, 1), True, False, 1),
        candidates=[
            {
                "tactic": [[128, 32], [1, 1], True, False, 1],
                "median_ms": 12.0,
                "cosine_similarity": 0.999,
                "finite": True,
                "matches_selected": True,
            },
            {
                "tactic": [[128, 64], [1, 1], False, False, 1],
                "median_ms": 8.0,
                "cosine_similarity": 0.50,
                "finite": True,
                "matches_selected": True,
            },
        ],
        min_cosine=0.98,
    )

    assert result["oracle_ms"] == 12.0
    assert result["speedup"] == pytest.approx(1.0)
