import json
from collections.abc import Generator
from pathlib import Path

import pytest
import torch

from flashinfer import (
    SfLayout,
    autotune,
    mm_mxfp8_dynamic_quant,
    mxfp8_quantize,
    shuffle_matrix_a,
    shuffle_matrix_sf_a,
)
from flashinfer.autotuner import AutoTuner, OptimizationProfile
from flashinfer.gemm import gemm_base
from flashinfer.utils import get_compute_capability


_MIN_COSINE_SIMILARITY = 0.98


def _skip_if_trtllm_dynamic_quant_unsupported() -> None:
    if not torch.cuda.is_available():
        pytest.skip("TRTLLM MXFP8 dynamic quantization requires CUDA")

    capability = get_compute_capability(torch.device("cuda"))
    if capability not in {(10, 0), (10, 3), (10, 7)}:
        pytest.skip("TRTLLM MXFP8 dynamic quantization requires SM100, SM103, or SM107")


@pytest.fixture(autouse=True)
def _isolate_autotuner() -> Generator[None, None, None]:
    AutoTuner._instance = None
    try:
        _skip_if_trtllm_dynamic_quant_unsupported()
        yield
    finally:
        AutoTuner._instance = None


def _prepare_trtllm_weight(
    n: int, k: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    weight = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
    weight_q, weight_sf = mxfp8_quantize(
        weight,
        sf_swizzle_layout=SfLayout.layout_linear,
    )
    weight_q = shuffle_matrix_a(weight_q, 128).reshape(n, k)
    weight_sf = shuffle_matrix_sf_a(
        weight_sf.reshape(n, k // 32),
        128,
        num_elts_per_sf=32,
    ).reshape(-1)
    return weight, weight_q.T, weight_sf


def _cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(
        a.float().flatten(),
        b.float().flatten(),
        dim=0,
    ).item()


def test_mm_mxfp8_dynamic_quant_rejects_non_bf16_activation() -> None:
    a = torch.empty((4, 4096), device="cuda", dtype=torch.float16)
    _, b, b_sf = _prepare_trtllm_weight(2688, 4096)
    with pytest.raises(ValueError, match="a must be a bfloat16 tensor"):
        mm_mxfp8_dynamic_quant(a, b, b_sf)


def test_mm_mxfp8_dynamic_quant_rejects_unsupported_backend() -> None:
    a = torch.empty((4, 4096), device="cuda", dtype=torch.bfloat16)
    _, b, b_sf = _prepare_trtllm_weight(2688, 4096)
    with pytest.raises(ValueError, match="backend must be 'trtllm'"):
        mm_mxfp8_dynamic_quant(a, b, b_sf, backend="cutlass")


def test_mm_mxfp8_dynamic_quant_rejects_n_below_128() -> None:
    a = torch.empty((4, 256), device="cuda", dtype=torch.bfloat16)
    b = torch.empty((256, 64), device="cuda", dtype=torch.float8_e4m3fn)
    b_sf = torch.empty((0,), device="cuda", dtype=torch.uint8)

    with pytest.raises(ValueError, match="N >= 128"):
        mm_mxfp8_dynamic_quant(a, b, b_sf)


def test_mm_mxfp8_dynamic_quant_rejects_zero_k() -> None:
    a = torch.empty((4, 0), device="cuda", dtype=torch.bfloat16)
    b = torch.empty((0, 128), device="cuda", dtype=torch.float8_e4m3fn)
    b_sf = torch.empty((0,), device="cuda", dtype=torch.uint8)

    with pytest.raises(ValueError, match="K must be positive"):
        mm_mxfp8_dynamic_quant(a, b, b_sf)


@pytest.mark.parametrize("m", [3, 16, 32, 33, 128])
@pytest.mark.parametrize("auto_tuning", [False, True])
def test_mm_mxfp8_dynamic_quant_matches_bf16(m: int, auto_tuning: bool) -> None:
    torch.manual_seed(0)
    a = torch.randn((m, 4096), device="cuda", dtype=torch.bfloat16)
    weight, b, b_sf = _prepare_trtllm_weight(2688, 4096)
    reference = a @ weight.T

    with autotune(auto_tuning):
        actual = mm_mxfp8_dynamic_quant(a, b, b_sf)

    assert _cosine_similarity(reference, actual) > _MIN_COSINE_SIMILARITY


class _RecordingTuner:
    def __init__(self) -> None:
        self.extras: list[tuple[bool]] = []
        self.tactics: list[list[int]] = []

    def choose_one(self, custom_op, runners, tuning_config, inputs):
        profile = OptimizationProfile(
            shapes=[
                list(value.shape) if isinstance(value, torch.Tensor) else []
                for value in inputs
            ],
            tensor_initializers=[None] * len(inputs),
        )
        self.extras = [runner.get_cache_key_extras(inputs) for runner in runners]
        self.tactics = [runner.get_valid_tactics(inputs, profile) for runner in runners]
        return runners[0], -1


def test_mm_mxfp8_dynamic_quant_offers_both_layouts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = _RecordingTuner()
    monkeypatch.setattr(AutoTuner, "get", classmethod(lambda cls: recorder))

    a = torch.randn((4, 4096), device="cuda", dtype=torch.bfloat16)
    _, b, b_sf = _prepare_trtllm_weight(2688, 4096)
    quantized_layouts: list[SfLayout] = []
    real_quantize = gemm_base.mxfp8_quantize

    def recording_quantize(
        input: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        quantized_layouts.append(kwargs["sf_swizzle_layout"])
        return real_quantize(input, **kwargs)

    monkeypatch.setattr(gemm_base, "mxfp8_quantize", recording_quantize)

    mm_mxfp8_dynamic_quant(a, b, b_sf)

    assert recorder.extras == [(True,), (False,)]
    assert all(recorder.tactics)
    assert quantized_layouts == [SfLayout.layout_8x4]


def test_mm_mxfp8_dynamic_quant_cache_round_trip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_path = tmp_path / "dynamic_quant.json"
    torch.manual_seed(0)
    a = torch.randn((4, 4096), device="cuda", dtype=torch.bfloat16)
    _, b, b_sf = _prepare_trtllm_weight(2688, 4096)

    with autotune(True, cache=str(cache_path)):
        tuned = mm_mxfp8_dynamic_quant(a, b, b_sf)

    AutoTuner._instance = None
    cache_hits: list[bool] = []
    real_search_cache = AutoTuner.search_cache

    def recording_search_cache(
        self,
        custom_op,
        runners,
        input_shapes,
        tuning_config,
        inputs=None,
    ):
        result = real_search_cache(
            self,
            custom_op,
            runners,
            input_shapes,
            tuning_config,
            inputs=inputs,
        )
        if custom_op == "mxfp8_dynamic_quant_gemm":
            cache_hits.append(result[0])
        return result

    monkeypatch.setattr(AutoTuner, "search_cache", recording_search_cache)
    with autotune(False, cache=str(cache_path)):
        cached = mm_mxfp8_dynamic_quant(a, b, b_sf)

    payload = json.loads(cache_path.read_text())
    assert any("mxfp8_dynamic_quant_gemm" in key for key in payload)
    assert cache_hits == [True]
    torch.testing.assert_close(cached, tuned, rtol=0, atol=0)


def test_mm_mxfp8_dynamic_quant_cuda_graph_replay() -> None:
    torch.manual_seed(0)
    static_a = torch.randn((4, 4096), device="cuda", dtype=torch.bfloat16)
    _, b, b_sf = _prepare_trtllm_weight(2688, 4096)
    static_out = torch.empty((4, 2688), device="cuda", dtype=torch.bfloat16)

    with autotune(True):
        mm_mxfp8_dynamic_quant(static_a, b, b_sf, out=static_out)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_out = mm_mxfp8_dynamic_quant(static_a, b, b_sf, out=static_out)

    for seed in (1, 2):
        torch.manual_seed(seed)
        next_a = torch.randn_like(static_a)
        static_a.copy_(next_a)
        graph.replay()
        replayed = graph_out.clone()
        eager = mm_mxfp8_dynamic_quant(next_a, b, b_sf)
        assert _cosine_similarity(replayed, eager) > _MIN_COSINE_SIMILARITY
