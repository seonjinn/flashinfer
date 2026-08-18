import pytest
import torch

from flashinfer import (
    SfLayout,
    mm_mxfp8_dynamic_quant,
    mxfp8_quantize,
    shuffle_matrix_a,
    shuffle_matrix_sf_a,
)


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
