import argparse
import sys
from pathlib import Path

import pytest


BENCHMARK_ROOT = Path(__file__).resolve().parents[2] / "benchmarks"
sys.path.insert(0, str(BENCHMARK_ROOT))

from routines.gemm import parse_gemm_args  # noqa: E402


def test_mm_mxfp8_accepts_dynamic_quant_flag() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--routine")
    args = parse_gemm_args(
        [
            "--routine",
            "mm_mxfp8",
            "--m",
            "4",
            "--n",
            "2688",
            "--k",
            "4096",
            "--backends",
            "trtllm",
            "--dynamic_quant",
            "--dynamic_quant_layout",
            "auto",
        ],
        parser,
    )
    assert args.dynamic_quant is True
    assert args.dynamic_quant_layout == "auto"


@pytest.mark.parametrize(
    ("dynamic_quant_layout", "backend"),
    [("auto", "cute-dsl"), ("8x4", "cute-dsl"), ("128x4", "cutlass")],
)
def test_mm_mxfp8_rejects_dynamic_quant_layout_backend_mismatch(
    dynamic_quant_layout: str,
    backend: str,
) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--routine")

    with pytest.raises(
        ValueError,
        match=rf"--dynamic_quant_layout {dynamic_quant_layout} supports only",
    ):
        parse_gemm_args(
            [
                "--routine",
                "mm_mxfp8",
                "--m",
                "4",
                "--n",
                "2688",
                "--k",
                "4096",
                "--backends",
                backend,
                "--dynamic_quant",
                "--dynamic_quant_layout",
                dynamic_quant_layout,
            ],
            parser,
        )
