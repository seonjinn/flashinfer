#!/usr/bin/env python3

from flashinfer.gemm.gemm_base import get_trtllm_gemm_module


module = get_trtllm_gemm_module()
print(module.trtllm_mxfp8_gemm_tactics(4, 2304, 8192, True))
