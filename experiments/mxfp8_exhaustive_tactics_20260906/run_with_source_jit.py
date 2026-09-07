#!/usr/bin/env python3

import os
import runpy
import sys
from pathlib import Path

from flashinfer.jit import env as jit_env


def main() -> None:
    jit_env.FLASHINFER_CSRC_DIR = Path(os.environ["PATCH_CSRC"])
    jit_env.FLASHINFER_INCLUDE_DIR = Path(os.environ["PATCH_INCLUDE"])
    jit_env.FLASHINFER_AOT_DIR = Path(os.environ["EMPTY_AOT"])

    if sys.argv[1:2] == ["-m"]:
        module_name = sys.argv[2]
        sys.argv = sys.argv[2:]
        runpy.run_module(module_name, run_name="__main__", alter_sys=True)
        return

    script = sys.argv[1]
    sys.argv = sys.argv[1:]
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
