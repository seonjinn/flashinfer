import importlib.metadata
import platform

import flashinfer
import torch
import vllm


def main() -> None:
    assert torch.cuda.is_available()
    print(f"python={platform.python_version()}")
    print(f"vllm={vllm.__version__}")
    print(f"torch={torch.__version__}")
    print(f"cuda={torch.version.cuda}")
    print(f"flashinfer={getattr(flashinfer, '__version__', 'unknown')}")
    print(f"flashinfer_package={importlib.metadata.version('flashinfer-python')}")
    print(f"cutlass_dsl={importlib.metadata.version('nvidia-cutlass-dsl')}")
    print(f"device={torch.cuda.get_device_name(0)}")
    print(f"capability={torch.cuda.get_device_capability(0)}")


if __name__ == "__main__":
    main()
