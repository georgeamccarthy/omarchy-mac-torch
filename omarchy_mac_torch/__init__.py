"""PyTorch on Apple Silicon GPUs under Linux (Omarchy), via the Honeykrisp Vulkan driver. See README.md.

The default backend is native C++ (omarchy_mac_torch/_native*.so, built by tools/build_native.py). Set
OMARCHY_MAC_TORCH_BACKEND=python to use the original Python/wgpu-py backend instead (also the fallback when the
native library hasn't been built)."""

import collections as _collections
import os as _os
import pathlib as _pathlib

_use_native = (any(_pathlib.Path(__file__).parent.glob("_native*.so"))
           and _os.environ.get("OMARCHY_MAC_TORCH_BACKEND", "native") == "native")

if _use_native:
    from omarchy_mac_torch import native as _backend  # registers the "omarchy_mac_torch" device (C++)
    from omarchy_mac_torch.conv import tune_cpu
    from omarchy_mac_torch.native import set_queue_depth, synchronize

    from omarchy_mac_torch.native import (FALLBACKS, accelerate, empty_cache, memory_allocated, memory_reserved)

    def gpu():
        raise RuntimeError("gpu() belongs to the Python backend; set OMARCHY_MAC_TORCH_BACKEND=python")

    __all__ = ["FALLBACKS", "accelerate", "empty_cache", "memory_allocated", "memory_reserved", "set_queue_depth",
               "synchronize", "tune_cpu"]
else:
    from omarchy_mac_torch import device as _device  # registers the "omarchy_mac_torch" device (Python backend)
    from omarchy_mac_torch.conv import FusedConv2d, accelerate, check, gpu, set_queue_depth, tune_cpu
    from omarchy_mac_torch.ops import FALLBACKS

    __all__ = ["FusedConv2d", "FALLBACKS", "accelerate", "check", "gpu", "set_queue_depth", "tune_cpu"]
