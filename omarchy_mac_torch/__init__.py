"""PyTorch on Apple Silicon GPUs under Linux (Omarchy), via the Honeykrisp Vulkan driver. See README.md.

    import omarchy_mac_torch            # registers the "omarchy_mac_torch" device
    model = model.to("omarchy_mac_torch")
    omarchy_mac_torch.accelerate(model)  # optional: fused conv/BN/ReLU/pool stacks
"""

from omarchy_mac_torch import native as _native  # registers the device (C++ backend)
from omarchy_mac_torch import optim as _optim  # noqa: F401  (fused optimizers by default)
from omarchy_mac_torch import debug
from omarchy_mac_torch import compile as _compile  # noqa: F401  (torch.compile(model, backend="omarchy_mac_torch"))
from omarchy_mac_torch.native import (DEVICE, FALLBACKS, empty_cache, memory_allocated, memory_reserved,
                                      set_queue_depth, synchronize, validate_kernels)
from omarchy_mac_torch.stacks import FusedSequential, accelerate
from omarchy_mac_torch.utils import tune_cpu

__version__ = "0.2.0"
__all__ = ["DEVICE", "FALLBACKS", "FusedSequential", "accelerate", "debug", "empty_cache", "memory_allocated",
           "memory_reserved", "set_queue_depth", "synchronize", "tune_cpu", "validate_kernels"]
