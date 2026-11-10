"""The "omarchy_mac_torch" device: registers it with PyTorch so `x.to("omarchy_mac_torch")` and
`model.to("omarchy_mac_torch")` work. Tensors, kernels and the CPU fallback are in
tensor.py and ops.py; csrc/guard.cpp supplies the C++ device guard.
"""

import ctypes
import os
import pathlib
import shutil
import sys
import types

import torch
from torch.utils.cpp_extension import load

_SO = pathlib.Path(__file__).parent / "_guard.so"
if _SO.exists():
    ctypes.CDLL(str(_SO))  # prebuilt: a build path containing spaces breaks the torch extension builder
else:
    os.environ["PATH"] = os.path.join(sys.prefix, "bin") + os.pathsep + os.environ["PATH"]  # for ninja
    (_SO.parent / "_build").mkdir(exist_ok=True)
    load("omt_guard", [str(pathlib.Path(__file__).parent.parent / "csrc" / "guard.cpp")],
         extra_cflags=["-O1"], is_python_module=False, build_directory=str(_SO.parent / "_build"))
    shutil.copy(_SO.parent / "_build" / "omt_guard.so", _SO)

torch.utils.rename_privateuse1_backend("omarchy_mac_torch")

from omarchy_mac_torch import ops  # noqa: E402
from omarchy_mac_torch.runtime import REC  # noqa: E402
from omarchy_mac_torch.tensor import OmtTensor, contiguous_strides, new  # noqa: E402

_mod = types.ModuleType("torch.omarchy_mac_torch")
_mod.is_available = lambda: True
_mod.device_count = lambda: 1
_mod.current_device = lambda: 0
_mod.set_device = lambda d: None
_mod._is_in_bad_fork = lambda: False
_mod.manual_seed_all = lambda s: None
_mod.get_amp_supported_dtype = lambda: [torch.float32]
_mod.synchronize = lambda device=None: REC.flush()
torch._register_device_module("omarchy_mac_torch", _mod)

_lib = torch.library.Library("aten", "IMPL")


def _empty(size, dtype=None, layout=None, device=None, pin_memory=None, memory_format=None):
    return new(size, dtype or torch.float32)


def _empty_strided(size, stride, dtype=None, layout=None, device=None, pin_memory=None):
    t = new(size, dtype or torch.float32)
    if tuple(stride) != contiguous_strides(tuple(size)):
        t = OmtTensor(t.st, tuple(size), tuple(stride), 0, t.dtype)
    return t


_lib.impl("empty.memory_format", _empty, "PrivateUse1")
_lib.impl("empty_strided", _empty_strided, "PrivateUse1")

# torch.optim.Adam uses the fused kernel (one op for all parameters) on the device.
_adam_init = torch.optim.Adam.__init__


def _init(self, params, *args, **kwargs):
    params = list(params)
    flat = [p for g in params for p in (g["params"] if isinstance(g, dict) else [g])]
    if flat and all(p.device.type == "omarchy_mac_torch" for p in flat) and kwargs.get("fused") is None \
            and kwargs.get("foreach") is None and not kwargs.get("differentiable"):
        kwargs["fused"] = True
    _adam_init(self, params, *args, **kwargs)


torch.optim.Adam.__init__ = _init

# Adam's step counters belong on the CPU: advancing them on the GPU costs a dispatch per parameter.
_init_group = torch.optim.Adam._init_group


def _init_group_cpu_steps(self, group, params_with_grad, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs,
                          state_steps):
    r = _init_group(self, group, params_with_grad, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps)
    for i, p in enumerate(params_with_grad):
        st = self.state[p]["step"]
        if st.device.type == "omarchy_mac_torch":
            self.state[p]["step"] = state_steps[i] = torch.tensor(float(st.item()), dtype=torch.float32)
    return r


torch.optim.Adam._init_group = _init_group_cpu_steps


# model.to("omarchy_mac_torch") also swaps in the fused GPU stacks.
_module_to = torch.nn.Module.to


def _to(self, *args, **kwargs):
    out = _module_to(self, *args, **kwargs)
    device = torch._C._nn._parse_to(*args, **kwargs)[0]
    if device is not None and device.type == "omarchy_mac_torch":
        from omarchy_mac_torch.conv import accelerate
        accelerate(out)
    return out


torch.nn.Module.to = _to
