"""torch.optim's SGD, Adam and AdamW default to the fused implementation when every parameter is on the device.

PyTorch has no hook for a backend to prefer its fused optimizers (left alone, they pick the per-tensor foreach path,
which works here but issues several dispatches per parameter), so the constructors are wrapped. Passing fused= or
foreach= explicitly is always respected; OMARCHY_MAC_TORCH_FUSED_OPTIM=0 turns the default off."""

import functools
import os

import torch

from omarchy_mac_torch.native import DEVICE


def _wrap(cls):
    init = cls.__init__
    if getattr(init, "_omt", False):
        return

    @functools.wraps(init)
    def __init__(self, params, *args, **kwargs):
        params = list(params)
        flat = [p for g in params for p in (g["params"] if isinstance(g, dict) else [g])]
        if (flat and all(p.device.type == DEVICE for p in flat) and kwargs.get("fused") is None
                and kwargs.get("foreach") is None and not kwargs.get("differentiable")
                and not isinstance(kwargs.get("lr", 0.0), torch.Tensor)):
            kwargs["fused"] = True
        init(self, params, *args, **kwargs)

    __init__._omt = True
    cls.__init__ = __init__


if os.environ.get("OMARCHY_MAC_TORCH_FUSED_OPTIM", "1") != "0":
    for _cls in (torch.optim.SGD, torch.optim.Adam, torch.optim.AdamW):
        _wrap(_cls)
