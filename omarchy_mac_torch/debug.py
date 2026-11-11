"""Shadow mode: run every op on the GPU *and* on the CPU and report the first one whose results differ.

    with omarchy_mac_torch.debug.shadow():          # or OMARCHY_MAC_TORCH_SHADOW=1
        loss = model(x).sum(); loss.backward()

Each op's inputs are copied to the CPU before the GPU runs it (so in-place ops compare too), the op runs on both, and
outputs (and mutated inputs) are compared with torch.allclose. Random ops are skipped. It is slow: for debugging."""

import contextlib
import os

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten, tree_map

from omarchy_mac_torch.native import DEVICE

_RANDOM = ("rand", "normal", "uniform", "bernoulli", "dropout", "randint", "randperm", "multinomial", "exponential", "empty")  # random or uninitialized
_SKIP = ("_local_scalar_dense", "empty", "set_", "detach", "alias", "view", "_to_copy", "copy_", "clone", "lift_fresh")


class Mismatch(AssertionError):
    pass


def _cpu(x):
    return x.detach().to("cpu").clone() if isinstance(x, torch.Tensor) and x.device.type == DEVICE else x


class ShadowMode(TorchDispatchMode):
    def __init__(self, rtol=1e-3, atol=1e-4, raise_on_mismatch=True):
        super().__init__()
        self.rtol, self.atol, self.raise_on_mismatch = rtol, atol, raise_on_mismatch
        self.mismatches = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        name = func.__name__
        on_dev = any(isinstance(t, torch.Tensor) and t.device.type == DEVICE for t in tree_flatten((args, kwargs))[0])
        if not on_dev or any(r in name for r in _RANDOM) or any(name.startswith(s) for s in _SKIP):
            return func(*args, **kwargs)
        cargs, ckwargs = tree_map(_cpu, (args, kwargs))
        out = func(*args, **kwargs)
        try:
            ckw = {k: ("cpu" if k == "device" else v) for k, v in ckwargs.items()}
            ref = func(*cargs, **ckw)
        except Exception:
            return out  # the CPU can't run it this way (device-specific args): nothing to compare
        got, _ = tree_flatten(out)
        exp, _ = tree_flatten(ref)
        # in-place / out= ops: compare the mutated inputs too
        got += [a for a in tree_flatten((args, kwargs))[0] if isinstance(a, torch.Tensor)]
        exp += [a for a in tree_flatten((cargs, ckwargs))[0] if isinstance(a, torch.Tensor)]
        for g, e in zip(got, exp):
            if not (isinstance(g, torch.Tensor) and isinstance(e, torch.Tensor)) or g.shape != e.shape:
                continue
            g = g.detach().cpu()
            ok = (torch.equal(g, e) if not g.is_floating_point()
                  else torch.allclose(g, e.to(g.dtype), rtol=self.rtol, atol=self.atol, equal_nan=True))
            if not ok:
                diff = (g.float() - e.float()).abs().max().item() if g.numel() else 0.0
                msg = f"omarchy-mac-torch shadow: {func} differs from the CPU (max |diff| {diff:.3g}, shape {tuple(g.shape)})"
                self.mismatches.append(msg)
                if self.raise_on_mismatch:
                    raise Mismatch(msg)
                break
        return out


@contextlib.contextmanager
def shadow(rtol=1e-3, atol=1e-4, raise_on_mismatch=True):
    """Compare every device op against the CPU inside this block (see the module docstring)."""
    mode = ShadowMode(rtol, atol, raise_on_mismatch)
    with mode:
        yield mode


if os.environ.get("OMARCHY_MAC_TORCH_SHADOW", "0") == "1":
    _global = ShadowMode()
    _global.__enter__()
