"""Fused conv / BatchNorm+ReLU / pool stacks. accelerate(model) swaps every nn.Sequential made only of supported layers
for a FusedSequential (same modules, parameters and state_dict) that runs from a prebuilt native plan."""

import os
import weakref

import torch
from torch import nn

from omarchy_mac_torch.native import DEVICE, core


def _pair(v):
    return v if isinstance(v, tuple) else (v, v)


def _is_pool(m):
    return (type(m) is nn.MaxPool2d and _pair(m.kernel_size) == (2, 2) and _pair(m.stride) == (2, 2)
            and _pair(m.padding) == (0, 0) and _pair(m.dilation) == (1, 1) and not m.ceil_mode and not m.return_indices)


def _is_avgpool(m):
    return (type(m) is nn.AvgPool2d and _pair(m.kernel_size) == (2, 2) and _pair(m.stride) == (2, 2)
            and _pair(m.padding) == (0, 0) and not m.ceil_mode and m.divisor_override is None)


def _is_depthwise(m):
    return (type(m) is nn.Conv2d and m.groups == m.in_channels == m.out_channels > 1
            and m.kernel_size[0] == m.kernel_size[1] and m.kernel_size[0] in (3, 5, 7)
            and m.stride == (1, 1) and m.padding == (m.kernel_size[0] // 2,) * 2
            and m.dilation == (1, 1) and m.padding_mode == "zeros" and m.weight.dtype == torch.float32)


def _is_dense(m):
    ks = m.kernel_size
    return (type(m) is nn.Conv2d and m.groups == 1 and m.stride == (1, 1) and m.dilation == (1, 1)
            and ks[0] == ks[1] and ks[0] in (1, 3, 5) and m.padding == (ks[0] // 2,) * 2
            and m.padding_mode == "zeros" and m.weight.dtype == torch.float32)


def fusable(seq):
    """Whether every layer of nn.Sequential `seq` has a fused-plan implementation."""
    mods = list(seq)
    if not mods or not isinstance(mods[0], nn.Conv2d):
        return False
    i = 0
    while i < len(mods):
        m = mods[i]
        if isinstance(m, nn.Conv2d):
            if not ((_is_dense(m) or _is_depthwise(m)) and m.bias is None):
                return False
        elif type(m) is nn.AdaptiveAvgPool2d:
            if m.output_size not in (1, (1, 1)):
                return False
        elif type(m) is nn.BatchNorm2d:
            if not (m.affine and m.track_running_stats and m.momentum is not None
                    and m.eps == 1e-5 and i + 1 < len(mods) and type(mods[i + 1]) is nn.ReLU):
                return False
            i += 1  # the ReLU is fused into the BN kernel
        elif not (_is_pool(m) or _is_avgpool(m)):
            return False
        i += 1
    return True


def _params(seq):
    out = []
    for m in seq:
        if isinstance(m, nn.Conv2d):
            out.append(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            out += [m.weight, m.bias]
    return out


def _layer_specs(seq):
    specs, mods, i = [], list(seq), 0
    while i < len(mods):
        m = mods[i]
        if isinstance(m, nn.Conv2d) and m.groups > 1:
            specs.append(("dw", m.in_channels, m.out_channels, m.kernel_size[0], 0.0))
        elif isinstance(m, nn.Conv2d):
            specs.append(("conv", m.in_channels, m.out_channels, m.kernel_size[0], 0.0))
        elif isinstance(m, nn.BatchNorm2d):
            specs.append(("bn", m.num_features, m.num_features, 0, float(m.momentum)))
            i += 1  # the ReLU is part of the BN kernel
        elif isinstance(m, nn.AdaptiveAvgPool2d):
            specs.append(("gap", 0, 0, 0, 0.0))
        elif _is_avgpool(m):
            specs.append(("apool", 0, 0, 0, 0.0))
        else:
            specs.append(("pool", 0, 0, 0, 0.0))
        i += 1
    return specs


_plans = weakref.WeakKeyDictionary()


def _plan(seq, shape, train, input_grad):
    cache = _plans.setdefault(seq, {})
    key = (tuple(shape), train, input_grad)
    plan = cache.get(key)
    if plan is None or plan.busy:
        bns = [m for m in seq if isinstance(m, nn.BatchNorm2d)]
        plan = core.make_plan(_layer_specs(seq), list(shape), train, input_grad, list(_params(seq)),
                              [m.running_mean for m in bns], [m.running_var for m in bns])
        if key not in cache:
            cache[key] = plan
    return plan


class _StackFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, seq, *params):
        plan = _plan(seq, x.shape, True, ctx.needs_input_grad[0])
        out = plan.forward(x)
        plan.busy = True
        ctx.plan = plan
        return out

    @staticmethod
    def backward(ctx, gout):
        plan = ctx.plan
        gx, grads = plan.backward(gout)
        plan.busy = False
        return (gx, None, *grads)


class FusedSequential(nn.Sequential):
    """The nn.Sequential it replaces (same modules and state_dict), run from a prebuilt native plan."""

    def _count_batch(self):
        for m in self:  # what nn.BatchNorm2d.forward would have done
            if isinstance(m, nn.BatchNorm2d) and m.num_batches_tracked is not None:
                m.num_batches_tracked.add_(1)

    def forward(self, x):
        if x.device.type == DEVICE and x.dtype == torch.float32 and x.dim() == 4:
            if self.training and torch.is_grad_enabled():
                self._count_batch()
                return _StackFn.apply(x, self, *_params(self))
            if not self.training and not torch.is_grad_enabled():
                return _plan(self, x.shape, False, False).forward(x)
        return super().forward(x)


def accelerate(model):
    """Swap every nn.Sequential of `model` made only of supported layers (conv, BatchNorm+ReLU, 2x2 pools, global
    average pool) for a fused native stack, in place. Parameters, buffers and state_dict are unchanged. Returns a
    short description."""
    count = 0
    for m in model.modules():
        if type(m) is nn.Sequential and fusable(m):
            m.__class__ = FusedSequential
            count += 1
    return f"{count} fused stacks on {core.adapter_name()}"


# Automatic fusion: the first time an nn.Sequential runs on a device tensor, swap it for a FusedSequential if every
# layer is supported. Uses PyTorch's public global forward pre-hook (no patching). OMARCHY_MAC_TORCH_AUTO_FUSE=0
# turns it off; accelerate(model) does the same swap explicitly.
_checked = weakref.WeakSet()


def _auto_fuse(module, args):
    if type(module) is not nn.Sequential or module in _checked:
        return None
    x = args[0] if args else None
    if not (isinstance(x, torch.Tensor) and x.device.type == DEVICE):
        return None
    _checked.add(module)
    if fusable(module):
        module.__class__ = FusedSequential
        return None
    return None


if os.environ.get("OMARCHY_MAC_TORCH_AUTO_FUSE", "1") != "0":
    nn.modules.module.register_module_forward_pre_hook(_auto_fuse)
