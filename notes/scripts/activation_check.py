"""New activations and 2x2 average pooling: forward and backward on the device vs CPU."""
import torch, torch.nn.functional as F, omarchy_mac_torch
torch.manual_seed(0)
fns = {"sigmoid": torch.sigmoid, "silu": F.silu, "hardswish": F.hardswish, "hardsigmoid": F.hardsigmoid, "tanh": torch.tanh,
       "avg_pool2d": lambda t: F.avg_pool2d(t, 2) if t.dim() == 4 else t}
for name, fn in fns.items():
    shape = (3, 4, 9, 9) if name == "avg_pool2d" else (50, 7)
    x = (torch.randn(*shape) * 3).requires_grad_(); xa = x.detach().to("omarchy_mac_torch").requires_grad_()
    r = fn(x); a = fn(xa); g = torch.randn_like(r)
    r.backward(g); a.backward(g.to("omarchy_mac_torch"))
    e = lambda p, q: (p - q.detach().cpu()).abs().max().item()
    print(f"{name:12s} fwd {e(r, a):.1e}  grad {e(x.grad, xa.grad):.1e}")
print("fallbacks:", {str(k): v for k, v in omarchy_mac_torch.FALLBACKS.items()})
