"""Strided / padded convolutions, general max pooling, adaptive average pooling and in-place ReLU vs the CPU."""
import copy, torch, torch.nn as nn, torch.nn.functional as F, omarchy_mac_torch
D = "omarchy_mac_torch"
e = lambda a, b: ((a.detach().cpu() - b.detach().cpu()).abs().max() / (a.detach().cpu().abs().max() + 1e-9)).item()
torch.manual_seed(0)
for (ci, co, ks, s, p, hw) in [(3, 8, 11, 4, 2, 63), (4, 6, 5, 1, 2, 17), (3, 6, 3, 2, 1, 15), (4, 6, 1, 1, 0, 9), (6, 10, 3, 1, 0, 12), (5, 7, 5, 2, 0, 21)]:
    ref = nn.Conv2d(ci, co, ks, stride=s, padding=p, bias=True); mod = copy.deepcopy(ref).to(D)
    x = torch.randn(3, ci, hw, hw, requires_grad=True); xa = x.detach().to(D).requires_grad_()
    r = ref(x); a = mod(xa); g = torch.randn_like(r); r.backward(g); a.backward(g.to(D))
    print(f"conv ci{ci} co{co} k{ks} s{s} p{p}: fwd {e(r, a):.1e} dw {e(ref.weight.grad, mod.weight.grad):.1e} db {e(ref.bias.grad, mod.bias.grad):.1e}"
          + (f" dx {e(x.grad, xa.grad):.1e}" if s == 1 else " (dx: host)"))
for (k, s, p, hw) in [(3, 2, 0, 13), (3, 2, 0, 27), (3, 2, 1, 14), (3, 1, 1, 9), (2, 2, 0, 12), (5, 3, 2, 17)]:
    x = torch.randn(2, 4, hw, hw, requires_grad=True); xa = x.detach().to(D).requires_grad_()
    r = F.max_pool2d(x, k, s, p); a = F.max_pool2d(xa, k, s, p); g = torch.randn_like(r); r.backward(g); a.backward(g.to(D))
    print(f"maxpool k{k} s{s} p{p} {hw}x{hw}: fwd {e(r, a):.1e} bwd {e(x.grad, xa.grad):.1e}")
for (hw, o) in [(6, 6), (13, 6), (10, 4), (7, 1)]:
    x = torch.randn(2, 5, hw, hw, requires_grad=True); xa = x.detach().to(D).requires_grad_()
    r = F.adaptive_avg_pool2d(x, o); a = F.adaptive_avg_pool2d(xa, o); g = torch.randn_like(r); r.backward(g); a.backward(g.to(D))
    print(f"adaptive_avg_pool {hw}->{o}: fwd {e(r, a):.1e} bwd {e(x.grad, xa.grad):.1e}")
x = torch.randn(4, 7); xa = x.to(D); print("relu_ in place:", e(torch.relu(x), F.relu(xa, inplace=True)), e(torch.relu(x), xa))
print("fallbacks:", dict(omarchy_mac_torch.FALLBACKS))
