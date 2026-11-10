"""Per-op conv forward + input/weight gradients on the device vs CPU for kernel sizes the backend supports."""
import copy, torch, omarchy_mac_torch
from torch import nn
torch.manual_seed(0)
for (ci, co, ks, groups) in [(8, 8, 5, 8), (8, 8, 7, 8), (8, 8, 3, 8), (6, 10, 5, 1), (6, 10, 3, 1), (6, 10, 1, 1)]:
    ref = nn.Conv2d(ci, co, ks, padding=ks // 2, groups=groups, bias=False); mod = copy.deepcopy(ref).to("omarchy_mac_torch")
    x = torch.randn(4, ci, 14, 14, requires_grad=True); xa = x.detach().to("omarchy_mac_torch").requires_grad_()
    out_r = ref(x); out_a = mod(xa); g = torch.randn_like(out_r)
    out_r.backward(g); out_a.backward(g.to("omarchy_mac_torch"))
    e = lambda a, b: ((a - b.detach().cpu()).abs().max() / (a.abs().max() + 1e-9)).item()
    print(f"ci{ci} co{co} k{ks} groups{groups}: fwd {e(out_r, out_a):.1e}  dx {e(x.grad, xa.grad):.1e}  dw {e(ref.weight.grad, mod.weight.grad):.1e}")
print("fallbacks:", {str(k): v for k, v in omarchy_mac_torch.FALLBACKS.items()})
