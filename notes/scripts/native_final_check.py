"""End-to-end check of the native backend: fused stacks (train + eval with folded BN), new ops, pool release."""
import copy, torch, torch.nn.functional as F, omarchy_mac_torch
from torch import nn
from omarchy_mac_torch.models import SeparableCNN, PlainCNN, fake_data
print(omarchy_mac_torch.accelerate(nn.Sequential()))
e = lambda a, b: (a.detach().cpu() - b.detach().cpu()).abs().max().item()

for cls in (SeparableCNN, PlainCNN):
    torch.manual_seed(0)
    ref = cls(); mod = copy.deepcopy(ref).to("omarchy_mac_torch")
    fused = [type(m).__name__ for m in mod.modules() if type(m).__name__ == "FusedSequential"]
    o1 = torch.optim.Adam(ref.parameters(), 1e-3); o2 = torch.optim.Adam(mod.parameters(), 1e-3)
    X, y = fake_data(64); out = []
    for _ in range(3):
        l1 = F.cross_entropy(ref(X), y, label_smoothing=0.1); o1.zero_grad(); l1.backward(); o1.step()
        l2 = F.cross_entropy(mod(X.to("omarchy_mac_torch")), y.to("omarchy_mac_torch"), label_smoothing=0.1); o2.zero_grad(); l2.backward(); o2.step()
        out.append((round(l1.item(), 5), round(l2.item(), 5)))
    ref.eval(); mod.eval(); Xe, _ = fake_data(200, 5)
    with torch.no_grad(): d_eval = e(ref(Xe), mod(Xe.to("omarchy_mac_torch")))
    print(f"{cls.__name__}: fused stacks {fused}, losses {out}, eval max diff {d_eval:.1e}")

# gradient accumulation across two backward passes (grad storage guard)
torch.manual_seed(0); ref = SeparableCNN(); mod = copy.deepcopy(ref).to("omarchy_mac_torch")
for seed in (1, 2):
    xb, yb = fake_data(64, seed)
    F.cross_entropy(ref(xb), yb).backward(); F.cross_entropy(mod(xb.to("omarchy_mac_torch")), yb.to("omarchy_mac_torch")).backward()
print("accumulated grad diff:", max(e(a.grad, b.grad) for a, b in zip(ref.parameters(), mod.parameters())))

# new ops
x = torch.randn(6, 10); xa = x.to("omarchy_mac_torch")
print("clamp", e(x.clamp(-0.5, 0.5), xa.clamp(-0.5, 0.5)), "log/abs/rsqrt/pow", e(x.abs().log(), xa.abs().log()), e((x.abs() + 1).rsqrt(), (xa.abs() + 1).rsqrt()), e(x.abs() ** 2.5, xa.abs() ** 2.5))
print("compare", torch.equal((x > 0.1).cpu(), (xa > 0.1).cpu()), torch.equal((x <= xa.cpu()), (xa <= xa).cpu() | (x <= xa.cpu())) )
print("where", e(torch.where(x > 0, x, torch.zeros_like(x)), torch.where(xa > 0, xa, torch.zeros_like(xa))))
print("cat/stack", e(torch.cat([x, x * 2]), torch.cat([xa, xa * 2])), e(torch.stack([x, x * 2]), torch.stack([xa, xa * 2])))
idx = torch.tensor([3, 1, 4, 1]); print("index / index_select", e(x[idx], xa[idx.to("omarchy_mac_torch")]), e(x.index_select(0, idx), xa.index_select(0, idx.to("omarchy_mac_torch"))))
a = torch.randn(3, 4, 5); b = torch.randn(3, 5, 2); print("bmm", e(a @ b, a.to("omarchy_mac_torch") @ b.to("omarchy_mac_torch")))
torch.manual_seed(0); d = F.dropout(torch.ones(1000).to("omarchy_mac_torch"), 0.3, True); kept = (d.cpu() > 0).float().mean().item(); print("dropout keep rate", round(kept, 3), "scale", round(d.cpu().max().item(), 3))
ln = nn.LayerNorm(10); ln_a = copy.deepcopy(ln).to("omarchy_mac_torch"); xx = torch.randn(6, 10, requires_grad=True); xxa = xx.detach().to("omarchy_mac_torch").requires_grad_()
r = ln(xx); aa = ln_a(xxa); g = torch.randn_like(r); r.backward(g); aa.backward(g.to("omarchy_mac_torch"))
print("layer_norm fwd/dx/dgamma", e(r, aa), e(xx.grad, xxa.grad), e(ln.weight.grad, ln_a.weight.grad))
conv = nn.Conv2d(3, 5, 3, padding=1, bias=True); conv_a = copy.deepcopy(conv).to("omarchy_mac_torch"); xc = torch.randn(2, 3, 9, 9)
r = conv(xc); aa = conv_a(xc.to("omarchy_mac_torch")); g = torch.randn_like(r); r.backward(g); aa.backward(g.to("omarchy_mac_torch"))
print("conv with bias fwd/dbias", e(r, aa), e(conv.bias.grad, conv_a.bias.grad))
print("uniform_/normal_ shapes", torch.empty(4, 3, device="omarchy_mac_torch").uniform_().shape, torch.empty(4, device="omarchy_mac_torch").normal_().shape)
print("memory before/after empty_cache:", omarchy_mac_torch.memory_reserved(), end=" -> "); omarchy_mac_torch.empty_cache(); print(omarchy_mac_torch.memory_reserved())
print("fallbacks:", dict(omarchy_mac_torch.FALLBACKS))
