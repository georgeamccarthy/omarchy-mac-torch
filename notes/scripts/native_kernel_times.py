"""Isolated GPU time per kernel type for one native training step (each dispatch run alone, 6x, averaged).
Includes a fixed per-submission overhead, so small kernels look bigger than they are. usage: small|large [teacher]"""
import sys, torch, torch.nn.functional as F, omarchy_mac_torch
from omarchy_mac_torch import native
from omarchy_mac_torch.models import MODELS, fake_data
core = native.core
name = sys.argv[1]; teach = "teacher" in sys.argv
bs = 64; X, y = fake_data(bs); xb, yb = X.to("omarchy_mac_torch"), y.to("omarchy_mac_torch")
m = MODELS[name]().to("omarchy_mac_torch"); t = MODELS["large"]().eval().to("omarchy_mac_torch") if teach else None
o = torch.optim.Adam(m.parameters(), 1e-3)
def step():
    s = m(xb); l = F.cross_entropy(s, yb, label_smoothing=0.1)
    if t is not None:
        with torch.no_grad(): soft = F.softmax(t(xb) / 4.0, 1)
        l = 0.3 * l + 0.7 * 16 * F.kl_div(F.log_softmax(s / 4.0, 1), soft, reduction="batchmean")
    o.zero_grad(); l.backward(); o.step()
for _ in range(4): step()
core.sync(); core.kernel_counts(True); core.debug_time_kernels(True); step(); core.sync(); core.debug_time_kernels(False)
times = {k: v for k, v in core.kernel_times(True).items() if not k.startswith("ew#")}
counts = core.kernel_counts(True)
tot = sum(times.values())
print(f"{name}{'+teacher' if teach else ''}: sum of isolated kernel times {tot:.2f} ms")
for k, v in sorted(times.items(), key=lambda x: -x[1])[:16]: print(f"  {k:28s} x{counts.get(k, 0):3d} {v:6.2f} ms  {100 * v / tot:4.1f}%")
