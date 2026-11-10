"""Where does a native-backend training step go? CPU cost (GPU submission skipped), dispatch counts, wall time.
usage: native_profile.py small|large [teacher]"""
import sys, time, torch, torch.nn.functional as F, omarchy_mac_torch
from omarchy_mac_torch import native
from omarchy_mac_torch.models import MODELS, fake_data
core = native.core
omarchy_mac_torch.tune_cpu()
name = sys.argv[1]; teach = "teacher" in sys.argv
bs = 64; X, y = fake_data(bs * 8)
m = MODELS[name]().to("omarchy_mac_torch"); t = MODELS["large"]().eval().to("omarchy_mac_torch") if teach else None
o = torch.optim.Adam(m.parameters(), 1e-3)
batches = [(X[i*bs:(i+1)*bs].to("omarchy_mac_torch"), y[i*bs:(i+1)*bs].to("omarchy_mac_torch")) for i in range(8)]
def step(i):
    xb, yb = batches[i % 8]
    s = m(xb); l = F.cross_entropy(s, yb, label_smoothing=0.1)
    if t is not None:
        with torch.no_grad(): soft = F.softmax(t(xb) / 4.0, 1)
        l = 0.3 * l + 0.7 * 16 * F.kl_div(F.log_softmax(s / 4.0, 1), soft, reduction="batchmean")
    o.zero_grad(); l.backward(); o.step(); return l.detach()
def timed(n=150):
    for i in range(10): step(i)
    float(step(0)); t0 = time.perf_counter(); tot = 0
    for i in range(n): tot = tot + step(i)
    t1 = time.perf_counter(); float(tot); return (t1 - t0) / n * 1000, (time.perf_counter() - t0) / n * 1000
for i in range(4): step(i)
core.sync(); core.kernel_counts(True); step(0); core.sync(); counts = {k: v for k, v in core.kernel_counts(True).items() if not k.startswith('ew#')}
print(f"{name}{'+teacher' if teach else ''}: {sum(counts.values())} dispatches/step:", dict(sorted(counts.items(), key=lambda x: -x[1])))
a, b = timed(); print(f"real: {a:.2f} ms/step issuing, {b:.2f} incl. drain")
core.debug_skip_submit(True); c, _ = timed(); core.debug_skip_submit(False)
print(f"CPU only (no GPU submit): {c:.2f} ms/step  => GPU/wait share ~{max(a - c, 0):.2f} ms")
