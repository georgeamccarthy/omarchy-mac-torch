"""Record per-step wall-clock times of training on the CPU and the GPU, for the README animations.
usage: record_training.py alexnet|resnet18|mobilenetv3 cpu|gpu [steps]   (writes docs/timings.json)"""
import json, pathlib, sys, time, torch, torch.nn.functional as F

name, dev_name = sys.argv[1], sys.argv[2]
steps = int(sys.argv[3]) if len(sys.argv) > 3 else 16
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
if dev_name == "gpu":
    import omarchy_mac_torch; omarchy_mac_torch.tune_cpu(); dev = "omarchy_mac_torch"
else:
    torch.set_num_threads(4); dev = "cpu"
from tools.acceptance import MODELS, make_opt
mk, mx, my, kind = MODELS[name]
torch.manual_seed(0)
m = mk()
for mod in m.modules():
    if isinstance(mod, torch.nn.Dropout): mod.p = 0.0
m = m.to(dev); opt = make_opt(kind, m.parameters())
x, y = mx().to(dev), my().to(dev)
def step():
    loss = F.cross_entropy(m(x), y); opt.zero_grad(); loss.backward(); opt.step(); return loss
for _ in range(3): float(step())
t0 = time.perf_counter(); ts, losses = [], []
for _ in range(steps):
    losses.append(float(step())); ts.append(time.perf_counter() - t0)   # float() waits for the step to finish
out = pathlib.Path(__file__).resolve().parents[2] / "docs" / "timings.json"
data = json.loads(out.read_text()) if out.exists() else {}
data.setdefault(name, {})[dev_name] = {"batch": x.shape[0], "shape": list(x.shape[1:]), "step_end_seconds": ts, "loss": losses}
out.write_text(json.dumps(data, indent=1))
print(f"{name} {dev_name}: {steps} steps in {ts[-1]:.2f}s = {ts[-1] / steps * 1000:.0f} ms/step")
