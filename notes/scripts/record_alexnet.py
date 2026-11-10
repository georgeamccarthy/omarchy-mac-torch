"""Record per-step wall-clock times of AlexNet training on the CPU and the GPU (for docs/alexnet.svg).
usage: record_alexnet.py cpu|gpu [steps] [batch] >> appends to docs/alexnet_timings.json"""
import json, os, sys, time, torch, torch.nn as nn, torch.nn.functional as F

dev_name = sys.argv[1]; steps = int(sys.argv[2]) if len(sys.argv) > 2 else 16; bs = int(sys.argv[3]) if len(sys.argv) > 3 else 16
if dev_name == "gpu":
    import omarchy_mac_torch; omarchy_mac_torch.tune_cpu(); dev = "omarchy_mac_torch"
else:
    torch.set_num_threads(4); dev = "cpu"

class AlexNet(nn.Module):
    def __init__(self, classes=1000):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 11, stride=4, padding=2), nn.ReLU(), nn.MaxPool2d(3, 2),
            nn.Conv2d(64, 192, 5, padding=2), nn.ReLU(), nn.MaxPool2d(3, 2),
            nn.Conv2d(192, 384, 3, padding=1), nn.ReLU(), nn.Conv2d(384, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d(3, 2))
        self.classifier = nn.Sequential(nn.Dropout(), nn.Linear(256 * 6 * 6, 4096), nn.ReLU(), nn.Dropout(),
                                        nn.Linear(4096, 4096), nn.ReLU(), nn.Linear(4096, classes))
    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))

torch.manual_seed(0)
m = AlexNet().to(dev); opt = torch.optim.SGD(m.parameters(), lr=1e-2, momentum=0.9)
x = torch.randn(bs, 3, 224, 224).to(dev); y = torch.randint(0, 1000, (bs,)).to(dev)
def step():
    loss = F.cross_entropy(m(x), y); opt.zero_grad(); loss.backward(); opt.step(); return loss
def sync():
    if dev_name == "gpu": omarchy_mac_torch.synchronize()
for _ in range(3): float(step())
sync(); t0 = time.perf_counter(); ts = []; losses = []
for _ in range(steps):
    l = step(); losses.append(float(l)); sync(); ts.append(time.perf_counter() - t0)   # float() waits for the step to finish
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "docs", "alexnet_timings.json")
data = json.load(open(out)) if os.path.exists(out) else {}
data[dev_name] = {"batch": bs, "step_end_seconds": ts, "loss": losses}
json.dump(data, open(out, "w"), indent=1)
print(f"{dev_name}: {steps} steps in {ts[-1]:.2f}s = {ts[-1]/steps*1000:.0f} ms/step, {bs*steps/ts[-1]:.1f} images/s")
