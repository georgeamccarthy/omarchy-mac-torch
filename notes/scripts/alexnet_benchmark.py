"""Train AlexNet (standard torchvision-style architecture, 224x224 input, ~61M parameters) on the CPU or the GPU and report
ms/step and images/s. usage: alexnet_benchmark.py cpu|gpu [batch] [steps]  (random data: timing doesn't depend on pixels)"""
import sys, time, torch, torch.nn as nn, torch.nn.functional as F

dev = sys.argv[1]; bs = int(sys.argv[2]) if len(sys.argv) > 2 else 16; steps = int(sys.argv[3]) if len(sys.argv) > 3 else 6
if dev == "gpu":
    import omarchy_mac_torch; omarchy_mac_torch.tune_cpu(); dev = "omarchy_mac_torch"
else:
    torch.set_num_threads(4)

class AlexNet(nn.Module):
    def __init__(self, classes=1000):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 11, stride=4, padding=2), nn.ReLU(), nn.MaxPool2d(3, 2),
            nn.Conv2d(64, 192, 5, padding=2), nn.ReLU(), nn.MaxPool2d(3, 2),
            nn.Conv2d(192, 384, 3, padding=1), nn.ReLU(),
            nn.Conv2d(384, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d(3, 2))
        self.classifier = nn.Sequential(
            nn.Dropout(), nn.Linear(256 * 6 * 6, 4096), nn.ReLU(),
            nn.Dropout(), nn.Linear(4096, 4096), nn.ReLU(), nn.Linear(4096, classes))
    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))

torch.manual_seed(0)
m = AlexNet().to(dev); opt = torch.optim.SGD(m.parameters(), lr=1e-2, momentum=0.9)
x = torch.randn(bs, 3, 224, 224).to(dev); y = torch.randint(0, 1000, (bs,)).to(dev)
def step():
    loss = F.cross_entropy(m(x), y); opt.zero_grad(); loss.backward(); opt.step(); return loss.detach()
def sync():
    if dev != "cpu": omarchy_mac_torch.synchronize()
for _ in range(2): step()
sync(); t0 = time.perf_counter()
for _ in range(steps): l = step()
float(l); sync(); dt = (time.perf_counter() - t0) / steps
print(f"AlexNet {sys.argv[1]} batch {bs}: {dt*1000:.0f} ms/step, {bs/dt:.1f} images/s (loss {float(l):.3f})")
if sys.argv[1] == "gpu": print("CPU fallbacks:", dict(omarchy_mac_torch.FALLBACKS))
