"""Inference (eval mode, no grad) ms/batch on the CPU (4 threads) or the GPU. usage: inference_benchmark.py alexnet|small|large cpu|gpu [batch]"""
import sys, time, torch, torch.nn as nn
name, where = sys.argv[1], sys.argv[2]
if where == "gpu":
    import omarchy_mac_torch; omarchy_mac_torch.tune_cpu(); dev = "omarchy_mac_torch"
else:
    torch.set_num_threads(4); dev = "cpu"
if name == "alexnet":
    bs = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    m = nn.Sequential(nn.Conv2d(3, 64, 11, stride=4, padding=2), nn.ReLU(), nn.MaxPool2d(3, 2), nn.Conv2d(64, 192, 5, padding=2), nn.ReLU(),
                      nn.MaxPool2d(3, 2), nn.Conv2d(192, 384, 3, padding=1), nn.ReLU(), nn.Conv2d(384, 256, 3, padding=1), nn.ReLU(),
                      nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d(3, 2), nn.Flatten(), nn.Dropout(), nn.Linear(9216, 4096), nn.ReLU(),
                      nn.Dropout(), nn.Linear(4096, 4096), nn.ReLU(), nn.Linear(4096, 1000))
    x = torch.randn(bs, 3, 224, 224)
else:
    from omarchy_mac_torch.models import MODELS
    bs = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
    m = MODELS[name](); x = torch.randn(bs, 1, 28, 28)
m = m.eval().to(dev); x = x.to(dev)
def run():
    with torch.no_grad(): return m(x)
def sync():
    if where == "gpu": omarchy_mac_torch.synchronize()
for _ in range(3): run()
sync(); t = time.perf_counter(); n = 20
for _ in range(n): out = run()
float(out.sum()); sync(); dt = (time.perf_counter() - t) / n
print(f"{name} inference {where} batch {bs}: {dt * 1000:.1f} ms/batch, {bs / dt:.0f} images/s")
