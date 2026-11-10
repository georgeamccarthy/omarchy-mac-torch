"""AlexNet inference (eval mode, no grad), ms per batch on the CPU (4 threads) or the GPU. usage: alexnet_inference.py cpu|gpu [batch]"""
import sys, time, torch, torch.nn as nn
where = sys.argv[1]; bs = int(sys.argv[2]) if len(sys.argv) > 2 else 16
if where == "gpu":
    import omarchy_mac_torch; omarchy_mac_torch.tune_cpu(); dev = "omarchy_mac_torch"
else:
    torch.set_num_threads(4); dev = "cpu"
torch.manual_seed(0)
m = nn.Sequential(nn.Conv2d(3, 64, 11, stride=4, padding=2), nn.ReLU(), nn.MaxPool2d(3, 2), nn.Conv2d(64, 192, 5, padding=2), nn.ReLU(),
                  nn.MaxPool2d(3, 2), nn.Conv2d(192, 384, 3, padding=1), nn.ReLU(), nn.Conv2d(384, 256, 3, padding=1), nn.ReLU(),
                  nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d(3, 2), nn.Flatten(), nn.Dropout(), nn.Linear(9216, 4096), nn.ReLU(),
                  nn.Dropout(), nn.Linear(4096, 4096), nn.ReLU(), nn.Linear(4096, 1000)).eval().to(dev)
x = torch.randn(bs, 3, 224, 224).to(dev)
def run():
    with torch.no_grad(): return m(x)
def sync():
    if where == "gpu": omarchy_mac_torch.synchronize()
for _ in range(3): run()
sync(); t = time.perf_counter(); n = 20
for _ in range(n): out = run()
float(out.sum()); sync(); dt = (time.perf_counter() - t) / n
print(f"AlexNet inference {where} batch {bs}: {dt * 1000:.1f} ms/batch, {bs / dt:.0f} images/s")
