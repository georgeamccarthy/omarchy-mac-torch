"""Training and inference time of the small reference CNNs with and without the fused conv/BN/ReLU/pool stacks.
usage: OMARCHY_MAC_TORCH_AUTO_FUSE=0|1 fusion_benchmark.py [batch]"""
import os, sys, time, torch, torch.nn.functional as F, omarchy_mac_torch as o
from omarchy_mac_torch.models import MODELS, fake_data
D = o.DEVICE; bs = int(sys.argv[1]) if len(sys.argv) > 1 else 128
for name in ("small", "large"):
    torch.manual_seed(0)
    m = MODELS[name]().to(D); opt = torch.optim.SGD(m.parameters(), lr=0.01, momentum=0.9)
    X, y = fake_data(bs); X, y = X.to(D), y.to(D)
    def step():
        loss = F.cross_entropy(m(X), y); opt.zero_grad(); loss.backward(); opt.step()
    for _ in range(3): step()
    o.synchronize(); t = time.perf_counter()
    for _ in range(20): step()
    o.synchronize(); tr = (time.perf_counter() - t) / 20
    m.eval()
    with torch.no_grad():
        for _ in range(3): m(X)
        o.synchronize(); t = time.perf_counter()
        for _ in range(20): m(X)
        o.synchronize(); inf = (time.perf_counter() - t) / 20
    fused = sum(type(s).__name__ == "FusedSequential" for s in m.modules())
    print(f"{name:5s} fuse={os.environ.get('OMARCHY_MAC_TORCH_AUTO_FUSE', '1')} stacks={fused}: train {tr*1000:.1f} ms/step, "
          f"eval {inf*1000:.1f} ms/batch (batch {bs})  fallbacks {dict(o.FALLBACKS)}")
