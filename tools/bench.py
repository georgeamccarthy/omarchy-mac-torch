"""Benchmark suite: GPU timings for a fixed set of workloads, saved per version in benchmarks/<version>.json and
compared with the previous saved run, so a change that slows something down shows up before it is merged.

usage: uv run tools/bench.py [--save] [--against benchmarks/0.1.0.json]
Run on an otherwise idle machine; numbers are only comparable on the same hardware."""
import argparse
import json
import pathlib
import platform
import time

import torch
import torch.nn.functional as F

import omarchy_mac_torch as omt
from omarchy_mac_torch.models import MODELS, PlainCNN, ResNet18, SeparableCNN, ViT

D = omt.DEVICE
ROOT = pathlib.Path(__file__).resolve().parent.parent


def timeit(fn, warm=3, reps=10):
    for _ in range(warm):
        fn()
    omt.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    omt.synchronize()
    return (time.perf_counter() - t) / reps * 1000


def train_step(model, x, y):
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    def f():
        loss = F.cross_entropy(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return f


def workloads():
    from tools.acceptance import alexnet  # noqa: E402 (same model as the README table)
    torch.manual_seed(0)
    a = alexnet().to(D)
    xa, ya = torch.randn(16, 3, 224, 224, device=D), torch.randint(0, 1000, (16,), device=D)
    yield "alexnet train b16", train_step(a, xa, ya)
    a.eval()
    yield "alexnet infer b16", lambda: a(xa)
    for name, mk, shape in (("plaincnn", PlainCNN, (128, 1, 28, 28)), ("separablecnn", SeparableCNN, (128, 1, 28, 28)),
                            ("resnet18", ResNet18, (32, 3, 64, 64)), ("vit", ViT, (32, 3, 32, 32))):
        m = mk().to(D)
        x, y = torch.randn(*shape, device=D), torch.randint(0, 10, (shape[0],), device=D)
        yield f"{name} train", train_step(m, x, y)
    big = torch.randn(4096, 4096, device=D)
    yield "matmul 4096", lambda: big @ big
    v = torch.randn(1 << 24, device=D)
    yield "sum 16M", lambda: v.sum()
    yield "add 16M", lambda: v + v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true", help=f"write benchmarks/{omt.__version__}.json")
    ap.add_argument("--against", default="", help="a saved run to compare with (default: the newest other one)")
    a = ap.parse_args()
    with torch.no_grad():
        pass
    res = {}
    for name, fn in workloads():
        ctx = torch.no_grad() if "infer" in name or "matmul" in name or "16M" in name else torch.enable_grad()
        with ctx:
            res[name] = round(timeit(fn), 3)
        print(f"{name:22s} {res[name]:9.2f} ms", flush=True)
    saved = sorted(ROOT.glob("benchmarks/*.json"))
    ref_path = pathlib.Path(a.against) if a.against else next((p for p in reversed(saved) if p.stem != omt.__version__), None)
    if ref_path and ref_path.exists():
        ref = json.loads(ref_path.read_text())["ms"]
        print(f"\ncompared with {ref_path.name}:")
        for k, v in res.items():
            if k in ref:
                ch = (v - ref[k]) / ref[k] * 100
                flag = "  <-- slower" if ch > 5 else ("  faster" if ch < -5 else "")
                print(f"  {k:22s} {ref[k]:9.2f} -> {v:9.2f} ms ({ch:+.0f}%){flag}")
    if a.save:
        out = ROOT / "benchmarks" / f"{omt.__version__}.json"
        out.write_text(json.dumps({"version": omt.__version__, "torch": torch.__version__,
                                   "adapter": omt.native.core.adapter_name(), "machine": platform.machine(),
                                   "ms": res}, indent=1) + "\n")
        print("saved", out)


if __name__ == "__main__":
    main()
