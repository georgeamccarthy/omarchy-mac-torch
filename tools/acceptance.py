"""Acceptance models: train a few steps of each standard architecture on the CPU and on the GPU from the same weights
and data, and report loss agreement, CPU fallbacks and speed. The table it prints goes in the README.

usage: uv run tools/acceptance.py [--steps 5] [--models alexnet,resnet18,...] [--json results/acceptance.json]"""
import argparse
import copy
import json
import pathlib
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import omarchy_mac_torch as omt
from omarchy_mac_torch.models import GPT, MobileNetV3Small, PlainCNN, ResNet18, SeparableCNN, ViT

D = omt.DEVICE


def alexnet():
    return nn.Sequential(
        nn.Conv2d(3, 64, 11, stride=4, padding=2), nn.ReLU(), nn.MaxPool2d(3, 2),
        nn.Conv2d(64, 192, 5, padding=2), nn.ReLU(), nn.MaxPool2d(3, 2),
        nn.Conv2d(192, 384, 3, padding=1), nn.ReLU(), nn.Conv2d(384, 256, 3, padding=1), nn.ReLU(),
        nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d(3, 2), nn.Flatten(),
        nn.Dropout(0.0), nn.Linear(9216, 4096), nn.ReLU(), nn.Dropout(0.0), nn.Linear(4096, 4096), nn.ReLU(),
        nn.Linear(4096, 1000))


# name -> (model, input batch, labels, optimizer)
MODELS = {
    "alexnet": (alexnet, lambda: torch.randn(16, 3, 224, 224), lambda: torch.randint(0, 1000, (16,)), "sgd"),
    "resnet18": (ResNet18, lambda: torch.randn(32, 3, 64, 64), lambda: torch.randint(0, 10, (32,)), "sgd"),
    "mobilenetv3": (MobileNetV3Small, lambda: torch.randn(32, 3, 96, 96), lambda: torch.randint(0, 10, (32,)), "adam"),
    "vit": (ViT, lambda: torch.randn(32, 3, 32, 32), lambda: torch.randint(0, 10, (32,)), "adamw"),
    "gpt": (GPT, lambda: torch.randint(0, 512, (16, 64)), lambda: torch.randint(0, 512, (16, 64)), "adamw"),
    "plaincnn": (PlainCNN, lambda: torch.randn(128, 1, 28, 28), lambda: torch.randint(0, 10, (128,)), "sgd"),
    "separablecnn": (SeparableCNN, lambda: torch.randn(128, 1, 28, 28), lambda: torch.randint(0, 10, (128,)), "sgd"),
}


def make_opt(kind, params):
    if kind == "sgd":
        return torch.optim.SGD(params, lr=0.01, momentum=0.9)
    if kind == "adam":
        return torch.optim.Adam(params, lr=1e-3)
    return torch.optim.AdamW(params, lr=1e-3, weight_decay=0.01)


def train(model, x, y, kind, steps, dev):
    opt = make_opt(kind, model.parameters())
    losses = []

    def step():
        out = model(x)
        loss = F.cross_entropy(out.reshape(-1, out.shape[-1]), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        return loss

    sync = omt.synchronize if dev == D else (lambda: None)
    losses.append(float(step()))  # warm-up (kernel compilation, autotuning)
    sync()
    t = time.perf_counter()
    for _ in range(steps):
        losses.append(float(step()))
    sync()
    return losses, (time.perf_counter() - t) / steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--json", default="results/acceptance.json")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    rows = []
    for name in a.models.split(","):
        mk, mx, my, kind = MODELS[name]
        torch.manual_seed(0)
        ref = mk()
        for mod in ref.modules():  # dropout draws different masks on each device: off, so the losses are comparable
            if isinstance(mod, nn.Dropout):
                mod.p = 0.0
        x, y = mx(), my()
        gpu = copy.deepcopy(ref).to(D)
        cl, ct = train(ref, x, y, kind, a.steps, "cpu")
        omt.FALLBACKS.clear()
        gl, gt = train(gpu, x.to(D), y.to(D), kind, a.steps, D)
        fb = dict(omt.FALLBACKS)
        diff = max(abs(p - q) / max(abs(p), 1e-6) for p, q in zip(cl, gl))
        rows.append({"model": name, "cpu_ms": ct * 1000, "gpu_ms": gt * 1000, "speedup": ct / gt,
                     "max_rel_loss_diff": diff, "fallbacks": fb, "losses_cpu": cl, "losses_gpu": gl})
        print(f"{name:13s} cpu {ct*1000:8.1f} ms  gpu {gt*1000:8.1f} ms  {ct/gt:5.2f}x  loss diff {diff:.1e}  "
              f"fallback ops {len(fb)}", flush=True)
    print("\n| Model | CPU ms/step | GPU ms/step | Speedup | Loss matches CPU | Ops on CPU fallback |\n|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['model']} | {r['cpu_ms']:.0f} | {r['gpu_ms']:.0f} | {r['speedup']:.2f}x | "
              f"{'yes' if r['max_rel_loss_diff'] < 1e-2 else 'NO'} ({r['max_rel_loss_diff']:.0e}) | "
              f"{', '.join(sorted(k.split(' ')[0] for k in r['fallbacks'])) or 'none'} |")
    p = pathlib.Path(a.json)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
