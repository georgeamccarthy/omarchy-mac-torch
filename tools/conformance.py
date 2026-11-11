"""Conformance sweep: run PyTorch's own OpInfo sample inputs (torch.testing._internal op_db, ~2000 ops with odd shapes,
broadcasting, non-contiguous views and edge-case arguments) on the device and on the CPU and compare.

usage: uv run tools/conformance.py [--ops add,softmax,...] [--grad] [--out results/conformance.json] [--limit N]

Each sample is classified: ok (matches the CPU, on the GPU), fallback (matches, but some op fell back to the CPU),
mismatch (different values), error (raised on the device but not on the CPU), skip (the CPU raised too, or the op
isn't float32). --grad also compares the gradients of a sum of the outputs. Writes JSON per op plus a summary table.
Not a test suite: a map of what works, to run before and after a change."""
import argparse
import collections
import json
import re
import pathlib
import sys
import time
import traceback

import torch
import omarchy_mac_torch as omt
from torch.testing._internal.common_methods_invocations import op_db
from torch.utils._pytree import tree_flatten, tree_map

D = omt.DEVICE
# outputs that are random or uninitialized: comparing values is meaningless (partial_views: as_strided into storage
# beyond the view, which a device copy of the view doesn't have)
NONDETERMINISTIC = re.compile(r"partial_views|rand|normal|uniform|bernoulli|dropout|empty|fractional|lowrank|multinomial|poisson|"
                              r"exponential|geometric|cauchy|log_normal|alpha_dropout|feature_alpha")


def to_dev(x):
    return x.detach().to(D).requires_grad_(x.requires_grad) if isinstance(x, torch.Tensor) else x


def flat_tensors(x):
    return [t for t in tree_flatten(x)[0] if isinstance(t, torch.Tensor)]


def close(a, b, rtol, atol):
    if a.shape != b.shape:
        return False
    a = a.detach().cpu()
    if not a.is_floating_point() and not a.is_complex():  # integers / bools: exact
        return torch.equal(a, b.to(a.dtype))
    return torch.allclose(a, b.to(a.dtype), rtol=rtol, atol=atol, equal_nan=True)


def random_sample(s):
    """attention / MHA samples with dropout enabled draw random masks"""
    if (s.kwargs.get("dropout_p") or 0) > 0:
        return True
    return bool(s.kwargs.get("training")) and any(isinstance(a, float) and 0 < a < 1 for a in s.args)


def run_sample(op, s, grad, rtol, atol):
    if random_sample(s):
        return "skip", "random (dropout)"
    args = (s.input, *s.args)
    try:
        ref = op(*args, **s.kwargs)
    except Exception:
        return "skip", "cpu raised"
    dargs, dkw = tree_map(to_dev, (args, s.kwargs))
    omt.FALLBACKS.clear()
    try:
        got = op(*dargs, **dkw)
        omt.synchronize()
    except Exception as e:
        msg = str(e).splitlines()[0]
        if "to be on cpu" in msg or "Sparse" in msg:  # an argument PyTorch requires on the CPU / a sparse layout
            return "skip", "argument must stay on the CPU or is sparse"
        return "error", f"{type(e).__name__}: {msg[:200]}"
    fell_back = dict(omt.FALLBACKS)
    rt, gt = flat_tensors(ref), flat_tensors(got)
    if len(rt) != len(gt):
        return "mismatch", "different number of outputs"
    for r, g in zip(rt, gt):
        if not close(g, r, rtol, atol):
            diff = (g.detach().cpu().double() - r.detach().double()).abs().max().item() if r.numel() and r.is_floating_point() else None
            return "mismatch", f"max |diff| {diff}" if diff is not None else "values differ"
    if grad:
        rin = [t for t in flat_tensors(args) if t.requires_grad]
        din = [t for t in flat_tensors(dargs) if t.requires_grad]
        outs_r = [t for t in rt if t.requires_grad]
        outs_d = [t for t in gt if t.requires_grad]
        if rin and outs_r:
            try:
                gr = torch.autograd.grad(sum(t.sum() for t in outs_r), rin, allow_unused=True)
            except Exception:
                gr = None
            if gr is not None:
                try:
                    gd = torch.autograd.grad(sum(t.sum() for t in outs_d), din, allow_unused=True)
                except Exception as e:
                    return "error", f"backward: {type(e).__name__}: {str(e).splitlines()[0][:200]}"
                for a, b in zip(gr, gd):
                    scale = max(1.0, a.abs().max().item()) if a is not None and a.numel() else 1.0  # gradients sum many terms
                    if (a is None) != (b is None) or (a is not None and not close(b, a, rtol * 10, atol * 10 * scale)):
                        return "mismatch", "gradient differs"
                fell_back.update(omt.FALLBACKS)
    if fell_back:
        return "fallback", ", ".join(sorted(fell_back))[:300]
    return "ok", ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ops", default="")
    ap.add_argument("--grad", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="max samples per op (0 = all)")
    ap.add_argument("--out", default="results/conformance.json")
    ap.add_argument("--rtol", type=float, default=1e-4)
    ap.add_argument("--atol", type=float, default=1e-5)
    a = ap.parse_args()
    want = set(filter(None, a.ops.split(",")))
    results, t0 = {}, time.time()
    for info in op_db:
        key = info.name + (("." + info.variant_test_name) if info.variant_test_name else "")
        if want and info.name not in want and key not in want:
            continue
        if torch.float32 not in info.dtypes or NONDETERMINISTIC.search(key):
            continue
        counts, notes = collections.Counter(), collections.Counter()
        try:
            samples = list(info.sample_inputs("cpu", torch.float32, requires_grad=a.grad and info.supports_autograd))
        except Exception:
            continue
        for i, s in enumerate(samples):
            if a.limit and i >= a.limit:
                break
            try:
                status, note = run_sample(info.op, s, a.grad and info.supports_autograd, a.rtol, a.atol)
            except Exception:
                status, note = "error", traceback.format_exc().splitlines()[-1][:200]
            counts[status] += 1
            if note:
                notes[f"{status}: {note}"] += 1
        results[key] = {"counts": dict(counts), "notes": dict(notes.most_common(5))}
        worst = next((k for k in ("error", "mismatch", "fallback", "ok") if counts.get(k)), "skip")
        print(f"{key:40s} {worst:8s} {dict(counts)}", flush=True)
    summary = collections.Counter()
    for r in results.values():
        c = r["counts"]
        summary[next((k for k in ("error", "mismatch", "fallback", "ok") if c.get(k)), "skip")] += 1
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": dict(summary), "ops": results, "seconds": round(time.time() - t0)}, indent=1))
    print(f"\n{len(results)} ops in {time.time() - t0:.0f}s, worst status per op: {dict(summary)} -> {out}")
    sys.exit(0)


if __name__ == "__main__":
    main()
