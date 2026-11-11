"""GFLOP/s of the convolution GEMMs (forward, grad input, grad weight) and of matmul on fixed shapes, for kernel work.
usage: uv run tools/gemm_bench.py [--check]   (--check also compares every result with the CPU)"""
import sys, time, torch, torch.nn.functional as F
import omarchy_mac_torch as omt

D = omt.DEVICE
# (name, n, ci, co, hw, ks, stride, pad): AlexNet's convs at batch 16 plus a ResNet-style 3x3
SHAPES = [("alex1", 16, 3, 64, 224, 11, 4, 2), ("alex2", 16, 64, 192, 27, 5, 1, 2), ("alex3", 16, 192, 384, 13, 3, 1, 1),
          ("alex4", 16, 384, 256, 13, 3, 1, 1), ("alex5", 16, 256, 256, 13, 3, 1, 1), ("res3x3", 32, 128, 128, 16, 3, 1, 1)]
check = "--check" in sys.argv


def timeit(fn, reps=10):
    fn(); omt.synchronize()
    t = time.perf_counter()
    for _ in range(reps): fn()
    omt.synchronize()
    return (time.perf_counter() - t) / reps


tot = {"fwd": [0, 0], "gin": [0, 0], "gw": [0, 0]}
for name, n, ci, co, hw, ks, s, p in SHAPES:
    torch.manual_seed(0)
    x, w = torch.randn(n, ci, hw, hw), torch.randn(co, ci, ks, ks) * 0.05
    xd, wd = x.to(D), w.to(D)
    y = F.conv2d(x, w, stride=s, padding=p) if check else None
    oh = (hw + 2 * p - ks) // s + 1
    gy = torch.randn(n, co, oh, oh); gyd = gy.to(D)
    flops = 2 * n * co * oh * oh * ci * ks * ks
    cb = lambda m: torch.ops.aten.convolution_backward(gyd, xd, wd, None, [s, s], [p, p], [1, 1], False, [0, 0], 1, m)
    t = {"fwd": timeit(lambda: F.conv2d(xd, wd, stride=s, padding=p)),
         "gin": timeit(lambda: cb([True, False, False])), "gw": timeit(lambda: cb([False, True, False]))}
    line = f"{name:7s} " + "  ".join(f"{k} {flops / v / 1e9:6.0f} GF/s {v * 1e3:6.2f} ms" for k, v in t.items())
    for k, v in t.items(): tot[k][0] += flops; tot[k][1] += v
    if check:
        ry = torch.ops.aten.convolution_backward(gy, x, w, None, [s, s], [p, p], [1, 1], False, [0, 0], 1, [True, True, False])
        e = lambda a, b: ((a.cpu() - b).abs().max() / b.abs().max()).item()
        line += f"   err fwd {e(F.conv2d(xd, wd, stride=s, padding=p), y):.0e} gin {e(cb([True, False, False])[0], ry[0]):.0e} gw {e(cb([False, True, False])[1], ry[1]):.0e}"
    print(line, flush=True)
print("overall " + "  ".join(f"{k} {f / t / 1e9:6.0f} GF/s" for k, (f, t) in tot.items()))
for m, k, nn in [(4096, 4096, 4096), (16, 9216, 4096), (2048, 512, 2048)]:
    a, b = torch.randn(m, k, device=D), torch.randn(k, nn, device=D)
    v = timeit(lambda: a @ b)
    print(f"matmul {m}x{k}x{nn}: {2 * m * k * nn / v / 1e9:6.0f} GF/s")
