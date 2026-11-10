"""Per-kernel GPU time for one AlexNet training step (each dispatch run alone, repeated) on the native backend."""
import sys, torch, torch.nn.functional as F, omarchy_mac_torch
from omarchy_mac_torch import native
sys.argv = ["x", "gpu", "16", "1"]
src = open(__file__.replace("alexnet_profile.py", "alexnet_benchmark.py")).read().split("for _ in range(2): step()")[0]
exec(src)
core = native.core
for _ in range(3): step()
core.sync(); core.kernel_counts(True); core.debug_time_kernels(True); step(); core.sync(); core.debug_time_kernels(False)
times = {k: v for k, v in core.kernel_times(True).items() if not k.startswith("ew#")}; counts = core.kernel_counts(True)
tot = sum(times.values())
print(f"sum of isolated kernel times: {tot:.1f} ms")
for k, v in sorted(times.items(), key=lambda x: -x[1])[:14]: print(f"  {k:36s} x{counts.get(k, 0):3d} {v:7.2f} ms {100 * v / tot:5.1f}%")
