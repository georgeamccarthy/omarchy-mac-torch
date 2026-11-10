"""Fixed GPU cost of one dependent dispatch (a chain of tiny elementwise ops), native backend."""
import time, torch, omarchy_mac_torch
for n in (64, 4096, 262144):
    x = torch.zeros(n, device="omarchy_mac_torch")
    def chain(k):
        y = x
        for _ in range(k): y = y + 1.0
        return y
    for _ in range(3): chain(50)
    omarchy_mac_torch.synchronize()
    t0 = time.perf_counter(); y = chain(400); omarchy_mac_torch.synchronize(); dt = (time.perf_counter() - t0) / 400 * 1e6
    print(f"{n:7d} elements: {dt:6.1f} us per dependent dispatch (incl. CPU issue)")
