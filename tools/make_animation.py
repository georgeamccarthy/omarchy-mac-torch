"""Render docs/alexnet.svg: a side-by-side animation of AlexNet training on the CPU and the GPU, played back at the
real measured speed from docs/alexnet_timings.json (see notes/scripts/record_alexnet.py). Pure SMIL, no dependencies."""
import json, pathlib

root = pathlib.Path(__file__).resolve().parent.parent
data = json.loads((root / "docs" / "alexnet_timings.json").read_text())
cpu, gpu = data["cpu"], data["gpu"]
cpu_t, gpu_t = cpu["step_end_seconds"], gpu["step_end_seconds"]
n, bs = len(cpu_t), cpu["batch"]
HOLD = 3.5
T = max(cpu_t[-1], gpu_t[-1]) + HOLD                       # loop length in seconds
speedup = cpu_t[-1] / gpu_t[-1]
W, H, PW, PH, X0, GAP, BARW = 920, 400, 410, 250, 40, 60, 330

def kt(t): return f"{min(max(t / T, 0), 1):.5f}"

def timed(display_from, display_to, extra=""):
    """An element visible from display_from to display_to seconds in each loop (discrete opacity animation)."""
    a, b = display_from, display_to
    if a <= 0:
        values, keys = "1;0", f"0;{kt(b)}"
    else:
        values, keys = "0;1;0", f"0;{kt(a)};{kt(b)}"
    return f'<animate attributeName="opacity" values="{values}" keyTimes="{keys}" calcMode="discrete" dur="{T:.3f}s" repeatCount="indefinite"/>'

def panel(x, title, sub, color, ts):
    total, ips = ts[-1], bs * len(ts) / ts[-1]
    out = [f'<g transform="translate({x},95)">',
           f'<rect width="{PW}" height="{PH}" rx="14" fill="#161b22" stroke="#30363d"/>',
           f'<text x="24" y="40" class="h" fill="{color}">{title}</text>',
           f'<text x="24" y="62" class="s">{sub}</text>']
    # step counter
    bounds = [0.0] + list(ts) + [T]
    for k in range(n + 1):
        out.append(f'<text x="24" y="122" class="big" opacity="0">step {k} / {n}{timed(bounds[k], bounds[k + 1])}</text>')
    # elapsed clock, 0.1 s resolution while running
    tick = 0.1
    steps = int(total / tick) + 1
    for k in range(steps + 1):
        a, b = k * tick, (k + 1) * tick
        if k == steps: b = T
        out.append(f'<text x="{PW - 24}" y="122" class="big" text-anchor="end" opacity="0">{min(a, total):.1f} s{timed(a, b)}</text>')
    # progress bar: width follows the recorded step completions
    vals = ["0"] + [f"{BARW * (i + 1) / n:.1f}" for i in range(n)] + [f"{BARW}"]
    keys = ["0"] + [kt(t) for t in ts] + ["1"]
    out.append(f'<rect x="24" y="148" width="{BARW + 0}" height="18" rx="9" fill="#21262d"/>')
    out.append(f'<rect x="24" y="148" width="0" height="18" rx="9" fill="{color}">'
               f'<animate attributeName="width" values="{";".join(vals)}" keyTimes="{";".join(keys)}" dur="{T:.3f}s" repeatCount="indefinite"/></rect>')
    # images/s and done mark once finished
    out.append(f'<text x="24" y="212" class="big" fill="{color}" opacity="0">{ips:.0f} images/s{timed(total, T)}</text>')
    out.append(f'<text x="24" y="236" class="s" opacity="0">{total / len(ts) * 1000:.0f} ms per step{timed(total, T)}</text>')
    out.append('</g>')
    return "\n".join(out)

svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img"
     aria-label="AlexNet training on the CPU versus the GPU, {speedup:.1f} times faster on the GPU">
<style>
  text {{ font-family: -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif; fill: #c9d1d9; }}
  .t {{ font-size: 24px; font-weight: 700; fill: #f0f6fc; }}
  .h {{ font-size: 22px; font-weight: 700; }}
  .s {{ font-size: 14px; fill: #8b949e; }}
  .big {{ font-size: 30px; font-weight: 600; font-variant-numeric: tabular-nums; }}
</style>
<rect width="{W}" height="{H}" rx="18" fill="#0d1117"/>
<text x="{W / 2}" y="46" class="t" text-anchor="middle">AlexNet training: CPU vs GPU</text>
<text x="{W / 2}" y="72" class="s" text-anchor="middle">MacBook Pro 16" (M1 Pro) running Omarchy · PyTorch {cpu.get("torch", "2.15")} · batch {bs}, 224x224 · real measured step times</text>
{panel(X0, "CPU", "PyTorch default backend, 4 threads", "#8b949e", cpu_t)}
{panel(X0 + PW + GAP, "GPU", "omarchy-mac-torch (Honeykrisp Vulkan)", "#3fb950", gpu_t)}
<g opacity="0">
  <text x="{W / 2}" y="372" class="h" text-anchor="middle" fill="#3fb950">GPU is {speedup:.1f}x faster</text>
  {timed(gpu_t[-1], T)}
</g>
</svg>'''
(root / "docs" / "alexnet.svg").write_text(svg)
print(f"wrote docs/alexnet.svg: loop {T:.1f}s, CPU {cpu_t[-1]:.2f}s, GPU {gpu_t[-1]:.2f}s, {speedup:.2f}x, {len(svg) // 1024} KB")
