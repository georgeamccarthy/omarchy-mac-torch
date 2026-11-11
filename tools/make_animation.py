"""Render docs/<model>.svg for AlexNet, ResNet-18 and MobileNetV3: the network drawn as a 3D block model, one copy for
the CPU and one for the GPU. Every training step a light sweeps forward through the layers, then back (the backward
pass), and a loading bar fills as steps finish. It plays at the real measured speed from docs/timings.json (see
notes/scripts/record_training.py), so the GPU copy loads faster. Pure SMIL, no dependencies.
usage: make_animation.py [alexnet resnet18 mobilenetv3]"""
import json, math, pathlib, sys

root = pathlib.Path(__file__).resolve().parent.parent
ALL = json.loads((root / "docs" / "timings.json").read_text())

# (name, kind, channels, spatial, units); kind: in / c(onv) / p(ool) / f(ully connected)
MODEL_LAYERS = {
    "alexnet": ("AlexNet", [("input", "in", 3, 224, 0), ("conv1", "c", 64, 55, 0), ("pool", "p", 64, 27, 0),
                            ("conv2", "c", 192, 27, 0), ("pool", "p", 192, 13, 0), ("conv3", "c", 384, 13, 0),
                            ("conv4", "c", 256, 13, 0), ("conv5", "c", 256, 13, 0), ("pool", "p", 256, 6, 0),
                            ("fc1", "f", 0, 0, 4096), ("fc2", "f", 0, 0, 4096), ("fc3", "f", 0, 0, 1000)]),
    "resnet18": ("ResNet-18", [("input", "in", 3, 64, 0), ("conv1", "c", 64, 32, 0), ("pool", "p", 64, 16, 0)]
                 + [("conv", "c", 64, 16, 0)] * 4 + [("conv", "c", 128, 8, 0)] * 4 + [("conv", "c", 256, 4, 0)] * 4
                 + [("conv", "c", 512, 2, 0)] * 4 + [("pool", "p", 512, 1, 0), ("fc", "f", 0, 0, 10)]),
    "mobilenetv3": ("MobileNetV3", [("input", "in", 3, 96, 0), ("stem", "c", 16, 48, 0), ("block", "c", 16, 24, 0),
                                    ("block", "c", 24, 12, 0), ("block", "c", 24, 12, 0), ("block", "c", 40, 6, 0),
                                    ("block", "c", 40, 6, 0), ("block", "c", 48, 6, 0), ("block", "c", 96, 3, 0),
                                    ("head", "c", 576, 3, 0), ("pool", "p", 576, 1, 0), ("fc1", "f", 0, 0, 256),
                                    ("fc2", "f", 0, 0, 10)]),
}


def render(model):
  title_name, LAYERS = MODEL_LAYERS[model]
  data = ALL[model]
  cpu, gpu = data["cpu"], data["gpu"]
  cpu_t, gpu_t = cpu["step_end_seconds"], gpu["step_end_seconds"]
  n, bs = len(cpu_t), cpu["batch"]
  HOLD = 3.5
  T = max(cpu_t[-1], gpu_t[-1]) + HOLD
  speedup = cpu_t[-1] / gpu_t[-1]
  W, H, PW, PH, X0, GAP, BARW = 940, 460, 410, 320, 40, 40, 362

  NL = len(LAYERS)
  GAPX = max(3, 8 - (NL - 12) // 2)
  side_in = LAYERS[0][3]
  def dims(l):
      _, k, c, s, u = l
      if k == "f": return 7, 12 + 10 * math.log2(u / 1000 + 1) * 1.6, 9           # thickness, height, depth
      return 4 + math.sqrt(c) * 1.45, 10 + s / side_in * 58, 10 + s / side_in * 58
  raw = [dims(l) for l in LAYERS]
  SCALE = (BARW - GAPX * (NL - 1)) / sum(d[0] for d in raw)
  OBL = 0.5                                                  # oblique depth projection

  def boxes(ox, oy):
      """Yield (index, kind, top, front, side) polygon strings; the model is centred on baseline oy, left edge ox."""
      x = ox
      for i, (l, (th, h, d)) in enumerate(zip(LAYERS, raw)):
          th = th * SCALE
          dx, dy = d * OBL * 0.8, d * OBL * 0.45
          y0 = oy + h / 2
          front = f"{x:.1f},{y0:.1f} {x + th:.1f},{y0:.1f} {x + th:.1f},{y0 - h:.1f} {x:.1f},{y0 - h:.1f}"
          top = f"{x:.1f},{y0 - h:.1f} {x + th:.1f},{y0 - h:.1f} {x + th + dx:.1f},{y0 - h - dy:.1f} {x + dx:.1f},{y0 - h - dy:.1f}"
          side = f"{x + th:.1f},{y0:.1f} {x + th + dx:.1f},{y0 - dy:.1f} {x + th + dx:.1f},{y0 - h - dy:.1f} {x + th:.1f},{y0 - h:.1f}"
          yield i, l[1], top, front, side
          x += th + GAPX

  def kt(t): return f"{min(max(t / T, 0), 1):.5f}"

  def timed(a, b):
      values, keys = ("1;0", f"0;{kt(b)}") if a <= 0 else ("0;1;0", f"0;{kt(a)};{kt(b)}")
      return f'<animate attributeName="opacity" values="{values}" keyTimes="{keys}" calcMode="discrete" dur="{T:.3f}s" repeatCount="indefinite"/>'

  def pulse_anim(events):
      """events: list of (peak_time, width). Opacity rises just before the peak and falls after it."""
      ks, vs = [0.0], [0]
      for t, w in events:
          for tt, v in ((t - 1e-3, 0), (t, 1), (t + w, 0)):
              if tt > ks[-1]: ks.append(tt); vs.append(v)
      ks.append(T); vs.append(0)
      return (f'<animate attributeName="opacity" values="{";".join(map(str, vs))}" '
              f'keyTimes="{";".join(kt(k) for k in ks)}" dur="{T:.3f}s" repeatCount="indefinite"/>')

  def panel(x, title, sub, color, bcolor, ts):
      total, ips = ts[-1], bs * len(ts) / ts[-1]
      starts = [0.0] + list(ts[:-1])
      o = [f'<g transform="translate({x},95)">',
           f'<rect width="{PW}" height="{PH}" rx="14" fill="#161b22" stroke="#30363d"/>',
           f'<text x="24" y="38" class="h" fill="{color}">{title}</text>',
           f'<text x="24" y="58" class="s">{sub}</text>']
      bounds = [0.0] + list(ts) + [T]
      for k in range(n + 1):
          o.append(f'<text x="24" y="102" class="big" opacity="0">step {k} / {n}{timed(bounds[k], bounds[k + 1])}</text>')
      tick, steps = 0.1, int(total / 0.1) + 1
      for k in range(steps + 1):
          a, b = k * tick, T if k == steps else (k + 1) * tick
          o.append(f'<text x="{PW - 24}" y="102" class="big" text-anchor="end" opacity="0">{min(a, total):.1f} s{timed(a, b)}</text>')
      ox, oy = 24 + 0, 200
      for i, kind, top, front, side in boxes(ox, oy):
          o.append(f'<g class="b"><polygon points="{top}" fill="#3b434d"/><polygon points="{front}" fill="#2a3038"/><polygon points="{side}" fill="#20252b"/></g>')
      for i, kind, top, front, side in boxes(ox, oy):
          fwd = [(s + (i / NL) * 0.40 * (e - s), 0.14 * (e - s)) for s, e in zip(starts, ts)]
          bwd = [(s + (0.46 + (NL - 1 - i) / NL * 0.46) * (e - s), 0.12 * (e - s)) for s, e in zip(starts, ts)]
          for ev, c in ((fwd, color), (bwd, bcolor)):
              o.append(f'<g opacity="0">{pulse_anim(ev)}<polygon points="{top}" fill="{c}"/><polygon points="{front}" fill="{c}" fill-opacity=".8"/><polygon points="{side}" fill="{c}" fill-opacity=".6"/></g>')
          o.append(f'<g opacity="0">{timed(total, T)}<polygon points="{top}" fill="{color}"/><polygon points="{front}" fill="{color}" fill-opacity=".8"/><polygon points="{side}" fill="{color}" fill-opacity=".6"/></g>')
      # loading bar, filling as steps complete
      vals = ["0"] + [f"{BARW * (i + 1) / n:.1f}" for i in range(n)] + [f"{BARW}"]
      keys = ["0"] + [kt(t) for t in ts] + ["1"]
      o.append(f'<rect x="24" y="258" width="{BARW}" height="16" rx="8" fill="#21262d"/>')
      o.append(f'<rect x="24" y="258" width="0" height="16" rx="8" fill="{color}"><animate attributeName="width" values="{";".join(vals)}" keyTimes="{";".join(keys)}" dur="{T:.3f}s" repeatCount="indefinite"/></rect>')
      o.append(f'<text x="24" y="306" class="big" fill="{color}" opacity="0">{ips:.0f} images/s{timed(total, T)}</text>')
      o.append(f'<text x="{PW - 24}" y="306" class="s" text-anchor="end" opacity="0">{total / len(ts) * 1000:.0f} ms per step{timed(total, T)}</text>')
      o.append('</g>')
      return "\n".join(o)

  svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img"
       aria-label="{title_name} training as a 3D model loading on the CPU versus the GPU, {speedup:.1f} times faster on the GPU">
  <style>
    text {{ font-family: -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif; fill: #c9d1d9; }}
    .t {{ font-size: 24px; font-weight: 700; fill: #f0f6fc; }}
    .h {{ font-size: 22px; font-weight: 700; }}
    .s {{ font-size: 14px; fill: #8b949e; }}
    .big {{ font-size: 28px; font-weight: 600; font-variant-numeric: tabular-nums; }}
    .b polygon {{ stroke: #4a525c; stroke-width: .6; }}
  </style>
  <rect width="{W}" height="{H}" rx="18" fill="#0d1117"/>
  <text x="{W / 2}" y="46" class="t" text-anchor="middle">{title_name} training: CPU vs GPU</text>
  <text x="{W / 2}" y="72" class="s" text-anchor="middle">MacBook Pro 16" (M1 Pro) running Omarchy · batch {bs}, {data["cpu"]["shape"][-1]}x{data["cpu"]["shape"][-1]} · real measured step times · light = forward pass, then backward</text>
  {panel(X0, "CPU", "PyTorch default backend, 4 threads", "#8b949e", "#c9d1d9", cpu_t)}
  {panel(X0 + PW + GAP, "GPU", "omarchy-mac-torch (Honeykrisp Vulkan)", "#3fb950", "#a5f0b5", gpu_t)}
  <g opacity="0">
    <text x="{W / 2}" y="450" class="h" text-anchor="middle" fill="#3fb950">GPU is {speedup:.1f}x faster</text>
    {timed(gpu_t[-1], T)}
  </g>
  </svg>'''
  (root / "docs" / f"{model}.svg").write_text(svg)
  print(f"wrote docs/{model}.svg: loop {T:.1f}s, CPU {cpu_t[-1]:.2f}s, GPU {gpu_t[-1]:.2f}s, {speedup:.2f}x, {len(svg) // 1024} KB")


for m in (sys.argv[1:] or list(MODEL_LAYERS)):
    render(m)
