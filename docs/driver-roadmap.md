# Driver roadmap

Driver-level work (Honeykrisp Vulkan driver, Mesa's Apple GPU compiler, the
Asahi kernel driver) that would speed up small-CNN training workloads, ranked from
easiest to hardest. Written 2026-11-10, after early convolution-kernel experiments.

For these workloads, the backend itself has bigger and cheaper wins first (fusion, fewer dispatches, tuned kernels). Driver work pays off after those.

| Rank by ease | Item | Impact rank | Expected payoff |
|---|---|---|---|
| 1 | GPU clock ramp-up for bursty compute | 4 | unknown until measured; could be large if clocks stay low |
| 2 | Zero-copy host memory | 5 | small, every step |
| 3 | Cooperative matrix in upstream Honeykrisp | 2 | kernel throughput from 3–7% of peak toward 30–60% |
| 4 | Cheaper submission and synchronisation | 1 | removes the ~0.8 ms per-call floor that dominates batch-64 steps |
| 5 | Better shader compiler code | 3 | every kernel, including ones we never hand-tune |

## 1. GPU clock ramp-up (easiest)

Training issues many short bursts of GPU work. If power management raises
the clock only after sustained load, bursts run at low frequency.

- Check: read the GPU frequency from sysfs during a training run.
- Fix: DVFS tuning in the Asahi kernel driver, possibly a runtime knob. Risk:
  firmware may own the power controller.

## 2. Zero-copy host memory (easy to medium)

Augmentation runs on the CPU, so every batch is copied to the GPU, although
the M1's memory is shared.

- Check whether Honeykrisp exposes `VK_EXT_external_memory_host` and fast
  host-cached, device-local memory. If yes, it's our plumbing only.
- If not, a contained Mesa feature with a clear spec.

## 3. Cooperative matrix upstream (medium)

Exposes Apple's hardware matrix instructions (simdgroup matrix) to Vulkan
via `VK_KHR_cooperative_matrix`, so conv-as-GEMM and linear layers use them.

- Patches exist in Joshua Warren's Omarchy Mesa fork
  (https://github.com/omacom/mesa/pull/6); stock Mesa doesn't expose it.
- Remaining work: review, conformance tests, edge cases, upstream
  acceptance. Shortcut: use the patched Mesa directly.

## 4. Cheaper submission and synchronisation (hard)

A batch-64 step is ~50 ops, and the spikes show a ~0.8 ms floor per GPU
call; at that rate overhead alone exceeds the CPU's whole step.

- First profile where the floor comes from: wgpu, Honeykrisp, the Asahi
  kernel driver (Rust) or the GPU firmware interface. If it's mostly wgpu,
  switching to raw Vulkan fixes it cheaply.
- Driver fixes: shorter submit/fence paths, timeline semaphores, fewer
  flushes between compute dispatches.

## 5. Better shader compiler code (hardest)

Codegen already swung performance 3× in the spike (accumulators spilled to
memory until rewritten as `vec4`s).

- Register allocation and fewer spills; unrolling fixed-trip loops;
  strength-reducing division/modulo by uniforms; shared-memory bank
  conflicts.
- Open-ended work in Mesa's AGX compiler on a reverse-engineered ISA, with
  regression risk for every other app.
