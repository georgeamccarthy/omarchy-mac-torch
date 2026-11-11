# Contributing

The usual way in: run your model, `print(omarchy_mac_torch.FALLBACKS)`, pick the op that fell back most, add it.
Agent-written PRs are welcome.

## Adding an op

1. **Find the aten op.** `FALLBACKS` names it (e.g. `isneginf`, `where.self_out`). Its signature is in
   `torch/include/ATen/ops/<op>_ops.h` (`struct <op> { using schema = ... }`).
2. **Write the kernel** in `csrc/kernels/<name>.wgsl`. Conventions:
   - storage buffers are arrays of 32-bit words (`array<f32>` / `array<u32>`); int64 is read through its low word,
     bool is one byte per element (pack 4 per word when writing, see `cmp.wgsl`);
   - offsets and sizes go in a uniform struct; use a 2-D grid (`gid.x + gid.y * nwg.x * 256u`) for large tensors;
   - never test NaN with `x != x` (the compiler may assume NaN can't happen): use `isnan_` from `cmp.wgsl`;
   - a kernel can't bind the same buffer read-only and read-write; `launch()` copies around that automatically, but
     an in-place op is faster reading its input through the output binding (see `ew.wgsl`'s `fa`/`fb`).
   Simple elementwise ops don't need a kernel: add a case to `ew.wgsl` and an `ew::` code in `omt.h`.
3. **Write the C++** in the matching `csrc/*.cpp`. Check what you support first, with `NEED(cond, "why")`, before
   launching anything; whatever fails a `NEED` goes to the CPU fallback automatically. Then register it:
   `OMT_IMPL(m, <op>_<overload>, your_fn);` (e.g. `OMT_IMPL(m, where_self_out, where_out_)`).
4. **Build**: `uv run tools/build_native.py` (it compiles every kernel on the GPU and fails on a WGSL error).
5. **Check**: `uv run --extra dev tools/conformance.py --grad --ops <opinfo name>` should report `ok` (and nothing
   should change from `fallback`/`ok` to `mismatch` in a full run). If a model was the motivation,
   `with omarchy_mac_torch.debug.shadow():` around it should report no mismatches.
6. If it's on a hot path, `uv run tools/bench.py` before and after.
7. Before sending it: `uv run --extra dev --extra build tools/check.py` (everything, on your GPU; exits non-zero on
   any failure, so it also works as a `.git/hooks/pre-push`).

## Ground rules

- Correct first: a GPU path that's sometimes wrong is worse than a CPU fallback.
- Kernels borrowed from other projects are welcome; credit them in the file and in the README with their licence.
  Borrow kernels and test ideas, not other projects' architecture (the PyTorch side follows `torch_openreg`).
- There's no unit test suite; the conformance sweep and the `notes/scripts/*_check.py` scripts are the checks.

## Reporting a bug

Use the issue template: it asks for `vulkaninfo`, versions, and `FALLBACKS`. `OMARCHY_MAC_TORCH_DEBUG=1` and
`debug.shadow()` usually point straight at the op.
