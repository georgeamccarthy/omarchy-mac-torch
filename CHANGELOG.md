# Changelog

## 0.2.0

A restructure for reliability. Measured on an M1 Pro: AlexNet training 167 -> 160 ms/step (3.1x the CPU), ResNet-18
3.6x, PyTorch's OpInfo sweep (~640 ops, forward and gradients) with no wrong answers.

Fixed (wrong results before):
- `avg_pool2d` ignored its kernel size, stride and padding (always 2x2).
- comparisons (`<`, `>`, `==`, ...) ignored broadcasting.
- `max_pool2d(..., return_indices=True)` returned the wrong indices for 2x2 windows.
- NaN handling in comparisons, `isnan`, `argmax` and max pool (the shader compiler may assume NaN never occurs).
- layer norm with only a bias, or only a weight, computed wrong gradients.
- `num_batches_tracked` wasn't updated by fused stacks; batch-norm statistics are now computed the stable way
  everywhere.
- strided bool tensors were allocated a quarter of the memory they need.
- bind groups leaked on `empty_cache()`; the download staging buffer leaked when it grew.

Changed:
- Unsupported cases fall back to the CPU instead of raising (softmax on 3-D tensors, argmax over other dims, addmm
  with alpha/beta, reductions over non-adjacent dims, ...). One wrapper does this for every op.
- Tensors store exactly PyTorch's bytes for every dtype (int64 as 8 bytes, bool as 1). `torch.save`, `deepcopy`,
  `.view(dtype)` and storage copies work through PyTorch's own storage path; the patches to `Tensor.__deepcopy__`,
  `__reduce_ex__`, `F.cross_entropy` and `nn.Module.to` are gone.
- Fused conv/BN/ReLU/pool stacks are applied automatically on first use (PyTorch's global forward hook) instead of by
  patching `Module.to`; `accelerate(model)` still does it explicitly.
- Device generator, `torch.manual_seed` seeding, RNG state, events and stream queries now work.
- Uploads no longer wait for the GPU; waits block on the exact submission instead of polling.
- Memory pool size classes are finer above 1 MiB (at most 12.5% waste instead of up to 50%).
- GPU errors are raised as Python exceptions; `OMARCHY_MAC_TORCH_DEBUG=1` names the op; out-of-memory frees the cache
  and retries, then raises `torch.OutOfMemoryError`.
- The backend refuses to load against a different torch than it was built for.

Added: strided convolution input gradients on the GPU, conv bias in the GEMM epilogue, in-place elementwise ops,
general strided elementwise kernel (any rank, any strides), softmax over any dim, workgroup-tree and split
reductions (a 16M-element sum: 36 ms -> 0.6 ms), general average pooling, `nll_loss`, GELU, AdamW, cat/stack along
any dim, `masked_fill`, integer add, `tools/conformance.py`, `tools/acceptance.py`, `tools/bench.py`, shadow mode,
strict mode, lavapipe CI.

Removed: the original Python/wgpu-py backend (`OMARCHY_MAC_TORCH_BACKEND=python`).
