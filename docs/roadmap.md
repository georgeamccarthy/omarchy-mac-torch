# Roadmap

Ordered by payoff. The conformance sweep (`results/conformance.json`) and acceptance table say where the time goes;
re-rank from them.

## 1. Transformers on the GPU
ViT and GPT train correctly but at ~0.2x the CPU because a handful of small ops in attention fall back, each costing a
GPU sync and two copies: `isneginf`, `all` (dim reductions over bool), `where.self_out`, `tril`/`triu`,
`embedding_dense_backward` (a scatter-add: float atomics via compare-and-swap on u32, or a sort-based kernel),
`arange` on the device. After those, a fused attention kernel (softmax(QK^T)V in one pass, flash-attention style;
ONNX Runtime's WebGPU provider has WGSL to start from).

## 2. The fallback long tail
From the sweep's ranking: `mul`/`div` with mixed dtypes (float x bool/int: cast in the kernel), `gather`/`scatter`,
`sum` over bool/int with dims, `amax`/`amin`, `linalg_vector_norm` (clip_grad_norm), `sort`, `flip`. Strided and
grouped convolutions (MobileNet-style downsampling; a loop over groups with channel offsets in the GEMM loaders).

## 3. Faster GEMMs
Convolutions run at a few hundred GFLOP/s on a GPU that can do several thousand. Options, cheapest first: hoist the
im2col index math out of the k loop; vec4 loads; larger register tiles; cooperative matrix (Apple's simdgroup
matrices, `VK_KHR_cooperative_matrix`, patches in Joshua Warren's Mesa fork). ggml-vulkan's matmul shaders are already
tuned on Honeykrisp and CubeCL has an autotuned WGSL matmul: borrow from them.

## 4. General fusion instead of fused stacks
`plans.cpp` fuses conv/BN/ReLU/pool only inside `nn.Sequential` with specific shapes (1.2-1.9x on small convnets).
The general version: BN, activation and bias folded into the conv GEMM epilogue for any model structure, or a
`torch.compile` backend that sees the whole graph. Once that exists, retire `plans.cpp` and `stacks.py`.

## 5. fp16 / bf16
`shader-f16` storage and math as a code-generation option in `gemm.py` and `ew.wgsl` (ONNX Runtime and ggml-vulkan
have fp16 kernels to borrow). Halves memory traffic, which is most of the elementwise cost.

## 6. Project
- Prebuilt wheels per torch version (the hatch hook already tags them per platform), a release workflow.
- `tools/check.py --full` as a pre-push hook, so every push has run the sweep and benchmarks on real hardware.
- Reports from M1/M2 Max/Ultra and M2 machines.

See also `docs/driver-roadmap.md` for driver-side work (Honeykrisp, the Asahi kernel driver).
