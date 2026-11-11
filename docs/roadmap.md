# Roadmap

Ordered by payoff. The conformance sweep (`results/conformance.json`) and acceptance table say where the time goes;
re-rank from them.

## 1. Transformers on the GPU
Done: attention's masking ops (`isneginf`, `all`/`any`, `where(out=)`, `tril`/`triu`, `arange`), embedding backward
(scatter-add by compare-and-swap) and batched `bmm` (one dispatch per batch instead of one per matrix) moved ViT from
0.17x to 1.2x the CPU and GPT from 0.19x to ~1.0-1.3x, with no fallbacks. Next: fewer elementwise dispatches (graph
replay, #4), and a fused attention kernel (softmax(QK^T)V in one pass, flash-attention style).

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
Done: a `torch.compile` backend (`compile.py`) that works on any model structure: BN+ReLU fused forward and backward,
bias/ReLU in the conv GEMM epilogue, eval BN folded into conv weights, generated kernels for elementwise chains. It is
as accurate as eager mode but only about as fast (ResNet-18 108 vs 109 ms; SeparableCNN 22 ms vs 17 ms with the fused
stacks). What the stacks still have: a prebuilt plan owns its memory and replays a fixed list of dispatches (1 ms of CPU
per step vs 5 ms running the graph op by op), and no copy-back of BatchNorm buffers after each step.
Next: graph capture and replay for compiled graphs (static memory planned at compile time, the dispatch list recorded
once, like CUDA graphs), and keeping BatchNorm buffer updates inside the graph. Then retire `plans.cpp` and `stacks.py`.

## 5. fp16 / bf16
`shader-f16` storage and math as a code-generation option in `gemm.py` and `ew.wgsl` (ONNX Runtime and ggml-vulkan
have fp16 kernels to borrow). Halves memory traffic, which is most of the elementwise cost.

## 6. Project
- Prebuilt wheels per torch version (the hatch hook already tags them per platform), a release workflow.
- `tools/check.py --full` as a pre-push hook, so every push has run the sweep and benchmarks on real hardware.
- Reports from M1/M2 Max/Ultra and M2 machines.

See also `docs/driver-roadmap.md` for driver-side work (Honeykrisp, the Asahi kernel driver).
