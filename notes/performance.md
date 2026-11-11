# performance notes

measured on a macbook pro 16" (m1 pro), omarchy, honeykrisp, pytorch 2.15, cpu on the 4 fastest cores.

## alexnet (61M params, 224x224)
| | cpu | gpu | speedup |
|---|---|---|---|
| training, batch 16 | 441 ms/step | 148 ms/step | 3.0x |
| inference, batch 16 | 61.7 ms | 49.2 ms | 1.25x |
| inference, batch 64 | 216.8 ms | 166.8 ms | 1.3x |

how it got there (training, batch 16): 456 ms on the cpu. first gpu run 3.8 s/step, because the 11x11 stride-4 first conv and the three 3x3 stride-2 max pools ran on the host. strided conv + general max pool kernels: 191 ms. fused sgd (the momentum update was a quarter of the step) and the matmul through the gemm template with split-k: 147 ms.

## where the time goes (native_kernel_times.py / alexnet_profile.py, each kernel run alone)
- convolution gemms are about half of a training step and run at 400-500 gflops. the gpu has several thousand. most of the gap is the loaders doing im2col index math for every element (integer divisions in the k loop) and the small tiles.
- the cpu side is not a factor: issuing a whole step costs 1-2 ms (about 1.6 us per op), the step is entirely gpu bound.
- a dependent dispatch costs 12-19 us on the gpu (dispatch_latency.py). dispatch count matters less than kernel efficiency.

## things that were tried
- batchnorm statistics/backward reductions split over 8 workgroups per channel + a finalize kernel: ~9% on a small convnet.
- gemm tile and split-k autotuning per shape: ~5%.
- folding batchnorm into the next conv's input loader in eval: no measurable win for 3x3 (the loader redoes the transform 9x per element), kept only for 1x1.
- a plain register-blocked matmul with a 64x64 tile was slower than a 16x16 one for tall-skinny classifier shapes (too few workgroups). fixed by using split-k.
- idle-gap throttling to keep the desktop smooth cost ~30% for a nominal 10%. replaced by a bounded queue whose chunk size adapts to measured gpu time.

## desktop smoothness
the gpu is shared with the compositor. the queue limiter keeps at most ~8 ms of queued gpu work (3 chunks, chunk size adapts), so a frame never waits long behind a training step. if you see lag, call `omarchy_mac_torch.set_queue_depth(depth, chunk)` with a smaller target.

## gemm template, round 2 (0.2.x)
measured with `tools/gemm_bench.py` (alexnet's convs at batch 16, plus matmul):
- shared-memory tiles as `vec4` arrays and unrolled loaders: conv 540 -> ~730 gflops, matmul 600 -> 880. the inner
  loop had been doing 12 scalar shared loads per 32 fmas; now 3 vec4 loads.
- strided grad-input split into stride phases (one dense gemm per (py, px), no skipped taps): alexnet's stride-4 first
  layer 844 -> 50 ms; resnet's stride-2 layers stop paying 4x.
- alexnet training 160 -> 128 ms/step (4.3x the cpu), resnet-18 131 -> 103 (4.9x).

tried and reverted (both slower because live registers cost more than integer divisions here; kernels start spilling):
- prefetching the next k-tile into registers (software pipelining): -50%.
- hoisting each loader slot's row decomposition out of the k loop + a k-side index table instead of divisions: -15-20%.

next levers: subgroup operations (share operands across a simd instead of through shared memory) and cooperative
matrix (`docs/roadmap.md` #3). the template is at ~15% of fp32 peak.
