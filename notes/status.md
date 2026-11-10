# status

what works, what doesn't, and what i'd do next. measured on one machine (macbook pro 16" m1 pro, omarchy, honeykrisp).

## works
- native c++ pytorch backend (`csrc/native.cpp`), device string `omarchy_mac_torch`. ordinary tensors on pooled gpu buffers, ~90 aten ops as wgsl kernels, boxed cpu fallback for everything else.
- convs: dense up to 11x11, stride 1-4, any padding, fwd + weight grad (+ input grad for stride 1). depthwise 3/5/7 stride 1. strided conv with a host input gradient falls back.
- pooling: max (any window/stride, gather backward), avg, adaptive avg. norm: batchnorm, layernorm. activations, dropout, softmax, cross entropy.
- matmul/linear via the same register-blocked gemm template (strided operands, split-k for skinny shapes), bmm.
- fused sgd and adam, fused conv/bn/relu/pool stacks (training and eval, bn folded into the next 1x1 conv in eval).
- bounded gpu queue that adapts its chunk size to keep ~8 ms of queued work so the desktop stays smooth.
- gemm tile + split-k autotuning per conv shape, cached in `~/.cache/omarchy-mac-torch/tuning.json`.
- `torch.save`, `state_dict`, `deepcopy` on device tensors.

## doesn't
- fp16 / bf16 / float64 (float32 only, ints/bools stored as 4 byte words).
- dilated, transposed, grouped (non-depthwise) convs: run on the host.
- attention, scatter/gather beyond dim 0, most of the long tail of ops (they fall back, correctly).
- streams/events are stubs (one in-order queue). no multi-device.
- fragile pytorch integration: `nn.Module.to`, `Adam/SGD.__init__`, `F.cross_entropy`, `Tensor.__deepcopy__` / `__reduce_ex__` are monkeypatched.
- only tested on an m1 pro. m3/m4 wait on the linux gpu driver.

## checks (no unit tests, scripts compare gpu vs cpu)
`notes/scripts/native_final_check.py`, `alexnet_ops_check.py`, `activation_check.py`, `conv_kinds_check.py`, `eval_check.py`, `grad_accum_check.py`.

## next, roughly by payoff
1. faster conv gemms. they run at a few hundred gflops, the gpu can do several thousand. hoist the im2col index math out of the k loop, try fp16 storage (needs shader-f16).
2. in-place elementwise kernels (an in-place op is currently out-of-place + a copy back).
3. more ops as people need them, send prs.
4. a real license/ci/wheels story, and replace the monkeypatches with pytorch hooks.
