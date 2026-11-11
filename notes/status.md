# status

what works, what doesn't, and what's next. measured on one machine (macbook pro 16" m1 pro, omarchy, honeykrisp).

## works
- native c++ backend (`csrc/`), device `omarchy_mac_torch`, registered the torch_openreg way: guard with events,
  hooks, device generator, storage-level serialization. tensors hold pytorch's exact bytes for every dtype.
- every op goes through one wrapper (`Safe<>` in `omt.h`): unsupported cases fall back to the cpu, never raise.
- convs: dense up to 11x11, stride 1-4, any padding, fwd + all grads at any stride, bias in the gemm. depthwise 3/5/7.
- general strided elementwise kernel (8 dims, any strides, in place, aliasing-safe), comparisons with broadcasting,
  where / masked_fill, softmax over any dim, sum/mean/argmax over any dims (tree and split reductions), nll_loss,
  pooling with pytorch's full semantics, batch/layer norm, dropout, fused sgd/adam/adamw.
- fused conv/bn/relu/pool stacks, applied automatically (1.2-1.9x on small convnets).
- conformance sweep over pytorch's opinfo db: ~640 ops, 0 mismatches, ~1/3 fully on the gpu.

## doesn't
- transformers are slow (~0.2x cpu): attention's masking ops fall back. top of `docs/roadmap.md`.
- fp16/bf16 math, dilated/transposed/grouped convs, strided depthwise.
- one in-order queue: streams are trivial, events are submission indices.
- only tested on an m1 pro. m3/m4 wait on the linux gpu driver.

## checks
`tools/check.py` runs them all locally (build, check scripts, conformance sweep; `--full` adds acceptance and
benchmarks). also `debug.shadow()` for one model.

## next
see `docs/roadmap.md`.
