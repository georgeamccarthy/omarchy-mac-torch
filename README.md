<p align="center">
  <img src="docs/alexnet.svg" alt="AlexNet training on the CPU vs the GPU, side by side. The GPU is about 3x faster." width="100%">
</p>

# omarchy-mac-torch

PyTorch on the GPU of an Apple Silicon Mac running Omarchy.

On macOS, PyTorch has `mps`. On Linux there is no Metal, so PyTorch on a Mac runs on the CPU even though the GPU has a good open source Vulkan driver (Honeykrisp, in Mesa). This is a PyTorch backend that runs on that driver. I wanted my Omarchy MacBook to train models on its own GPU.

Unofficial. Not affiliated with Omarchy, Asahi Linux or Mesa, just built on top of their work.

```python
import torch
import omarchy_mac_torch              # registers the device

dev = "omarchy_mac_torch"
model = MyModel().to(dev)
loss = torch.nn.functional.cross_entropy(model(x.to(dev)), y.to(dev))
loss.backward()                       # normal PyTorch from here: optimizers, state_dict, torch.save
```

The device name is long because it matches the project name. Set `dev` once and forget it.

## Speed

AlexNet (61M parameters, 224x224), same model, data and optimizer on both. MacBook Pro 16" (M1 Pro) on Omarchy, PyTorch 2.15. The CPU uses the 4 fastest cores, which is where it is quickest on this machine.

| AlexNet | CPU | GPU | Speedup |
|---|---|---|---|
| **Training**, batch 16 | 441 ms/step (36 img/s) | 148 ms/step (108 img/s) | **3.0x** |
| **Inference**, batch 16 | 61.7 ms (259 img/s) | 49.2 ms (325 img/s) | 1.25x |
| **Inference**, batch 64 | 216.8 ms (295 img/s) | 166.8 ms (384 img/s) | 1.3x |

The animation at the top is the training row, played back at the measured speed. `notes/scripts/record_alexnet.py` records it and `tools/make_animation.py` draws it.

Training gains the most because the CPU is slow at backward passes. Inference only gains a little because the CPU is already decent at forward-only AlexNet, and the convolution kernels here reach a few hundred GFLOP/s on a GPU that can do several thousand. There is room to go faster.

## What works

- Convolutions: kernels up to 11x11, stride 1 to 4, any padding, forward and backward. A strided conv doesn't need an input gradient on the GPU path, so it works as a first layer like AlexNet's. Depthwise 3x3, 5x5 and 7x7. Fused conv + batchnorm + ReLU + pool stacks.
- Batchnorm, max pool (any window and stride), average pool, adaptive average pool, ReLU / SiLU / sigmoid / tanh / hardswish, dropout, layer norm, softmax, cross entropy.
- Linear layers and matmul (including batched), elementwise math, reductions, indexing, and cat / stack along the first dim.
- SGD (momentum, nesterov) and Adam, fused so the update is one pass over the weights.
- `torch.save`, `state_dict`, `deepcopy` and the other things you'd expect from a normal device.
- Anything missing falls back to the CPU and still gives the right answer, just slowly. `omarchy_mac_torch.FALLBACKS` lists what fell back.

What isn't there: fp16 / bf16 (float32 only), dilated, transposed and grouped convs (other than depthwise), attention, and most of PyTorch's op surface. It covers about 90 ops out of roughly 2000. It does convnets well but it isn't a drop-in replacement yet.

Tested on a MacBook Pro 16" M1 Pro, Omarchy (Arch Linux ARM), Mesa's Honeykrisp. Nothing else. Other M1 and M2 machines on the same driver should work, and a report would be useful. M3 and M4 depend on the Linux GPU driver supporting them, not on this project.

## Where this goes

I want to add more ops, fp16, faster kernels, and more chips as the driver supports them. The quickest way to get what you need is to clone it, add the op you're missing, and send a PR. That's how most of it got built: run a model, see what falls back, fix that. Agent PRs are welcome too. If your agent hits a missing op and writes it, send it.

Finding your way around:

- `csrc/native.cpp` is the backend: allocator, command queue, every op, the CPU fallback and the fused stacks.
- `omarchy_mac_torch/gemm.py` generates the convolution and matmul kernels (WGSL). `native.py` registers the kernels and a few small PyTorch patches.
- There are no unit tests. The checks are scripts in `notes/scripts/` that compare the GPU against the CPU: `native_final_check.py` (models and ops), `alexnet_ops_check.py` (conv and pool ops) and a few more. Run them before and after a change.
- `notes/` has what I found about performance and what's missing.

## Install

```sh
git clone https://github.com/georgeamccarthy/omarchy-mac-torch
cd omarchy-mac-torch
uv sync --extra build               # torch, numpy, ninja, wgpu (for the wgpu-native library and headers)
uv run tools/build_native.py        # builds the backend, the path must not contain spaces
uv run python notes/scripts/native_final_check.py
uv run python notes/scripts/alexnet_benchmark.py gpu    # and `cpu` to compare
```

You need Linux on an Apple Silicon Mac with the Honeykrisp Vulkan driver working (`vulkaninfo` should list "Apple M1 Pro" or similar), Python 3.13 and PyTorch 2.15. The backend is built against one PyTorch version, so rebuild when you change it.

## Credit

This builds on a lot of other people's work.

- [omarchy-mlx](https://github.com/joshuaswarren/omarchy-mlx) runs MLX on the Apple GPU under Omarchy through Honeykrisp, and [metal2vk](https://github.com/joshuaswarren/metal2vk), from the same project, compiles Metal kernels to Vulkan. This project is the PyTorch version of the same idea, and omarchy-mlx is the closest prior art.
- [Asahi Linux](https://asahilinux.org/) and [Mesa's Honeykrisp](https://docs.mesa3d.org/drivers/asahi.html) made the GPU usable on Linux at all. None of this works without that driver.
- [Omarchy](https://omarchy.org/), which I run this on.
- [wgpu-native](https://github.com/gfx-rs/wgpu-native), which runs the WGSL kernels over Vulkan.
- [PyTorch](https://pytorch.org/) and its `PrivateUse1` out-of-tree backend support, and [pytorch_dlprim](https://github.com/artyom-beilis/pytorch_dlprim) (an OpenCL backend), which I used as a reference for what a backend needs.

## License

MIT, see [LICENSE](LICENSE).
