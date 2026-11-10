<p align="center">
  <img src="docs/alexnet.svg" alt="AlexNet training on the CPU vs the GPU, side by side. The GPU is about 3x faster." width="100%">
</p>

# omarchy-mac-torch

pytorch on the gpu of an apple silicon mac running omarchy. it's linux, not macos.

on macos pytorch has `mps`. on linux there's no metal, so pytorch on a mac just runs on the cpu even though the gpu has a good open source vulkan driver (honeykrisp, in mesa). this is a pytorch backend that runs on that driver. i built it because i wanted my omarchy mac to train things on its own gpu.

unofficial. not affiliated with omarchy, asahi linux or mesa, just built on top of their work.

```python
import torch
import omarchy_mac_torch              # registers the device

dev = "omarchy_mac_torch"
model = MyModel().to(dev)
loss = torch.nn.functional.cross_entropy(model(x.to(dev)), y.to(dev))
loss.backward()                       # normal pytorch from here: optimizers, state_dict, torch.save
```

the device string is long on purpose, it matches the project name. set `dev` once and move on.

## how fast

alexnet (61M params, 224x224), same model, same data, same optimizer on both. macbook pro 16" (m1 pro), omarchy, pytorch 2.15. the cpu runs on the 4 fastest cores because that's where it's fastest on this machine.

| alexnet | cpu | gpu | speedup |
|---|---|---|---|
| **training**, batch 16 | 441 ms/step (36 img/s) | 148 ms/step (108 img/s) | **3.0x** |
| **inference**, batch 16 | 61.7 ms (259 img/s) | 49.2 ms (325 img/s) | 1.25x |
| **inference**, batch 64 | 216.8 ms (295 img/s) | 166.8 ms (384 img/s) | 1.3x |

the animation up top is the training row, played back at the real measured speed (`notes/scripts/record_alexnet.py` records it, `tools/make_animation.py` draws it).

training gets the big win because the cpu is slow at backward passes. inference is only a little faster because the cpu is already decent at forward only alexnet and the convolution kernels here run at a few hundred gflops on a gpu that can do several thousand. there's headroom, i just haven't gone after it yet.

## what works right now

- convolutions: any kernel up to 11x11, stride 1 to 4, any padding, forward and backward (a strided conv needs no input gradient on the gpu path, so it's fine as a first layer like alexnet's). depthwise 3x3, 5x5, 7x7. fused conv + batchnorm + relu + pool stacks.
- batchnorm, max pool (any window and stride), average pool, adaptive average pool, relu / silu / sigmoid / tanh / hardswish, dropout, layer norm, softmax, cross entropy.
- linear layers and matmul (including batched), elementwise math, reductions, indexing and cat / stack along the first dim.
- sgd (momentum, nesterov) and adam, fused so the update is one pass over the weights.
- `torch.save`, `state_dict`, `deepcopy`, everything you'd expect from a normal device.
- anything it doesn't have falls back to the cpu automatically and still gives the right answer, just slowly. `omarchy_mac_torch.FALLBACKS` tells you what fell back.

what's not there: fp16 / bf16 (float32 only), dilated / transposed / grouped convs (other than depthwise), attention, and most of pytorch's op surface. it's about 90 ops out of roughly 2000. it does convnets well, it's not a drop in for everything yet.

tested on: macbook pro 16" m1 pro, omarchy (arch linux arm), mesa's honeykrisp. nothing else. other m1 / m2 machines on the same driver should work and i'd love a report. m3 and m4 depend on the linux gpu driver getting there, not on this.

## where this goes

i want to keep adding to it: more ops, fp16, faster kernels, more chips as the driver supports them. but honestly the best way to get what you need is to **clone it, add the op or feature you need as you go, and send a pr.** that's how i built most of it: run a model, see what falls back, fix that. agent prs are welcome too, if your agent hits a missing op and writes it, send it.

how to find your way around:

- `csrc/native.cpp` is the backend: allocator, command queue, every op, the cpu fallback, the fused stacks.
- `omarchy_mac_torch/gemm.py` generates the convolution and matmul kernels (wgsl). `native.py` registers the kernels and the small pytorch patches.
- there are no unit tests here. the checks are scripts in `notes/scripts/` that compare the gpu against the cpu: `native_final_check.py` (models and ops), `alexnet_ops_check.py` (conv and pool ops), and a few more. run them before and after you change something.
- `notes/` has what i've found about performance and what's missing.

## install

```sh
git clone https://github.com/georgeamccarthy/omarchy-mac-torch
cd omarchy-mac-torch
uv sync --extra build               # torch, numpy, ninja, wgpu (for the wgpu-native library + headers)
uv run tools/build_native.py        # builds the backend, the path must not contain spaces
uv run python notes/scripts/native_final_check.py
uv run python notes/scripts/alexnet_benchmark.py gpu    # and `cpu` to compare
```

needs linux on an apple silicon mac with the honeykrisp vulkan driver working (`vulkaninfo` should list "Apple M1 Pro" or similar), python 3.13, pytorch 2.15. the backend is built against one pytorch version so rebuild when you change it.

## credit

this stands on a lot of other people's work:

- [joshua warren](https://joshuawarren.com/projects/): [omarchy-mlx](https://github.com/joshuaswarren/omarchy-mlx) and [metal2vk](https://github.com/joshuaswarren/metal2vk) got me thinking about running ml on this gpu through vulkan, and they're the closest prior art for the omarchy-on-a-mac side of this.
- [asahi linux](https://asahilinux.org/) and [mesa's honeykrisp](https://docs.mesa3d.org/drivers/asahi.html), the people who made the gpu usable on linux at all. none of this exists without that driver.
- [omarchy](https://omarchy.org/), the distro i run this on.
- [wgpu-native](https://github.com/gfx-rs/wgpu-native), which runs the wgsl kernels over vulkan.
- [pytorch](https://pytorch.org/) and its `PrivateUse1` out-of-tree backend support, plus [pytorch_dlprim](https://github.com/artyom-beilis/pytorch_dlprim) (an opencl backend) as a reference for what a backend like this needs.

## license

mit, see [LICENSE](LICENSE).
