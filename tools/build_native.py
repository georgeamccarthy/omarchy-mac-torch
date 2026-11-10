"""Build omarchy_mac_torch/_native.so (the C++ backend) against torch and the wgpu-native library that ships with wgpu-py.
usage: uv run tools/build_native.py   (build dir must not contain spaces)"""
import os, pathlib, shutil, sys
import torch, wgpu
from torch.utils.cpp_extension import load

root = pathlib.Path(__file__).resolve().parent.parent
res = pathlib.Path(wgpu.__file__).parent / "resources"
lib = res / "libwgpu_native-release.so"
shutil.copy(lib, root / "omarchy_mac_torch" / lib.name)
build = root / "omarchy_mac_torch" / "_build_native"
build.mkdir(exist_ok=True)
shutil.copy(lib, build / lib.name)  # so the extension imports from the build dir too (rpath $ORIGIN)
mod = load(name="_native", sources=[str(root / "csrc" / "native.cpp")], extra_include_paths=[str(res)],
           extra_cflags=["-O2", "-std=c++20", "-fvisibility=default"],
           extra_ldflags=[f"-L{root / 'omarchy_mac_torch'}", "-l:libwgpu_native-release.so", f"-Wl,-rpath,{root / 'omarchy_mac_torch'}"],
           build_directory=str(build), verbose=True)
for f in build.glob("_native*.so"):
    shutil.copy(f, root / "omarchy_mac_torch" / f.name)
print("built", [p.name for p in (root / "omarchy_mac_torch").glob("_native*.so")])
