"""Loads the C++ backend (_native), registers the "omarchy_mac_torch" PrivateUse1 device with PyTorch the way
PyTorch's reference out-of-tree backend (torch_openreg) does, and hands the backend its GEMM kernel generator."""

import json
import os
import pathlib
import types
import ctypes

import torch

_HERE = pathlib.Path(__file__).parent
DEVICE = "omarchy_mac_torch"


def _check_build():
    info_file = _HERE / "_build_info.json"
    if not info_file.exists() or not any(_HERE.glob("_native*.so")):
        raise ImportError("omarchy-mac-torch: the native backend isn't built; run `uv run tools/build_native.py`")
    built = json.loads(info_file.read_text())["torch"]
    if built.split("+")[0] != torch.__version__.split("+")[0]:
        raise ImportError(f"omarchy-mac-torch was built against torch {built} but torch {torch.__version__} is installed; "
                          "rebuild with `uv run tools/build_native.py`")


_check_build()
ctypes.CDLL(str(_HERE / "libwgpu_native-release.so"), mode=ctypes.RTLD_GLOBAL)
from omarchy_mac_torch import _native as core  # noqa: E402

torch.utils.rename_privateuse1_backend(DEVICE)
# tensor.omarchy_mac_torch(), module.omarchy_mac_torch(), storage methods (used by torch.load's map_location and by
# storage-level copies), is_omarchy_mac_torch
torch.utils.generate_methods_for_privateuse1_backend(for_tensor=True, for_module=True, for_packed_sequence=True,
                                                     for_storage=True)

# ---- GEMM kernels: generated per tile shape on demand ----
from omarchy_mac_torch import gemm as _gemm  # noqa: E402

_KINDS = ["forward", "grad_input", "grad_weight", "forward_bn", "mm", "grad_input_phase"]


def _cdiv(a, b):
    return -(-a // b)


def _conv_planner(kind, n, ci, co, ih, iw, oh, ow, ks, stride, pad, tile_index=-1):
    if kind in (0, 3):
        M, N = n * oh * ow, co
    elif kind == 1:
        M, N = n * ih * iw, ci
    elif kind == 5:  # one stride phase
        M, N = n * _cdiv(ih, stride) * _cdiv(iw, stride), ci
    else:
        M, N = co, ci * ks * ks
    if tile_index >= 0:
        tile = _gemm.TILES[tile_index]
    else:  # untuned default: least padding waste, then the larger register block
        tile = min(_gemm.TILES, key=lambda s: (_cdiv(M, s[0]) * s[0] * _cdiv(N, s[1]) * s[1], -s[2] * s[3]))
    bm, bn, tm, tn = tile
    kernel = f"gemm2:{_KINDS[kind]}:{bm}x{bn}x{tm}x{tn}"
    core.register_source(kernel, _gemm.source(kind, bm, bn, tm, tn))
    return (kernel, bm, bn)


def _mm_planner(a_kfast, b_nfast, tile_index):
    bm, bn, tm, tn = _gemm.TILES[tile_index]
    name = f"mm2:{'k' if a_kfast else 'm'}{'n' if b_nfast else 'k'}:{bm}x{bn}x{tm}x{tn}"
    core.register_source(name, _gemm.source(4, bm, bn, tm, tn, a_kfast, b_nfast))
    return (name, bm, bn)


core.set_conv_planner(_conv_planner)
core.set_mm_planner(_mm_planner)
core.set_fold(int(os.environ.get("OMARCHY_MAC_TORCH_FOLD", "1")))
core.set_tuning(os.environ.get("OMARCHY_MAC_TORCH_TUNE", "1") != "0", len(_gemm.TILES))
core.set_debug(os.environ.get("OMARCHY_MAC_TORCH_DEBUG", "0") == "1")


def validate_kernels():
    """Compile every kernel (the static ones and one of each generated GEMM kind); returns the failures."""
    for kind in (0, 1, 2, 3, 5):
        _conv_planner(kind, 2, 8, 8, 8, 8, 8, 8, 3, 1, 1, 0)
    for a in (False, True):
        for b in (False, True):
            _mm_planner(a, b, 0)
    return core.validate_kernels()


# ---- the device module: torch.omarchy_mac_torch.* ----
_mod = types.ModuleType("torch." + DEVICE)
_mod.is_available = lambda: True
_mod.is_initialized = lambda: True
_mod._lazy_init = lambda: None
_mod._is_in_bad_fork = lambda: False
_mod.device_count = lambda: 1
_mod.current_device = lambda: 0
_mod.set_device = lambda d: None
_mod.get_amp_supported_dtype = lambda: [torch.float32]
_mod.synchronize = lambda device=None: core.sync()
_mod.empty_cache = core.empty_cache
_mod.memory_allocated = lambda device=None: core.memory_allocated()
_mod.memory_reserved = lambda device=None: core.memory_reserved()
_mod.manual_seed = lambda seed: core.manual_seed(int(seed))
_mod.manual_seed_all = lambda seed: core.manual_seed(int(seed))  # torch.manual_seed seeds this too
_mod.seed = core.seed
_mod.seed_all = core.seed
_mod.initial_seed = core.initial_seed
_mod.get_rng_state = lambda device=None: core.get_rng_state()
_mod.set_rng_state = lambda state, device=None: core.set_rng_state(state)


class _DeviceContext:
    def __init__(self, device):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


_mod.device = _DeviceContext
torch._register_device_module(DEVICE, _mod)


def synchronize():
    core.sync()


def set_queue_depth(depth=3, chunk=64):
    """At most `depth` chunks of up to `chunk` dispatches are queued on the GPU, so the desktop compositor never waits
    behind more than a few milliseconds of training work. Smaller values keep the desktop smoother."""
    core.set_queue(chunk, depth)


def empty_cache():
    core.empty_cache()


def memory_allocated():
    return core.memory_allocated()


def memory_reserved():
    return core.memory_reserved()


class _Fallbacks:
    """Ops that ran on the CPU instead of the GPU, with why: {"op (reason)": count}. Read-only mapping view."""

    def _d(self):
        return core.fallback_counts()

    def keys(self):
        return self._d().keys()

    def items(self):
        return self._d().items()

    def __getitem__(self, k):
        return self._d()[k]

    def __iter__(self):
        return iter(self._d())

    def __len__(self):
        return len(self._d())

    def __bool__(self):
        return bool(self._d())

    def clear(self):
        core.reset_fallbacks()

    def __repr__(self):
        return repr(self._d())


FALLBACKS = _Fallbacks()

# ---- on-disk cache of the GEMM tuning (a few seconds of timing per new convolution shape) ----
import atexit  # noqa: E402

_TUNE_FILE = os.path.expanduser("~/.cache/omarchy-mac-torch/tuning.json")
try:
    with open(_TUNE_FILE) as f:
        core.tune_load({k: tuple(v) for k, v in json.load(f).items()})
except (OSError, ValueError, TypeError):
    pass


def _save_tuning():
    try:
        os.makedirs(os.path.dirname(_TUNE_FILE), exist_ok=True)
        old = {}
        try:
            with open(_TUNE_FILE) as f:
                old = json.load(f)
        except (OSError, ValueError):
            pass
        old.update({k: list(v) for k, v in core.tune_dump().items()})
        with open(_TUNE_FILE, "w") as f:
            json.dump(old, f)
    except Exception:
        pass


atexit.register(_save_tuning)
