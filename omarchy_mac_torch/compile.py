"""torch.compile backend: `torch.compile(model, backend="omarchy_mac_torch")`.

Dynamo captures the model (any structure, not just nn.Sequential) and AOTAutograd turns it into aten graphs for the
forward and backward passes. Four passes rewrite those graphs, then the graph runs op by op on the device:

  1. training BatchNorm2d followed by ReLU -> omt.bn_relu_train (normalize + ReLU in one kernel)
  2. its backward (threshold_backward -> native_batch_norm_backward) -> omt.bn_relu_backward (the ReLU mask is folded
     into the BatchNorm reductions)
  3. conv -> [eval BatchNorm] -> [ReLU] -> omt.conv_act / omt.conv_bn_act_eval (BatchNorm folded into the conv's
     weights, bias and ReLU applied in the GEMM epilogue)
  4. chains of elementwise ops -> one generated kernel each (omt.fused_pointwise)

Every fused op falls back to its plain aten composition when the device path doesn't apply, so a rewritten graph is
always correct. OMARCHY_MAC_TORCH_COMPILE_DEBUG=1 prints what each pass did."""

import hashlib
import math
import operator
import os

import torch
from torch._dynamo import register_backend
from torch._dynamo.backends.common import aot_autograd
from torch.fx import Node

from omarchy_mac_torch.native import DEVICE, core

aten = torch.ops.aten
omt = torch.ops.omt
_DEBUG = os.environ.get("OMARCHY_MAC_TORCH_COMPILE_DEBUG", "0") == "1"


def _val(n):
    return n.meta.get("val") if isinstance(n, Node) else None


def _dev_f32(n, dim=None):
    v = _val(n)
    return (isinstance(v, torch.Tensor) and v.device.type == DEVICE and v.dtype == torch.float32
            and (dim is None or v.dim() == dim))


def _getitems(n):
    return {u.args[1]: u for u in n.users if u.op == "call_function" and u.target is operator.getitem}


def _only_user(n):
    us = list(n.users)
    return us[0] if len(us) == 1 else None


def _replace_tuple(graph, old, new_target, new_args, remap=None):
    """Replace multi-output node `old` (consumed through getitem) by `new_target(*new_args)`; remap: old idx -> new idx"""
    with graph.inserting_before(old):
        new = graph.call_function(new_target, tuple(new_args))
    for i, gi in _getitems(old).items():
        j = remap.get(i, i) if remap else i
        with graph.inserting_before(gi):
            ngi = graph.call_function(operator.getitem, (new, j))
        ngi.meta = dict(gi.meta)
        gi.replace_all_uses_with(ngi)
        graph.erase_node(gi)
    graph.erase_node(old)
    return new


# ---- pass 1 and 2: BatchNorm + ReLU (training) ----

def _bn_relu(graph):
    count = 0
    for n in list(graph.nodes):
        if n.op != "call_function" or n.target is not aten._native_batch_norm_legit_functional.default:
            continue
        x, w, b, rm, rv, training, mom, eps = n.args[:8]
        if not training or any(a is None for a in (w, b, rm, rv)) or not _dev_f32(x, 4):
            continue
        g0 = _getitems(n).get(0)
        relu = _only_user(g0) if g0 is not None else None
        if relu is None or relu.target is not aten.relu.default:
            continue
        with graph.inserting_before(n):
            fused = graph.call_function(omt.bn_relu_train.default, (x, w, b, rm, rv, mom, eps))
        with graph.inserting_before(relu):
            y = graph.call_function(operator.getitem, (fused, 0))
        y.meta = dict(relu.meta)
        relu.replace_all_uses_with(y)
        graph.erase_node(relu)
        graph.erase_node(g0)
        for i, gi in _getitems(n).items():
            with graph.inserting_before(gi):
                ngi = graph.call_function(operator.getitem, (fused, i))
            ngi.meta = dict(gi.meta)
            gi.replace_all_uses_with(ngi)
            graph.erase_node(gi)
        graph.erase_node(n)
        count += 1
    return count


def _bn_relu_backward(graph):
    count = 0
    for n in list(graph.nodes):
        if n.op != "call_function" or n.target is not aten.native_batch_norm_backward.default:
            continue
        g, x, w, rm, rv, mean, invstd, train, eps, mask = n.args[:10]
        if not train or w is None or not _dev_f32(x, 4):
            continue
        if not (isinstance(g, Node) and g.target is aten.threshold_backward.default and g.args[2] == 0 and len(g.users) == 1):
            continue
        gout, y = g.args[0], g.args[1]
        _replace_tuple(graph, n, omt.bn_relu_backward.default, (gout, x, y, w, mean, invstd, list(mask)))
        graph.erase_node(g)
        count += 1
    return count


# ---- pass 3: conv epilogues ----

def _dense_conv(n):
    if n.op != "call_function" or n.target is not aten.convolution.default:
        return False
    x, w, b, stride, padding, dilation, transposed, out_pad, groups = n.args[:9]
    wv = _val(w)
    return (_dev_f32(x, 4) and not transposed and groups == 1 and list(dilation) == [1, 1] and wv is not None
            and wv.shape[2] == wv.shape[3] and len(set(stride)) == 1 and len(set(padding)) == 1)


def _conv_epilogues(graph):
    count = 0
    for n in list(graph.nodes):
        if not _dense_conv(n):
            continue
        x, w, b, stride, padding = n.args[:5]
        user = _only_user(n)
        if user is None:
            continue
        if user.target is aten._native_batch_norm_legit_no_training.default and user.args[0] is n:
            _, bw, bb, rm, rv, mom, eps = user.args[:7]
            g0 = _getitems(user).get(0)
            if any(a is None for a in (bw, bb, rm, rv)) or g0 is None or len(_getitems(user)) != 1:
                continue
            relu = _only_user(g0)
            act = 1 if relu is not None and relu.target is aten.relu.default else 0
            last = relu if act else g0
            with graph.inserting_before(n):
                fused = graph.call_function(omt.conv_bn_act_eval.default,
                                            (x, w, b, bw, bb, rm, rv, eps, list(stride), list(padding), act))
            fused.meta = dict(last.meta)
            last.replace_all_uses_with(fused)
            for dead in ([relu] if act else []) + [g0, user, n]:
                graph.erase_node(dead)
            count += 1
        elif user.target is aten.relu.default:
            with graph.inserting_before(n):
                fused = graph.call_function(omt.conv_act.default, (x, w, b, list(stride), list(padding), 1))
            fused.meta = dict(user.meta)
            user.replace_all_uses_with(fused)
            graph.erase_node(user)
            graph.erase_node(n)
            count += 1
    return count


# ---- pass 4: pointwise fusion with generated WGSL ----

def _lit(v):
    v = float(v)
    if math.isinf(v):
        return "bitcast<f32>(0x7f800000u)" if v > 0 else "bitcast<f32>(0xff800000u)"
    if math.isnan(v):
        return "bitcast<f32>(0x7fc00000u)"
    return f"f32({v!r})"


def _alpha(kw):
    return _lit(kw.get("alpha", 1))


# op -> function(operand expressions, kwargs) -> WGSL expression. Operands are tensors or python scalars.
_PW = {
    aten.add.Tensor: lambda a, kw: f"({a[0]} + {_alpha(kw)} * {a[1]})",
    aten.sub.Tensor: lambda a, kw: f"({a[0]} - {_alpha(kw)} * {a[1]})",
    aten.mul.Tensor: lambda a, kw: f"({a[0]} * {a[1]})",
    aten.div.Tensor: lambda a, kw: f"({a[0]} / {a[1]})",
    aten.add.Scalar: lambda a, kw: f"({a[0]} + {_alpha(kw)} * {a[1]})",
    aten.sub.Scalar: lambda a, kw: f"({a[0]} - {_alpha(kw)} * {a[1]})",
    aten.mul.Scalar: lambda a, kw: f"({a[0]} * {a[1]})",
    aten.div.Scalar: lambda a, kw: f"({a[0]} / {a[1]})",
    aten.rsub.Scalar: lambda a, kw: f"({a[1]} - {_alpha(kw)} * {a[0]})",
    aten.neg.default: lambda a, kw: f"(-{a[0]})",
    aten.relu.default: lambda a, kw: f"max({a[0]}, 0.0)",
    aten.exp.default: lambda a, kw: f"exp({a[0]})",
    aten.log.default: lambda a, kw: f"log({a[0]})",
    aten.sqrt.default: lambda a, kw: f"sqrt({a[0]})",
    aten.rsqrt.default: lambda a, kw: f"inverseSqrt({a[0]})",
    aten.reciprocal.default: lambda a, kw: f"(1.0 / {a[0]})",
    aten.abs.default: lambda a, kw: f"abs({a[0]})",
    aten.tanh.default: lambda a, kw: f"tanh({a[0]})",
    aten.sigmoid.default: lambda a, kw: f"(1.0 / (1.0 + exp(-{a[0]})))",
    aten.silu.default: lambda a, kw: f"({a[0]} / (1.0 + exp(-{a[0]})))",
    aten.maximum.default: lambda a, kw: f"max({a[0]}, {a[1]})",
    aten.minimum.default: lambda a, kw: f"min({a[0]}, {a[1]})",
    aten.pow.Tensor_Scalar: lambda a, kw: f"powf({a[0]}, {a[1]})",
    aten.threshold_backward.default: lambda a, kw: f"select(0.0, {a[0]}, {a[1]} > {a[2]})",
    aten.sigmoid_backward.default: lambda a, kw: f"({a[0]} * {a[1]} * (1.0 - {a[1]}))",
    aten.tanh_backward.default: lambda a, kw: f"({a[0]} * (1.0 - {a[1]} * {a[1]}))",
    aten.silu_backward.default: lambda a, kw: (f"({a[0]} * (1.0 / (1.0 + exp(-{a[1]}))) * (1.0 + {a[1]} * "
                                               f"(1.0 - 1.0 / (1.0 + exp(-{a[1]})))))"),
    aten.gelu.default: lambda a, kw: (f"(0.5 * {a[0]} * (1.0 + tanh(0.7978845608 * ({a[0]} + 0.044715 * {a[0]} * {a[0]} * {a[0]}))))"
                                      if kw.get("approximate", "none") == "tanh" else f"(0.5 * {a[0]} * (1.0 + erf_({a[0]} * 0.7071067812)))"),
    aten.gelu_backward.default: lambda a, kw: (
        f"gelu_bwd_tanh({a[0]}, {a[1]})" if kw.get("approximate", "none") == "tanh" else
        f"({a[0]} * (0.5 * (1.0 + erf_({a[1]} * 0.7071067812)) + {a[1]} * 0.3989422804 * exp(-0.5 * {a[1]} * {a[1]})))"),
    aten.clamp.default: lambda a, kw: f"clamp_({a[0]}, {a[1] if len(a) > 1 and a[1] != 'None' else _lit(-math.inf)}, "
                                      f"{a[2] if len(a) > 2 and a[2] != 'None' else _lit(math.inf)})",
}
_PW_ARGS = {aten.gelu.default: ("approximate",), aten.gelu_backward.default: ("approximate",)}

_HELPERS = """
fn erf_(x: f32) -> f32 {
  let s = sign(x); let z = abs(x);
  let t = 1.0 / (1.0 + 0.3275911 * z);
  return s * (1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-z * z));
}
fn powf(x: f32, p: f32) -> f32 {
  if (p == 2.0) { return x * x; }
  if (p == 1.0) { return x; }
  if (p == 0.5) { return sqrt(x); }
  if (p == 0.0) { return 1.0; }
  let r = pow(abs(x), p);
  if (x >= 0.0) { return r; }
  if (p != floor(p)) { return bitcast<f32>(0x7fc00000u); }
  return select(r, -r, (i32(p) & 1) == 1);
}
fn clamp_(x: f32, lo: f32, hi: f32) -> f32 { return min(max(x, lo), hi); }
fn gelu_bwd_tanh(g: f32, y: f32) -> f32 {
  let k = 0.7978845608 * (y + 0.044715 * y * y * y); let t = tanh(k);
  return g * (0.5 * (1.0 + t) + 0.5 * y * (1.0 - t * t) * 0.7978845608 * (1.0 + 0.134145 * y * y));
}
"""


def _pointwise_ok(n):
    if n.op != "call_function" or n.target not in _PW or not _dev_f32(n) or _val(n).dim() > 8:
        return False
    for a in n.args:
        if isinstance(a, Node):
            if not _dev_f32(a):
                return False
        elif not (isinstance(a, (int, float)) or a is None) or isinstance(a, bool):
            return False
    allowed = set(_PW_ARGS.get(n.target, ())) | {"alpha"}
    return set(n.kwargs) <= allowed


def _pointwise(graph):
    group_of, groups = {}, []
    for n in graph.nodes:
        if not _pointwise_ok(n):
            continue
        shape = tuple(_val(n).shape)
        target = None
        for a in n.args:  # join the group of a producer whose only user is this node (and has the same shape)
            if isinstance(a, Node) and a in group_of and len(a.users) == 1 and tuple(_val(a).shape) == shape:
                target = group_of[a]
                break
        if target is None:
            groups.append([n])
            group_of[n] = len(groups) - 1
            continue
        g = groups[target]
        ext = {a for m in g + [n] for a in m.args if isinstance(a, Node) and a not in g and a is not n}
        if len(ext) > 6:
            groups.append([n])
            group_of[n] = len(groups) - 1
            continue
        g.append(n)
        group_of[n] = target
    count = 0
    for g in groups:
        if len(g) < 2:
            continue
        members = set(g)
        inputs = []
        for m in g:
            for a in m.args:
                if isinstance(a, Node) and a not in members and a not in inputs:
                    inputs.append(a)
        names = {a: f"v{i}" for i, a in enumerate(inputs)}
        body = []
        for j, m in enumerate(g):
            ops = [names[a] if isinstance(a, Node) else ("None" if a is None else _lit(a)) for a in m.args]
            kw = {k: v for k, v in m.kwargs.items()}
            names[m] = f"t{j}"
            body.append(f"  let t{j} = {_PW[m.target](ops, kw)};")
        src = _kernel_source(len(inputs), body, f"t{len(g) - 1}")
        name = "pw_" + hashlib.sha1(src.encode()).hexdigest()[:16]
        core.register_source(name, src)
        last = g[-1]
        with graph.inserting_before(last):
            fused = graph.call_function(omt.fused_pointwise.default, (name, list(inputs)))
        fused.meta = dict(last.meta)
        last.replace_all_uses_with(fused)
        for m in reversed(g):
            graph.erase_node(m)
        count += 1
    return count


def _kernel_source(k, body, result):
    decl = "\n".join(f"@group(0) @binding({i}) var<storage, read> i{i}: array<f32>;" for i in range(k))
    idx = "\n".join(f"  var x{i} = u.off[{i // 4}][{i % 4}];" for i in range(k))
    step = "\n".join(f"    x{i} += c * u.st[{2 * i} + q][r];" for i in range(k))
    loads = "\n".join(f"  let v{i} = i{i}[x{i}];" for i in range(k))
    return f"""// generated by omarchy_mac_torch/compile.py (pointwise fusion)
struct U {{ n: u32, nd: u32, p0: u32, p1: u32, sh: array<vec4<u32>, 2>, st: array<vec4<u32>, 12>, off: array<vec4<u32>, 2> }};
{decl}
@group(0) @binding({k}) var<storage, read_write> o: array<f32>;
@group(0) @binding({k + 1}) var<uniform> u: U;
{_HELPERS}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {{
  let i = gid.x + gid.y * nwg.x * 256u;
  if (i >= u.n) {{ return; }}
{idx}
  var rem = i;
  for (var d = i32(u.nd) - 1; d >= 0; d--) {{
    let q = u32(d) / 4u; let r = u32(d) % 4u;
    let s = u.sh[q][r]; let c = rem % s; rem = rem / s;
{step}
  }}
{loads}
{chr(10).join(body)}
  o[i] = {result};
}}
"""


def _compile(gm, example_inputs):
    g = gm.graph
    stats = {"bn_relu": _bn_relu(g), "bn_relu_backward": _bn_relu_backward(g), "conv_epilogue": _conv_epilogues(g),
             "pointwise_kernels": _pointwise(g)}
    g.eliminate_dead_code()
    g.lint()
    gm.recompile()
    if _DEBUG:
        print("[omarchy-mac-torch compile]", stats)
    from functorch.compile import make_boxed_func
    return make_boxed_func(gm.forward)


backend = aot_autograd(fw_compiler=_compile, bw_compiler=_compile, inference_compiler=_compile)
register_backend(name=DEVICE)(backend)
