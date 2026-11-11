// Fused ops that the torch.compile backend (omarchy_mac_torch/compile.py) rewrites graphs to use. Each has a
// composite implementation in plain aten ops (used on other devices and whenever the fast path doesn't apply), so a
// rewritten graph is always correct.
//
//   omt::bn_relu_train      training BatchNorm2d + ReLU: shifted split statistics, normalize + ReLU in one kernel
//   omt::bn_relu_backward   its backward: ReLU mask folded into the BN backward reductions and apply
//   omt::conv_act           conv2d + bias + optional ReLU in the GEMM epilogue
//   omt::conv_bn_act_eval   eval conv2d + BatchNorm (folded into the conv's weights and bias) + optional ReLU
//   omt::fused_pointwise    a chain of elementwise ops compiled into one generated kernel
#include "omt.h"


namespace omt {

// ------------------------------------------------------------------------------------------------
// composite (reference) implementations
// ------------------------------------------------------------------------------------------------

static std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor> bn_relu_train_ref(const Tensor& x, const Tensor& w, const Tensor& b,
                                                                          const Tensor& rm, const Tensor& rv, double mom, double eps) {
  auto r = at::_native_batch_norm_legit_functional(x, w, b, rm, rv, true, mom, eps);
  return {at::relu(std::get<0>(r)), std::get<1>(r), std::get<2>(r), std::get<3>(r), std::get<4>(r)};
}
static std::tuple<Tensor, Tensor, Tensor> bn_relu_backward_ref(const Tensor& g, const Tensor& x, const Tensor& y, const Tensor& w,
                                                               const Tensor& mean, const Tensor& invstd, std::array<bool, 3> mask) {
  Tensor gz = at::threshold_backward(g, y, 0);
  return at::native_batch_norm_backward(gz, x, w, std::nullopt, std::nullopt, mean, invstd, true, 0.0, mask);
}
static Tensor conv_act_ref(const Tensor& x, const Tensor& w, const std::optional<Tensor>& b, IntArrayRef stride,
                           IntArrayRef padding, int64_t act) {
  Tensor y = at::convolution(x, w, b, stride, padding, {1, 1}, false, {0, 0}, 1);
  return act == 1 ? at::relu(y) : y;
}
static std::pair<Tensor, Tensor> fold_bn(const Tensor& cw, const std::optional<Tensor>& cb, const Tensor& w, const Tensor& b,
                                         const Tensor& rm, const Tensor& rv, double eps) {
  Tensor scale = w * at::rsqrt(rv + eps);  // per output channel
  Tensor fw = cw * scale.view({-1, 1, 1, 1});
  Tensor fb = (cb && cb->defined() ? (*cb - rm) : -rm) * scale + b;
  return {fw, fb};
}
static Tensor conv_bn_act_eval_ref(const Tensor& x, const Tensor& cw, const std::optional<Tensor>& cb, const Tensor& w,
                                   const Tensor& b, const Tensor& rm, const Tensor& rv, double eps, IntArrayRef stride,
                                   IntArrayRef padding, int64_t act) {
  auto f = fold_bn(cw, cb, w, b, rm, rv, eps);
  return conv_act_ref(x, f.first, f.second, stride, padding, act);
}
static Tensor fused_pointwise_ref(c10::string_view, at::TensorList) {
  TORCH_CHECK(false, "omt::fused_pointwise only runs on omarchy_mac_torch tensors");
}

// ------------------------------------------------------------------------------------------------
// device implementations
// ------------------------------------------------------------------------------------------------

static Buf* qbuf(int64_t n, int64_t c, int64_t h, int64_t w, int64_t p0, float eps) {
  return uniform({u(n), u(c), u(h), u(w), 0, 0, u(p0), fbits(eps)});
}
static void ew_launch(const char* k, const std::vector<Binding>& b, int64_t total) {
  uint32_t gx, gy;
  grid(total, 256, gx, gy);
  launch(k, b, gx, gy);
}
static bool whole(const Tensor& t) { return f32(t) && t.is_contiguous() && t.storage_offset() == 0; }

// The new running stats are written to new tensors (the graph is functional, and autograd saved the old ones). mean and
// invstd are returned as two halves of one buffer, so
// the backward kernels can bind them as one (mean, invstd) table.
static std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor> bn_relu_train_dev(const Tensor& x_, const Tensor& w, const Tensor& b,
                                                                          const Tensor& rm, const Tensor& rv, double mom, double eps) {
  NEED(f32(x_) && x_.dim() == 4 && whole(w) && whole(b) && whole(rm) && whole(rv), "4-D float32 with contiguous parameters");
  Tensor x = base0(x_);
  int64_t n = x.size(0), c = x.size(1), h = x.size(2), wd = x.size(3), count = n * h * wd, S = n >= 8 ? 8 : 1;
  Tensor nrm = empty_dev({c}, at::kFloat), nrv = empty_dev({c}, at::kFloat);
  Tensor stats = empty_dev({2 * c}, at::kFloat), part = empty_dev({c * S * 2}, at::kFloat), y = empty_dev(x.sizes(), at::kFloat);
  Buf* qs = qbuf(n, c, h, wd, S, (float)eps);
  launch("bn_stats_part", {bind(x), bind(part), bind_buf(qs)}, u(c), u(S));
  launch("bn_stats_fin_functional", {bind(x), bind(part), bind(stats), bind(rm), bind(rv), bind(nrm), bind(nrv), bind_buf(qs),
                                     bind_buf(uniform({fbits((float)mom), fbits((float)count / std::max<int64_t>(count - 1, 1)), 0, 0}))},
         u(cdiv(c, 64)));
  ew_launch("bn_relu", {bind(x), bind(stats), bind(w), bind(b), bind(y), bind_buf(qbuf(n, c, h, wd, 0, (float)eps))}, x.numel());
  Tensor var = stats.narrow(0, c, c);
  ew_into(ew::RSQRT_EPS, &var, nullptr, var, (float)eps);  // stats = (mean, invstd) from here on
  return {y, stats.narrow(0, 0, c), var, nrm, nrv};
}

static std::tuple<Tensor, Tensor, Tensor> bn_relu_backward_dev(const Tensor& g_, const Tensor& x_, const Tensor& y_, const Tensor& w,
                                                               const Tensor& mean, const Tensor& invstd, std::array<bool, 3> mask) {
  NEED(f32(g_) && f32(x_) && f32(y_) && x_.dim() == 4 && whole(w) && f32(mean) && f32(invstd), "4-D float32");
  Tensor g = base0(g_), x = base0(x_), y = base0(y_);
  int64_t n = x.size(0), c = x.size(1), h = x.size(2), wd = x.size(3), S = n >= 8 ? 8 : 1;
  Tensor stats;
  uint32_t mode;  // 0xffffffff: stats holds (mean, invstd)
  if (mean.storage_offset() == 0 && bufof(mean) == bufof(invstd) && invstd.storage_offset() == c && mean.is_contiguous() &&
      invstd.is_contiguous()) {
    stats = at::detail::make_tensor<c10::TensorImpl>(c10::Storage(mean.storage()), mean.key_set(), mean.dtype());
    stats.unsafeGetTensorImpl()->set_sizes_contiguous({2 * c});
    mode = 0xffffffffu;  // as bn_relu_train left it: no rebuilding
  } else {
    stats = empty_dev({2 * c}, at::kFloat);
    copy_dev(mean, stats.narrow(0, 0, c));
    copy_dev(invstd, stats.narrow(0, c, c));
    mode = 0xffffffffu;
  }
  Tensor part = empty_dev({c * S * 2}, at::kFloat), dgb = empty_dev({2, c}, at::kFloat);
  Buf* qs = uniform({u(n), u(c), u(h), u(wd), 0, 0, u(S), mode});
  launch("bn_bwd_part", {bind(g), bind(x), bind(stats), bind(y), bind(part), bind_buf(qs)}, u(c), u(S));
  launch("bn_bwd_fin", {bind(part), bind(dgb), bind_buf(qs)}, u(cdiv(c, 64)));
  Tensor gx;
  if (mask[0]) {
    gx = empty_dev(x.sizes(), at::kFloat);
    ew_launch("bn_bwd_apply", {bind(g), bind(x), bind(stats), bind(y), bind(w), bind(dgb), bind(gx),
                               bind_buf(uniform({u(n), u(c), u(h), u(wd), 0, 0, 0, mode}))}, x.numel());
  }
  return {gx, mask[1] ? dgb.select(0, 0) : Tensor(), mask[2] ? dgb.select(0, 1) : Tensor()};
}

static Tensor conv_act_dev(const Tensor& x_, const Tensor& w_, const std::optional<Tensor>& b, IntArrayRef stride,
                           IntArrayRef padding, int64_t act) {
  NEED(f32(x_) && f32(w_) && x_.dim() == 4 && w_.dim() == 4 && w_.size(2) == w_.size(3) && stride[0] == stride[1] &&
       padding[0] == padding[1] && stride[0] >= 1 && stride[0] <= 4 && w_.size(2) <= 11 && padding[0] <= w_.size(2) &&
       x_.size(1) == w_.size(1) && (!b || !b->defined() || f32(*b)), "dense square conv");
  Tensor x = base0(x_), w = base0(w_);
  Geo g = conv_geo(x.size(0), x.size(1), w.size(0), x.size(2), x.size(3), w.size(2), stride[0], padding[0]);
  Tensor out = empty_dev({g.n, g.co, g.oh, g.ow}, at::kFloat);
  Tensor bb = b && b->defined() ? base0(*b) : Tensor();
  gemm_forward(x, w, bb.defined() ? &bb : nullptr, out, g, act == 1);
  return out;
}

static Tensor conv_bn_act_eval_dev(const Tensor& x, const Tensor& cw, const std::optional<Tensor>& cb, const Tensor& w,
                                   const Tensor& b, const Tensor& rm, const Tensor& rv, double eps, IntArrayRef stride,
                                   IntArrayRef padding, int64_t act) {
  NEED(f32(cw) && f32(w) && f32(b) && f32(rm) && f32(rv), "float32");
  auto f = fold_bn(cw, cb, w, b, rm, rv, eps);  // a few tiny elementwise ops on the weights, then one GEMM
  return conv_act_dev(x, f.first, f.second, stride, padding, act);
}

// generated pointwise kernel over up to 6 broadcast inputs (see compile.py: the uniform layout must match)
static Tensor fused_pointwise_dev(c10::string_view kernel, at::TensorList ins) {
  NEED(!ins.empty() && ins.size() <= 6, "1 to 6 inputs");
  std::vector<int64_t> shape = ins[0].sizes().vec();
  for (auto& t : ins) {
    NEED(f32(t), "float32 device inputs");
    shape = at::infer_size(shape, t.sizes());
  }
  NEED(shape.size() <= 8, "rank <= 8");
  Tensor out = empty_dev(shape, at::kFloat);
  int64_t n = out.numel();
  if (!n) return out;
  int R = (int)shape.size();
  std::vector<uint32_t> w = {u(n), u(R), 0, 0};
  for (int d = 0; d < 8; d++) w.push_back(d < R ? u(shape[d]) : 1u);
  for (size_t k = 0; k < 6; k++)
    for (int d = 0; d < 8; d++) {
      uint32_t st = 0;
      if (k < ins.size() && d < R) {
        int di = d - (R - (int)ins[k].dim());
        if (di >= 0 && ins[k].size(di) != 1) st = u(ins[k].stride(di));
      }
      w.push_back(st);
    }
  for (int k = 0; k < 8; k++) w.push_back(k < (int)ins.size() ? u(ins[k].storage_offset()) : 0u);
  std::vector<Binding> b;  // inputs at bindings 0..k-1, then the output, then the uniform (as compile.py emits them)
  for (auto& t : ins) b.push_back(bind(t));
  b.push_back(bind(out));
  b.push_back(bind_buf(uniform(w)));
  std::string name(kernel);
  ew_launch(name.c_str(), b, n);
  return out;
}

// eager entry points for the non-functional BatchNorm variants that traced / compiled graphs call
static std::tuple<Tensor, Tensor, Tensor> bn_legit(const Tensor& x, const std::optional<Tensor>& w, const std::optional<Tensor>& b,
                                                   Tensor& rm, Tensor& rv, bool training, double mom, double eps) {
  return at::native_batch_norm(x, w, b, rm, rv, training, mom, eps);
}
static std::tuple<Tensor, Tensor, Tensor> bn_legit_no_training(const Tensor& x, const std::optional<Tensor>& w,
                                                               const std::optional<Tensor>& b, const Tensor& rm,
                                                               const Tensor& rv, double mom, double eps) {
  return at::native_batch_norm(x, w, b, rm, rv, false, mom, eps);
}

// fast path, else the reference composite (which dispatches aten ops, each with its own fallback)
#define TRY_DEV(dev, ref, ...)          \
  OMT_LOCK;                              \
  try {                                  \
    return dev(__VA_ARGS__);             \
  } catch (const Unsupported& e) {       \
    note_fallback(#dev, e.why);          \
    return ref(__VA_ARGS__);             \
  }

static std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor> bn_relu_train_k(const Tensor& x, const Tensor& w, const Tensor& b, const Tensor& rm,
                                                                        const Tensor& rv, double mom, double eps) {
  TRY_DEV(bn_relu_train_dev, bn_relu_train_ref, x, w, b, rm, rv, mom, eps)
}
static std::tuple<Tensor, Tensor, Tensor> bn_relu_backward_k(const Tensor& g, const Tensor& x, const Tensor& y, const Tensor& w,
                                                             const Tensor& mean, const Tensor& invstd, std::array<bool, 3> mask) {
  TRY_DEV(bn_relu_backward_dev, bn_relu_backward_ref, g, x, y, w, mean, invstd, mask)
}
static Tensor conv_act_k(const Tensor& x, const Tensor& w, const std::optional<Tensor>& b, IntArrayRef s, IntArrayRef p, int64_t act) {
  TRY_DEV(conv_act_dev, conv_act_ref, x, w, b, s, p, act)
}
static Tensor conv_bn_act_eval_k(const Tensor& x, const Tensor& cw, const std::optional<Tensor>& cb, const Tensor& w, const Tensor& b,
                                 const Tensor& rm, const Tensor& rv, double eps, IntArrayRef s, IntArrayRef p, int64_t act) {
  TRY_DEV(conv_bn_act_eval_dev, conv_bn_act_eval_ref, x, cw, cb, w, b, rm, rv, eps, s, p, act)
}
static Tensor fused_pointwise_k(c10::string_view kernel, at::TensorList ins) {
  OMT_LOCK;
  return fused_pointwise_dev(kernel, ins);  // compile.py only emits it for float32 device tensors of rank <= 8
}

TORCH_LIBRARY(omt, m) {
  m.def("bn_relu_train(Tensor x, Tensor w, Tensor b, Tensor rm, Tensor rv, float momentum, float eps) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.def("bn_relu_backward(Tensor g, Tensor x, Tensor y, Tensor w, Tensor mean, Tensor invstd, bool[3] mask) -> (Tensor, Tensor, Tensor)");
  m.def("conv_act(Tensor x, Tensor w, Tensor? b, int[] stride, int[] padding, int act) -> Tensor");
  m.def("conv_bn_act_eval(Tensor x, Tensor cw, Tensor? cb, Tensor w, Tensor b, Tensor rm, Tensor rv, float eps, int[] stride, int[] padding, int act) -> Tensor");
  m.def("fused_pointwise(str kernel, Tensor[] inputs) -> Tensor");
}
TORCH_LIBRARY_IMPL(omt, CompositeExplicitAutograd, m) {
  m.impl("bn_relu_train", &bn_relu_train_ref);
  m.impl("bn_relu_backward", &bn_relu_backward_ref);
  m.impl("conv_act", &conv_act_ref);
  m.impl("conv_bn_act_eval", &conv_bn_act_eval_ref);
  m.impl("fused_pointwise", &fused_pointwise_ref);
}
TORCH_LIBRARY_IMPL(omt, PrivateUse1, m) {
  m.impl("bn_relu_train", &bn_relu_train_k);
  m.impl("bn_relu_backward", &bn_relu_backward_k);
  m.impl("conv_act", &conv_act_k);
  m.impl("conv_bn_act_eval", &conv_bn_act_eval_k);
  m.impl("fused_pointwise", &fused_pointwise_k);
}
TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  OMT_IMPL(m, _native_batch_norm_legit, bn_legit);
  OMT_IMPL(m, _native_batch_norm_legit_no_training, bn_legit_no_training);
}

}  // namespace omt
