// Batch norm, layer norm, pooling, dropout, random fills.
#include "omt.h"

#include <ATen/CPUGeneratorImpl.h>
#include <ATen/native/Pool.h>

namespace omt {

at::CPUGeneratorImpl* generator(const std::optional<at::Generator>& gen);  // device.cpp

static Buf* qbuf(int64_t n, int64_t c, int64_t h, int64_t w, int64_t oh = 0, int64_t ow = 0, int64_t p0 = 0, int64_t p1 = 0) {
  return uniform({u(n), u(c), u(h), u(w), u(oh), u(ow), u(p0), u(p1)});
}
static void ew_launch(const char* k, const std::vector<Binding>& b, int64_t total) {
  uint32_t gx, gy;
  grid(total, 256, gx, gy);
  launch(k, b, gx, gy);
}
static bool defined(const std::optional<Tensor>& t) { return t && t->defined(); }

// ------------------------------------------------------------------------------------------------
// batch norm (NCHW, affine, with running statistics)
// ------------------------------------------------------------------------------------------------

static std::tuple<Tensor, Tensor, Tensor> batch_norm_(const Tensor& x_, const std::optional<Tensor>& weight,
                                                      const std::optional<Tensor>& bias, const std::optional<Tensor>& rm,
                                                      const std::optional<Tensor>& rv, bool training, double momentum,
                                                      double eps) {
  NEED(defined(weight) && defined(bias) && defined(rm) && defined(rv) && f32(x_) && x_.dim() == 4,
       "batch norm without affine parameters or running stats, or not 4-D");
  NEED(weight->is_contiguous() && bias->is_contiguous() && rm->is_contiguous() && rv->is_contiguous() &&
       weight->storage_offset() == 0 && bias->storage_offset() == 0 && rm->storage_offset() == 0 && rv->storage_offset() == 0,
       "batch norm parameters must be whole contiguous tensors");
  Tensor x = base0(x_);
  int64_t n = x.size(0), c = x.size(1), h = x.size(2), w = x.size(3);
  Buf* q = qbuf(n, c, h, w);
  Tensor y = empty_dev(x.sizes(), at::kFloat);
  Tensor sm, si;
  if (training) {  // shifted, split statistics (the fused stacks' kernels): stable for large means
    Tensor stats = empty_dev({2 * c}, at::kFloat);
    int64_t count = n * h * w, S = n >= 8 ? 8 : 1;
    Tensor part = empty_dev({c * S * 2}, at::kFloat);
    Buf* qs = qbuf(n, c, h, w, 0, 0, S);
    Buf* r = uniform({fbits((float)momentum), fbits((float)count / std::max<int64_t>(count - 1, 1)), 0, 0});
    launch("bn_stats_part", {bind(x), bind(part), bind_buf(qs)}, u(c), u(S));
    launch("bn_stats_fin", {bind(x), bind(part), bind(stats), bind(*rm), bind(*rv), bind_buf(qs), bind_buf(r)}, u(cdiv(c, 64)));
    sm = stats.narrow(0, 0, c);
    Tensor var = stats.narrow(0, c, c);
    si = ew_new(ew::RSQRT_EPS, &var, nullptr, (float)eps);
  } else {
    sm = *rm;
    si = empty_dev({c}, at::kFloat);
    launch("bn_eval_inv", {bind(*rv), bind(si), bind_buf(uniform({u(c), fbits((float)eps), 0, 0}))}, u(cdiv(c, 64)));
  }
  ew_launch("bn_apply", {bind(x), bind(sm), bind(si), bind(*weight), bind(*bias), bind(y), bind_buf(q)}, n * c * h * w);
  if (!training) return {y, empty_dev({0}, at::kFloat), empty_dev({0}, at::kFloat)};
  return {y, sm, si};
}

static std::tuple<Tensor, Tensor, Tensor> batch_norm_backward_(const Tensor& gy_, const Tensor& x_, const std::optional<Tensor>& weight,
                                                               const std::optional<Tensor>&, const std::optional<Tensor>&,
                                                               const std::optional<Tensor>& sm, const std::optional<Tensor>& si,
                                                               bool train, double, std::array<bool, 3> mask) {
  NEED(train && defined(weight) && defined(sm) && defined(si) && f32(x_) && f32(gy_) && x_.dim() == 4,
       "eval-mode or non-affine batch norm backward");
  NEED(weight->is_contiguous() && weight->storage_offset() == 0, "contiguous weight");
  Tensor gy = base0(gy_), x = base0(x_);
  int64_t n = x.size(0), c = x.size(1), h = x.size(2), w = x.size(3);
  Buf* q = qbuf(n, c, h, w);
  Tensor dgb = empty_dev({2, c}, at::kFloat);
  launch("bnb_reduce", {bind(gy), bind(x), bind(*sm), bind(*si), bind(dgb), bind_buf(q)}, u(c));
  Tensor gx;
  if (mask[0]) {
    gx = empty_dev(x.sizes(), at::kFloat);
    ew_launch("bnb_apply", {bind(gy), bind(x), bind(*sm), bind(*si), bind(*weight), bind(dgb), bind(gx), bind_buf(q)}, n * c * h * w);
  }
  return {gx, mask[1] ? dgb.select(0, 0) : Tensor(), mask[2] ? dgb.select(0, 1) : Tensor()};
}

// ------------------------------------------------------------------------------------------------
// layer norm
// ------------------------------------------------------------------------------------------------

static std::tuple<Tensor, Tensor, Tensor> layer_norm_(const Tensor& x_, IntArrayRef normalized, const std::optional<Tensor>& w,
                                                      const std::optional<Tensor>& b, double eps) {
  NEED(f32(x_) && defined(w) == defined(b), "float32 layer norm with both or neither affine parameters");
  Tensor x = base0(x_);
  int64_t D = 1;
  for (auto d : normalized) D *= d;
  int64_t R = D ? x.numel() / D : 0;
  NEED(R > 0 && R <= 65535, "layer norm row count");
  std::vector<int64_t> stat_shape(x.sizes().begin(), x.sizes().end() - normalized.size());
  for (size_t i = 0; i < normalized.size(); i++) stat_shape.push_back(1);
  Tensor out = empty_dev(x.sizes(), at::kFloat), mean = empty_dev(stat_shape, at::kFloat), rstd = empty_dev(stat_shape, at::kFloat);
  bool aff = defined(w);
  Tensor g = aff ? base0(*w) : x, bt = aff ? base0(*b) : x;
  launch("ln_fwd", {bind(x), bind(g), bind(bt), bind(out), bind(mean), bind(rstd),
                    bind_buf(uniform({u(R), u(D), fbits((float)eps), aff ? 1u : 0u}))}, u(R));
  return {out, mean, rstd};
}
static std::tuple<Tensor, Tensor, Tensor> layer_norm_backward_(const Tensor& gy_, const Tensor& x_, IntArrayRef normalized,
                                                               const Tensor& mean_, const Tensor& rstd_,
                                                               const std::optional<Tensor>& w, const std::optional<Tensor>& b,
                                                               std::array<bool, 3> mask) {
  NEED(f32(x_) && f32(gy_) && defined(w) == defined(b), "float32 layer norm backward with both or neither affine parameters");
  Tensor gy = base0(gy_), x = base0(x_), mean = base0(mean_), rstd = base0(rstd_);
  int64_t D = 1;
  for (auto d : normalized) D *= d;
  int64_t R = D ? x.numel() / D : 0;
  NEED(R > 0 && R <= 65535, "layer norm row count");
  bool aff = defined(w);
  Tensor g = aff ? base0(*w) : x;
  Tensor dx, dg, db;
  Buf* ub = uniform({u(R), u(D), 0, aff ? 1u : 0u});
  if (mask[0]) {
    dx = empty_dev(x.sizes(), at::kFloat);
    launch("ln_bwd_x", {bind(gy), bind(x), bind(g), bind(mean), bind(rstd), bind(dx), bind_buf(ub)}, u(R));
  }
  if (aff && (mask[1] || mask[2])) {
    dg = empty_dev(normalized, at::kFloat);
    db = empty_dev(normalized, at::kFloat);
    launch("ln_bwd_w", {bind(gy), bind(x), bind(mean), bind(rstd), bind(dg), bind(db), bind_buf(ub)}, u(cdiv(D, 64)));
  }
  return {dx, mask[1] ? dg : Tensor(), mask[2] ? db : Tensor()};
}

// ------------------------------------------------------------------------------------------------
// pooling
// ------------------------------------------------------------------------------------------------

struct Pool2d {
  int64_t kh, kw, sh, sw, ph, pw, dh, dw;
};
static Pool2d pool_args(IntArrayRef k, IntArrayRef s, IntArrayRef p, IntArrayRef d) {
  auto at2 = [](IntArrayRef a, size_t i, int64_t dflt) { return a.empty() ? dflt : a[a.size() == 1 ? 0 : i]; };
  Pool2d r;
  r.kh = at2(k, 0, 1); r.kw = at2(k, 1, 1);
  r.sh = at2(s, 0, r.kh); r.sw = at2(s, 1, r.kw);
  r.ph = at2(p, 0, 0); r.pw = at2(p, 1, 0);
  r.dh = at2(d, 0, 1); r.dw = at2(d, 1, 1);
  return r;
}
static Buf* pool_uniform(int64_t nc, int64_t ih, int64_t iw, int64_t oh, int64_t ow, const Pool2d& a, uint32_t mode, int64_t div) {
  return uniform({u(nc), u(ih), u(iw), u(oh), u(ow), u(a.kh), u(a.kw), u(a.sh), u(a.sw), u(a.ph), u(a.pw), u(a.dh),
                  u(a.dw), mode, u(div), 0});
}
static bool pool_ok(const Tensor& x, const Pool2d& a) {
  return f32(x) && x.dim() == 4 && a.kh > 0 && a.kw > 0 && a.sh > 0 && a.sw > 0 && a.ph <= a.kh / 2 && a.pw <= a.kw / 2;
}

static std::tuple<Tensor, Tensor> max_pool2d_(const Tensor& x_, IntArrayRef k, IntArrayRef s, IntArrayRef p, IntArrayRef d, bool ceil) {
  Pool2d a = pool_args(k, s, p, d);
  NEED(pool_ok(x_, a), "max pool: 4-D float32 with padding <= kernel / 2");
  Tensor x = base0(x_);
  int64_t n = x.size(0), c = x.size(1), h = x.size(2), w = x.size(3);
  int64_t oh = at::native::pooling_output_shape<int64_t>(h, a.kh, a.ph, a.sh, a.dh, ceil);
  int64_t ow = at::native::pooling_output_shape<int64_t>(w, a.kw, a.pw, a.sw, a.dw, ceil);
  NEED(oh > 0 && ow > 0, "empty pooling output");
  Tensor out = empty_dev({n, c, oh, ow}, at::kFloat), idx = empty_dev({n, c, oh, ow}, at::kLong);
  ew_launch("maxpool_gen_fwd", {bind(x), bind(out), bind(idx), bind_buf(pool_uniform(n * c, h, w, oh, ow, a, 0, 0))}, out.numel());
  return {out, idx};
}
static Tensor max_pool2d_backward_(const Tensor& gy_, const Tensor& x, IntArrayRef k, IntArrayRef s, IntArrayRef p, IntArrayRef d,
                                   bool, const Tensor& idx_) {
  Pool2d a = pool_args(k, s, p, d);
  NEED(pool_ok(x, a) && f32(gy_), "max pool backward");
  Tensor gy = base0(gy_), idx = base0(idx_);
  int64_t n = x.size(0), c = x.size(1), h = x.size(2), w = x.size(3);
  Tensor gx = empty_dev(x.sizes(), at::kFloat);
  ew_launch("maxpool_gen_bwd", {bind(gy), bind(idx), bind(gx), bind_buf(pool_uniform(n * c, h, w, gy.size(2), gy.size(3), a, 0, 0))}, gx.numel());
  return gx;
}

static Tensor avg_pool2d_(const Tensor& x_, IntArrayRef k, IntArrayRef s, IntArrayRef p, bool ceil, bool include_pad,
                          std::optional<int64_t> divisor) {
  Pool2d a = pool_args(k, s, p, {});
  NEED(pool_ok(x_, a) && (!divisor || *divisor > 0), "avg pool: 4-D float32 with padding <= kernel / 2");
  Tensor x = base0(x_);
  int64_t n = x.size(0), c = x.size(1), h = x.size(2), w = x.size(3);
  int64_t oh = at::native::pooling_output_shape<int64_t>(h, a.kh, a.ph, a.sh, 1, ceil);
  int64_t ow = at::native::pooling_output_shape<int64_t>(w, a.kw, a.pw, a.sw, 1, ceil);
  NEED(oh > 0 && ow > 0, "empty pooling output");
  Tensor out = empty_dev({n, c, oh, ow}, at::kFloat);
  ew_launch("avgpool_gen_fwd", {bind(x), bind(out), bind_buf(pool_uniform(n * c, h, w, oh, ow, a, include_pad ? 1 : 0, divisor.value_or(0)))},
            out.numel());
  return out;
}
static Tensor avg_pool2d_backward_(const Tensor& gy_, const Tensor& x, IntArrayRef k, IntArrayRef s, IntArrayRef p, bool,
                                   bool include_pad, std::optional<int64_t> divisor) {
  Pool2d a = pool_args(k, s, p, {});
  NEED(pool_ok(x, a) && f32(gy_) && (!divisor || *divisor > 0), "avg pool backward");
  Tensor gy = base0(gy_);
  int64_t n = x.size(0), c = x.size(1), h = x.size(2), w = x.size(3);
  Tensor gx = empty_dev(x.sizes(), at::kFloat);
  ew_launch("avgpool_gen_bwd", {bind(gy), bind(gx), bind_buf(pool_uniform(n * c, h, w, gy.size(2), gy.size(3), a, include_pad ? 1 : 0,
                                                                          divisor.value_or(0)))}, gx.numel());
  return gx;
}

static Buf* adapool_params(const Tensor& x, int64_t oh, int64_t ow) {
  return uniform({u(x.size(0)), u(x.size(1)), u(x.size(2)), u(x.size(3)), u(oh), u(ow), 0, 0, 0, 0, 0, 0});
}
static Tensor adaptive_avg_pool2d_(const Tensor& x_, IntArrayRef os) {
  NEED(f32(x_) && x_.dim() == 4, "adaptive avg pool: 4-D float32");
  Tensor x = base0(x_);
  Tensor out = empty_dev({x.size(0), x.size(1), os[0], os[1]}, at::kFloat);
  ew_launch("adapool_fwd", {bind(x), bind(out), bind_buf(adapool_params(x, os[0], os[1]))}, out.numel());
  return out;
}
static Tensor adaptive_avg_pool2d_backward_(const Tensor& gy_, const Tensor& x) {
  NEED(f32(x) && f32(gy_) && x.dim() == 4, "adaptive avg pool backward");
  Tensor gy = base0(gy_);
  Tensor gx = empty_dev(x.sizes(), at::kFloat);
  ew_launch("adapool_bwd", {bind(gy), bind(gx), bind_buf(adapool_params(x, gy.size(2), gy.size(3)))}, gx.numel());
  return gx;
}

// ------------------------------------------------------------------------------------------------
// dropout and random fills
// ------------------------------------------------------------------------------------------------

static uint64_t next_seed(const std::optional<at::Generator>& gen) {
  auto* g = generator(gen);
  std::lock_guard<std::mutex> lock(g->mutex_);
  return g->random64();
}

static std::tuple<Tensor, Tensor> dropout_(const Tensor& x_, double p, std::optional<bool> train) {
  NEED(f32(x_), "float32 dropout");
  Tensor x = contig(x_);
  Tensor mask = empty_dev(x.sizes(), at::kBool), out = empty_dev(x.sizes(), at::kFloat);
  int64_t n = x.numel();
  if (!n) return {out, mask};
  if (!train.value_or(true) || p == 0) {
    copy_dev(x, out);
    fill_dev(mask, true);
    return {out, mask};
  }
  uint32_t seed = (uint32_t)next_seed(std::nullopt);
  float scale = p >= 1 ? 0.f : (float)(1.0 / (1.0 - p));
  Buf* ub = uniform({u(n), seed, fbits((float)p), fbits(scale), u(x.storage_offset()), 0, 0, 0});
  uint32_t gx, gy;
  grid(cdiv(n, 4), 256, gx, gy);
  launch("dropout", {bind(x), bind(out), bind(mask), bind_buf(ub)}, gx, gy);
  return {out, mask};
}
static Tensor dropout_backward_(const Tensor& g_, const Tensor& mask_, double scale) {
  NEED(f32(g_) && is_dev(mask_) && mask_.scalar_type() == at::kBool, "dropout backward");
  Tensor g = contig(g_), mask = contig(mask_);
  Tensor out = empty_dev(g.sizes(), at::kFloat);
  int64_t n = g.numel();
  if (!n) return out;
  ew_launch("mulmask", {bind(g), bind(mask), bind(out), bind_buf(uniform({u(n), fbits((float)scale), u(g.storage_offset()), u(mask.storage_offset())}))}, n);
  return out;
}

// random numbers are drawn on the host from a CPU generator seeded by the device generator, then uploaded
static Tensor& uniform_(Tensor& self, double from, double to, std::optional<at::Generator> gen) {
  at::Generator cpu = at::make_generator<at::CPUGeneratorImpl>(next_seed(gen));
  Tensor h = at::empty(self.sizes(), self.options().device(c10::kCPU)).uniform_(from, to, cpu);
  return self.copy_(h);
}
static Tensor& normal_(Tensor& self, double mean, double std, std::optional<at::Generator> gen) {
  at::Generator cpu = at::make_generator<at::CPUGeneratorImpl>(next_seed(gen));
  Tensor h = at::empty(self.sizes(), self.options().device(c10::kCPU)).normal_(mean, std, cpu);
  return self.copy_(h);
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  OMT_IMPL(m, native_batch_norm, batch_norm_);
  OMT_IMPL(m, native_batch_norm_backward, batch_norm_backward_);
  OMT_IMPL(m, native_layer_norm, layer_norm_);
  OMT_IMPL(m, native_layer_norm_backward, layer_norm_backward_);
  OMT_IMPL(m, max_pool2d_with_indices, max_pool2d_);
  OMT_IMPL(m, max_pool2d_with_indices_backward, max_pool2d_backward_);
  OMT_IMPL(m, avg_pool2d, avg_pool2d_);
  OMT_IMPL(m, avg_pool2d_backward, avg_pool2d_backward_);
  OMT_IMPL(m, _adaptive_avg_pool2d, adaptive_avg_pool2d_);
  OMT_IMPL(m, _adaptive_avg_pool2d_backward, adaptive_avg_pool2d_backward_);
  OMT_IMPL(m, native_dropout, dropout_);
  OMT_IMPL(m, native_dropout_backward, dropout_backward_);
  OMT_IMPL(m, uniform_, uniform_);
  OMT_IMPL(m, normal_, normal_);
}

}  // namespace omt
