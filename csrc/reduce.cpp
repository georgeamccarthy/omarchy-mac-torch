// Reductions (sum, mean, argmax/argmin), softmax / log_softmax over any dim, nll_loss.
#include "omt.h"

namespace omt {

// view x (contiguous) as (outer, red, inner) around the reduced dims; non-adjacent reduced dims are moved last first
struct Split {
  Tensor x;
  int64_t outer = 1, red = 1, inner = 1;
  std::vector<int64_t> out_shape;
};
static Split split_dims(const Tensor& x_, IntArrayRef dims_in, bool keepdim) {
  int64_t nd = x_.dim();
  std::vector<int64_t> dims;
  if (dims_in.empty()) for (int64_t i = 0; i < nd; i++) dims.push_back(i);
  else for (auto d : dims_in) dims.push_back(c10::maybe_wrap_dim(d, std::max<int64_t>(nd, 1)));
  std::sort(dims.begin(), dims.end());
  dims.erase(std::unique(dims.begin(), dims.end()), dims.end());
  Split s;
  for (int64_t i = 0; i < nd; i++) {
    bool r = std::find(dims.begin(), dims.end(), i) != dims.end();
    if (r) { if (keepdim) s.out_shape.push_back(1); }
    else s.out_shape.push_back(x_.size(i));
  }
  bool adjacent = true;
  for (size_t i = 1; i < dims.size(); i++) adjacent = adjacent && dims[i] == dims[i - 1] + 1;
  if (nd == 0) { s.x = x_; return s; }
  if (adjacent) {
    s.x = contig(x_);
    for (int64_t i = 0; i < dims.front(); i++) s.outer *= x_.size(i);
    for (int64_t i = dims.front(); i <= dims.back(); i++) s.red *= x_.size(i);
    for (int64_t i = dims.back() + 1; i < nd; i++) s.inner *= x_.size(i);
  } else {
    std::vector<int64_t> perm;
    for (int64_t i = 0; i < nd; i++) if (std::find(dims.begin(), dims.end(), i) == dims.end()) { perm.push_back(i); s.outer *= x_.size(i); }
    for (auto d : dims) { perm.push_back(d); s.red *= x_.size(d); }
    s.x = contig(x_.permute(perm));
  }
  return s;
}

static void launch_reduce(const Tensor& x, int64_t outer, int64_t red, int64_t inner, const Tensor& out, float scale) {
  int64_t outs = outer * inner;
  if (outs < 128 && red >= 16384) {  // too few outputs to fill the GPU: split each reduction over K workgroups
    int64_t K = std::min<int64_t>(256, cdiv(red, 4096)), chunk = cdiv(red, K);
    Tensor part = empty_dev({outs * K}, at::kFloat);
    uint32_t gx, gy;
    grid(outs * K, 1, gx, gy);
    launch("reduce_part", {bind(x), bind(part), bind_buf(uniform({u(outer), u(red), u(inner), u(x.storage_offset()), u(chunk), u(K), 0, 0}))}, gx, gy);
    launch_reduce(part, outs, K, 1, out, scale);
    return;
  }
  Buf* ub = uniform({u(outer), u(red), u(inner), u(x.storage_offset()), u(out.storage_offset()), fbits(scale), 0, 0});
  uint32_t gx, gy;
  if (red >= 64 || outs < 4096) {  // one workgroup per output, tree reduction
    grid(outs, 1, gx, gy);
    launch("reduce_wg", {bind(x), bind(out), bind_buf(ub)}, gx, gy);
  } else {  // many short reductions: one thread each
    grid(outs, 64, gx, gy);
    launch("reduce", {bind(x), bind(out), bind_buf(ub)}, gx, gy);
  }
}

static Tensor reduce_dims(const Tensor& x, IntArrayRef dims, bool keepdim, bool mean, std::optional<at::ScalarType> dt) {
  NEED(f32(x) && (!dt || *dt == at::kFloat), "float32 reductions");
  Split s = split_dims(x, dims, keepdim);
  Tensor out = empty_dev(s.out_shape, at::kFloat);
  if (out.numel() == 0) return out;
  if (s.red == 0) { fill_dev(out, mean ? NAN : 0.f); return out; }
  launch_reduce(s.x, s.outer, s.red, s.inner, out, mean ? 1.f / (float)s.red : 1.f);
  return out;
}

static Tensor sum_default(const Tensor& x, std::optional<at::ScalarType> dt) {
  auto st = x.scalar_type();
  if (st == at::kBool || st == at::kByte || st == at::kLong || st == at::kInt) {  // counts: int64 result
    NEED(!dt || *dt == at::kLong, "sum dtype");
    Tensor xc = contig(x);
    Tensor out = empty_dev({}, at::kLong);
    int kind = st == at::kInt ? 1 : (st == at::kLong ? 2 : 3);
    launch("isum", {bind(xc), bind(out), bind_buf(uniform({u(xc.numel()), u(xc.storage_offset()), u(kind), 0}))}, 1);
    return out;
  }
  return reduce_dims(x, {}, false, false, dt);
}
static Tensor sum_dim(const Tensor& x, at::OptionalIntArrayRef dim, bool keepdim, std::optional<at::ScalarType> dt) {
  return reduce_dims(x, dim.value_or(IntArrayRef{}), keepdim, false, dt);
}
static Tensor mean_dim(const Tensor& x, at::OptionalIntArrayRef dim, bool keepdim, std::optional<at::ScalarType> dt) {
  return reduce_dims(x, dim.value_or(IntArrayRef{}), keepdim, true, dt);
}
static Tensor mean_default(const Tensor& x, std::optional<at::ScalarType> dt) { return reduce_dims(x, {}, false, true, dt); }

static Tensor arg_impl(const Tensor& x_, std::optional<int64_t> dim, bool keepdim, uint32_t mode) {
  NEED(f32(x_) && x_.numel() > 0, "float32 argmax");
  Split s;
  if (!dim) {
    s.x = contig(x_);
    s.red = x_.numel();
    if (keepdim) s.out_shape.assign(x_.dim(), 1);
  } else {
    int64_t d = c10::maybe_wrap_dim(*dim, std::max<int64_t>(x_.dim(), 1));
    s = split_dims(x_, x_.dim() ? IntArrayRef{d} : IntArrayRef{}, keepdim);
  }
  Tensor out = empty_dev(s.out_shape, at::kLong);
  uint32_t gx, gy;
  grid(s.outer * s.inner, 1, gx, gy);
  launch("argmax_row", {bind(s.x), bind(out), bind_buf(uniform({u(s.outer), u(s.red), u(s.inner), u(s.x.storage_offset()), mode, 0, 0, 0}))}, gx, gy);
  return out;
}
static Tensor argmax_(const Tensor& x, std::optional<int64_t> dim, bool keepdim) { return arg_impl(x, dim, keepdim, 0); }
static Tensor argmin_(const Tensor& x, std::optional<int64_t> dim, bool keepdim) { return arg_impl(x, dim, keepdim, 1); }

// ---- softmax ----

static void dims3(const Tensor& x, int64_t dim, int64_t& outer, int64_t& d, int64_t& inner) {
  int64_t nd = std::max<int64_t>(x.dim(), 1);
  dim = c10::maybe_wrap_dim(dim, nd);
  outer = 1; inner = 1; d = x.dim() ? x.size(dim) : 1;
  for (int64_t i = 0; i < dim; i++) outer *= x.size(i);
  for (int64_t i = dim + 1; i < x.dim(); i++) inner *= x.size(i);
}
static Tensor softmax_impl(const Tensor& x_, int64_t dim, bool half, uint32_t mode) {
  NEED(f32(x_) && !half, "float32 softmax");
  Tensor x = contig(x_);
  int64_t outer, d, inner;
  dims3(x, dim, outer, d, inner);
  Tensor out = empty_dev(x.sizes(), at::kFloat);
  if (!x.numel()) return out;
  uint32_t gx, gy;
  grid(outer * inner, 1, gx, gy);
  launch("softmax", {bind(x), bind(out), bind_buf(uniform({u(outer), u(d), u(inner), mode, u(x.storage_offset()), 0, 0, 0}))}, gx, gy);
  return out;
}
static Tensor softmax_(const Tensor& x, int64_t dim, bool half) { return softmax_impl(x, dim, half, 0); }
static Tensor log_softmax_(const Tensor& x, int64_t dim, bool half) { return softmax_impl(x, dim, half, 1); }
static Tensor smax_bwd_impl(const Tensor& g_, const Tensor& y_, int64_t dim, at::ScalarType dt, uint32_t mode) {
  NEED(f32(g_) && f32(y_) && dt == at::kFloat, "float32 softmax backward");
  Tensor g = base0(g_), y = base0(y_);
  int64_t outer, d, inner;
  dims3(y, dim, outer, d, inner);
  Tensor out = empty_dev(y.sizes(), at::kFloat);
  if (!y.numel()) return out;
  uint32_t gx, gy;
  grid(outer * inner, 1, gx, gy);
  launch("softmax_bwd", {bind(g), bind(y), bind(out), bind_buf(uniform({u(outer), u(d), u(inner), mode, 0, 0, 0, 0}))}, gx, gy);
  return out;
}
static Tensor softmax_bwd_(const Tensor& g, const Tensor& y, int64_t dim, at::ScalarType dt) { return smax_bwd_impl(g, y, dim, dt, 0); }
static Tensor log_softmax_bwd_(const Tensor& g, const Tensor& y, int64_t dim, at::ScalarType dt) { return smax_bwd_impl(g, y, dim, dt, 1); }

// ---- nll_loss: F.cross_entropy decomposes into log_softmax + nll_loss, both on the GPU ----

static std::tuple<Tensor, Tensor> nll_fwd(const Tensor& x_, const Tensor& t_, const std::optional<Tensor>& w_,
                                          int64_t reduction, int64_t ignore) {
  NEED(f32(x_) && x_.dim() == 2 && is_dev(t_) && t_.scalar_type() == at::kLong, "nll_loss on (N, C) float32");
  bool hw = w_ && w_->defined();
  NEED(!hw || f32(*w_), "float32 weight");
  Tensor x = contig(x_), t = contig(t_);
  Tensor w = hw ? contig(*w_) : x;
  int64_t n = x.size(0), c = x.size(1);
  Tensor loss = empty_dev({n}, at::kFloat), wr = empty_dev({n}, at::kFloat);
  uint32_t gx, gy;
  grid(n, 256, gx, gy);
  launch("nll_fwd", {bind(x), bind(t), bind(w), bind(loss), bind(wr),
                     bind_buf(uniform({u(n), u(c), (uint32_t)(int32_t)ignore, hw ? 1u : 0u, u(x.storage_offset()),
                                       u(t.storage_offset()), hw ? u(w.storage_offset()) : 0u, 0}))}, gx, gy);
  if (reduction == 0) {  // none
    Tensor tw = empty_dev({}, at::kFloat);
    fill_dev(tw, 0.f);
    return {loss, tw};
  }
  Tensor ls = reduce_dims(loss, {}, false, false, std::nullopt), tw = reduce_dims(wr, {}, false, false, std::nullopt);
  if (reduction == 2) return {ls, tw};
  return {ew_new(ew::DIV, &ls, &tw), tw};
}

static Tensor nll_bwd(const Tensor& g_, const Tensor& x, const Tensor& t_, const std::optional<Tensor>& w_,
                      int64_t reduction, int64_t ignore, const Tensor& total_weight) {
  NEED(f32(x) && x.dim() == 2 && f32(g_) && is_dev(t_) && t_.scalar_type() == at::kLong, "nll_loss backward on (N, C)");
  bool hw = w_ && w_->defined();
  NEED(!hw || f32(*w_), "float32 weight");
  Tensor g = contig(g_), t = contig(t_);
  Tensor w = hw ? contig(*w_) : x;
  int64_t n = x.size(0), c = x.size(1);
  Tensor scale;
  if (reduction == 1) scale = ew_new(ew::RECIP, &total_weight, nullptr);
  else { scale = empty_dev({}, at::kFloat); fill_dev(scale, 1.f); }
  Tensor gx = empty_dev(x.sizes(), at::kFloat);
  uint32_t gxx, gy;
  grid(n * c, 256, gxx, gy);
  launch("nll_bwd", {bind(g), bind(t), bind(w), bind(scale), bind(gx),
                     bind_buf(uniform({u(n), u(c), (uint32_t)(int32_t)ignore, hw ? 1u : 0u, u(t.storage_offset()),
                                       hw ? u(w.storage_offset()) : 0u, reduction == 0 ? 1u : 0u, u(g.storage_offset())}))},
         gxx, gy);
  return gx;
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  OMT_IMPL(m, sum, sum_default);
  OMT_IMPL(m, sum_dim_IntList, sum_dim);
  OMT_IMPL(m, mean_dim, mean_dim);
  OMT_IMPL(m, mean, mean_default);
  OMT_IMPL(m, argmax, argmax_);
  OMT_IMPL(m, argmin, argmin_);
  OMT_IMPL(m, _softmax, softmax_);
  OMT_IMPL(m, _log_softmax, log_softmax_);
  OMT_IMPL(m, _softmax_backward_data, softmax_bwd_);
  OMT_IMPL(m, _log_softmax_backward_data, log_softmax_bwd_);
  OMT_IMPL(m, nll_loss_forward, nll_fwd);
  OMT_IMPL(m, nll_loss_backward, nll_bwd);
}

}  // namespace omt
