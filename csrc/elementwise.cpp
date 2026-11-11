// Elementwise ops: one general strided kernel (ew.wgsl) for float math, copies and fills of any dtype, comparisons
// with broadcasting (cmp.wgsl), where / masked_fill (where3.wgsl), integer add/sub (iarith.wgsl).
#include "omt.h"

#include <ATen/MemoryOverlap.h>
#include <ATen/native/Resize.h>

#include <limits>

namespace omt {

// ------------------------------------------------------------------------------------------------
// layouts: broadcast strides per operand, then merge adjacent dims that are contiguous for every operand
// ------------------------------------------------------------------------------------------------

struct Layout {
  int nd = 0;
  int64_t sh[8];
  int64_t st[4][8];  // per operand
};

static std::vector<int64_t> bstrides(const Tensor& t, IntArrayRef shape) {
  int R = (int)shape.size();
  std::vector<int64_t> st(R, 0);
  for (int i = 0; i < t.dim(); i++) {
    int di = R - (int)t.dim() + i;
    st[di] = t.size(i) == 1 ? 0 : t.stride(i);
  }
  return st;
}

// shape: iteration shape; strides[k]: operand k's (broadcast) strides. `wpe` 32-bit words per element: 2 splits
// each 8-byte element into a (lo, hi) pair by appending an inner dim of size 2.
static Layout coalesce(IntArrayRef shape, const std::vector<std::vector<int64_t>>& strides, int wpe = 1) {
  int ops = (int)strides.size();
  std::vector<int64_t> sh;
  std::vector<std::vector<int64_t>> st(ops);
  for (size_t d = 0; d < shape.size(); d++) {
    if (shape[d] == 1) continue;
    sh.push_back(shape[d]);
    for (int k = 0; k < ops; k++) st[k].push_back(strides[k][d] * wpe);
  }
  if (wpe == 2) {
    sh.push_back(2);
    for (int k = 0; k < ops; k++) st[k].push_back(1);
  }
  // merge dim d into d+1 when d steps exactly over d+1 for every operand
  for (int d = (int)sh.size() - 2; d >= 0; d--) {
    bool ok = true;
    for (int k = 0; k < ops && ok; k++) ok = st[k][d] == st[k][d + 1] * sh[d + 1];
    if (!ok) continue;
    sh[d + 1] *= sh[d];
    sh.erase(sh.begin() + d);
    for (int k = 0; k < ops; k++) st[k].erase(st[k].begin() + d);
  }
  if (sh.empty()) {
    sh.push_back(1);
    for (int k = 0; k < ops; k++) st[k].push_back(0);
  }
  NEED(sh.size() <= 8, "more than 8 non-contiguous dims");
  Layout L;
  L.nd = (int)sh.size();
  for (int d = 0; d < 8; d++) {
    L.sh[d] = d < L.nd ? sh[d] : 1;
    for (int k = 0; k < ops; k++) L.st[k][d] = d < L.nd ? st[k][d] : 0;
  }
  return L;
}

static void push8(std::vector<uint32_t>& w, const int64_t* v) {
  for (int d = 0; d < 8; d++) w.push_back(u(v[d]));
}

// ------------------------------------------------------------------------------------------------
// the general kernel
// ------------------------------------------------------------------------------------------------

// a, b: optional inputs (broadcast to out's shape); everything in 32-bit words (wpe words per element)
static void launch_ew(uint32_t op, const Tensor* a, const Tensor* b, const Tensor& out, uint32_t alpha_bits,
                      uint32_t sval_bits, int wpe) {
  int64_t n = out.numel();
  if (n == 0) return;
  IntArrayRef shape = out.sizes();
  std::vector<std::vector<int64_t>> st = {std::vector<int64_t>(out.strides().begin(), out.strides().end()),
                                          a ? bstrides(*a, shape) : std::vector<int64_t>(shape.size(), 0),
                                          b ? bstrides(*b, shape) : std::vector<int64_t>(shape.size(), 0)};
  Layout L = coalesce(shape, st, wpe);
  int64_t total = 1;
  for (int d = 0; d < L.nd; d++) total *= L.sh[d];
  bool fa = a && bufof(*a) == bufof(out), fb = b && bufof(*b) == bufof(out);
  std::vector<uint32_t> w = {u(total), op, b ? 0u : 1u, u(L.nd), alpha_bits, sval_bits,
                             a ? u(off_bytes(*a) / 4) : 0u, b ? u(off_bytes(*b) / 4) : 0u, u(off_bytes(out) / 4),
                             fa ? 1u : 0u, fb ? 1u : 0u, 0u};
  push8(w, L.sh);
  push8(w, L.st[1]);
  push8(w, L.st[2]);
  push8(w, L.st[0]);
  Buf* A = (a && !fa) ? bufof(*a) : dummy_buf();
  Buf* B = (b && !fb) ? bufof(*b) : dummy_buf();
  uint32_t gx, gy;
  grid(total, 256, gx, gy);
  launch("ew", {bind_buf(A), bind_buf(B), bind(out), bind_buf(uniform(w))}, gx, gy);
}

void ew_into(uint32_t op, const Tensor* a, const Tensor* b, const Tensor& out, float alpha, float sval) {
  NEED(f32(out) && (!a || f32(*a)) && (!b || f32(*b)), "float32 only");
  at::assert_no_internal_overlap(out);
  if (a) at::assert_no_partial_overlap(out, *a);
  if (b) at::assert_no_partial_overlap(out, *b);
  launch_ew(op, a, b, out, fbits(alpha), fbits(sval), 1);
}

Tensor ew_new(uint32_t op, const Tensor* a, const Tensor* b, float alpha, float sval) {
  NEED((!a || f32(*a)) && (!b || f32(*b)), "float32 only");
  std::vector<int64_t> shape = a ? a->sizes().vec() : b->sizes().vec();
  if (a && b) shape = at::infer_size(a->sizes(), b->sizes());
  Tensor out = empty_dev(shape, at::kFloat);
  launch_ew(op, a, b, out, fbits(alpha), fbits(sval), 1);
  return out;
}

// ------------------------------------------------------------------------------------------------
// copies, fills and transfers for every dtype
// ------------------------------------------------------------------------------------------------

// The storage span a view covers, as a contiguous CPU tensor of its dtype (strides are never negative in PyTorch).
static int64_t span_elems(const Tensor& t) {
  int64_t m = 0;
  for (int i = 0; i < t.dim(); i++) {
    if (t.size(i) == 0) return 0;
    m += (t.size(i) - 1) * t.stride(i);
  }
  return m + 1;
}

static Tensor download_raw(const Tensor& t);
Tensor download(const Tensor& t) {
  Tensor out = download_raw(t);
  if (t.is_conj()) out = out.conj_physical();  // the stored bytes are the un-conjugated values
  if (t.is_neg()) out = out.neg();
  return out;
}
static Tensor download_raw(const Tensor& t) {
  Tensor out = at::empty(t.sizes(), t.options().device(c10::kCPU));
  if (t.numel() == 0) return out;
  if (t.is_contiguous()) {
    download_bytes(bufof(t), off_bytes(t), out.data_ptr(), out.nbytes());
    return out;
  }
  if (t.element_size() == 4 || t.element_size() == 8) {  // gather on the GPU, then read only what's needed
    Tensor c = empty_dev(t.sizes(), t.scalar_type());
    launch_ew(ew::COPY, &t, nullptr, c, 0, 0, (int)(t.element_size() / 4));
    download_bytes(bufof(c), 0, out.data_ptr(), out.nbytes());
    return out;
  }
  Tensor span = at::empty({span_elems(t)}, out.options());
  download_bytes(bufof(t), off_bytes(t), span.data_ptr(), span.nbytes());
  out.copy_(span.as_strided(t.sizes(), t.strides(), 0));
  return out;
}

// write CPU values into a device view of any layout
static void host_copy_into(const Tensor& cpu, const Tensor& dst) {
  Tensor src = cpu.to(dst.scalar_type()).expand(dst.sizes());
  if (dst.is_conj()) src = src.conj_physical();  // store what the lazily conjugated / negated view will read back
  if (dst.is_neg()) src = src.neg();
  if (dst.is_contiguous()) {
    Tensor c = src.contiguous();
    upload_bytes(bufof(dst), off_bytes(dst), c.data_ptr(), c.nbytes());
    return;
  }
  Tensor span = at::empty({span_elems(dst)}, src.options());
  download_bytes(bufof(dst), off_bytes(dst), span.data_ptr(), span.nbytes());
  span.as_strided(dst.sizes(), dst.strides(), 0).copy_(src);
  upload_bytes(bufof(dst), off_bytes(dst), span.data_ptr(), span.nbytes());
}

void copy_dev(const Tensor& src_, const Tensor& dst) {
  TORCH_INTERNAL_ASSERT(src_.scalar_type() == dst.scalar_type());
  if (dst.numel() == 0) return;
  if (src_.is_conj() != dst.is_conj() || src_.is_neg() != dst.is_neg()) {  // lazy conj / neg bits differ: values change
    host_copy_into(download(src_), dst);
    return;
  }
  Tensor src = src_.sizes() == dst.sizes() ? src_ : src_.expand(dst.sizes());
  at::assert_no_internal_overlap(dst);
  if (src.is_alias_of(dst) && src.storage_offset() == dst.storage_offset() && src.strides() == dst.strides()) return;
  if (bufof(src) == bufof(dst) && at::get_overlap_status(src, dst) != at::MemOverlapStatus::No) {
    Tensor tmp = empty_dev(src.sizes(), src.scalar_type());  // overlapping copy: go through a temporary
    copy_dev(src, tmp);
    src = tmp;
  }
  int64_t es = dst.element_size();
  if (es == 4 || es == 8) {
    launch_ew(ew::COPY, &src, nullptr, dst, 0, 0, (int)(es / 4));
    return;
  }
  // 1- and 2-byte dtypes: whole-word buffer copies when both sides are aligned and contiguous, else via the host
  bool plain = src.is_contiguous() && dst.is_contiguous() && src_.numel() == dst.numel() &&
               off_bytes(src) % 4 == 0 && off_bytes(dst) % 4 == 0;
  uint64_t bytes = dst.nbytes();
  if (plain && (bytes % 4 == 0 || owns_storage(dst))) {
    record_copy(bufof(src), off_bytes(src), bufof(dst), off_bytes(dst), (bytes + 3) / 4 * 4);
    return;
  }
  host_copy_into(download(src), dst);
}

void fill_dev(const Tensor& t, const Scalar& v) {
  if (t.numel() == 0) return;
  switch (t.scalar_type()) {
    case at::kFloat: launch_ew(ew::FILL, nullptr, nullptr, t, 0, fbits(v.to<float>()), 1); return;
    case at::kInt: launch_ew(ew::FILL, nullptr, nullptr, t, 0, (uint32_t)v.to<int32_t>(), 1); return;
    case at::kLong: {
      uint64_t x = (uint64_t)v.to<int64_t>();
      launch_ew(ew::FILL2, nullptr, nullptr, t, (uint32_t)(x >> 32), (uint32_t)x, 2);
      return;
    }
    case at::kBool:
    case at::kByte:
    case at::kChar:
      if (owns_storage(t)) {  // fill whole words (the padding after the last element belongs to this tensor)
        uint32_t byte = t.scalar_type() == at::kBool ? (v.to<bool>() ? 1u : 0u) : (uint32_t)(uint8_t)v.to<int64_t>();
        Tensor words = at::detail::make_tensor<c10::TensorImpl>(c10::Storage(t.storage()), t.key_set(),
                                                               caffe2::TypeMeta::Make<float>());
        words.unsafeGetTensorImpl()->set_sizes_contiguous({cdiv(t.numel(), 4)});
        launch_ew(ew::FILL, nullptr, nullptr, words, 0, byte * 0x01010101u, 1);
        return;
      }
      break;
    default: break;
  }
  host_copy_into(at::full(t.sizes(), v, t.options().device(c10::kCPU)), t);
}

static int cast_kind(at::ScalarType t) {
  switch (t) {
    case at::kFloat: return 0;
    case at::kInt: return 1;
    case at::kLong: return 2;
    case at::kBool: return 3;
    case at::kByte: return 4;
    default: return -1;
  }
}
// dtype conversion on the GPU (float32, int32, int64, bool, uint8)
Tensor cast_dev(const Tensor& src_, at::ScalarType dt) {
  if (src_.scalar_type() == dt) return src_;
  int sk = cast_kind(src_.scalar_type()), dk = cast_kind(dt);
  NEED(is_dev(src_) && sk >= 0 && dk >= 0, "dtype conversion between float32/int32/int64/bool/uint8");
  Tensor src = contig(src_);
  Tensor out = empty_dev(src.sizes(), dt);
  int64_t n = src.numel();
  if (!n) return out;
  uint32_t gx, gy;
  grid(cdiv(n, 4), 256, gx, gy);
  launch("cast", {bind(src), bind(out), bind_buf(uniform({u(n), u(sk), u(dk), u(src.storage_offset()), 0, 0, 0, 0}))}, gx, gy);
  return out;
}

// ------------------------------------------------------------------------------------------------
// float arithmetic
// ------------------------------------------------------------------------------------------------

static bool cpu_scalar(const Tensor& t) { return !is_dev(t) && t.dim() == 0; }
static bool is_int(const Tensor& t) { return t.scalar_type() == at::kLong || t.scalar_type() == at::kInt; }
static Tensor int_binary(uint32_t op, const Tensor& a, const Tensor* b, const Scalar* s, const Tensor* out);

static Tensor binary(uint32_t op, const Tensor& a, const Tensor& b, float alpha = 1.f) {
  if (is_dev(a) && is_int(a) && (op == ew::ADD || op == ew::SUB)) {
    if (cpu_scalar(b)) {
      NEED(!b.is_floating_point(), "int tensor with a float scalar (type promotion)");
      Scalar s = b.item<int64_t>() * (int64_t)alpha;
      return int_binary(op, a, nullptr, &s, nullptr);
    }
    NEED(alpha == 1.f, "integer add with alpha");
    return int_binary(op, a, &b, nullptr, nullptr);
  }
  if (cpu_scalar(b)) {
    if (!f32(a) && is_dev(a) && cast_kind(a.scalar_type()) > 0 && (op == ew::DIV || b.is_floating_point())) {
      Tensor af = cast_dev(a, at::kFloat);  // int / bool tensor with a float scalar or true division: float result
      return ew_new(op, &af, nullptr, alpha, b.item<float>());
    }
    return ew_new(op, &a, nullptr, alpha, b.item<float>());
  }
  if (is_dev(a) && is_dev(b) && !(f32(a) && f32(b))) {  // mixed dtypes: promote to float32 when the result is float
    bool af = f32(a), bf = f32(b);
    bool ints = cast_kind(a.scalar_type()) > 0 && cast_kind(b.scalar_type()) > 0;
    NEED((af && cast_kind(b.scalar_type()) > 0) || (bf && cast_kind(a.scalar_type()) > 0) || (ints && op == ew::DIV),
         "mixed dtypes other than float32 with int/bool");
    Tensor a2 = af ? a : cast_dev(a, at::kFloat), b2 = bf ? b : cast_dev(b, at::kFloat);
    return ew_new(op, &a2, &b2, alpha);
  }
  if (cpu_scalar(a)) {
    float s = a.item<float>();
    if (op == ew::MUL || (op == ew::ADD && alpha == 1.f)) return ew_new(op, &b, nullptr, 1.f, s);
    if (op == ew::SUB) return ew_new(ew::RSUB, &b, nullptr, alpha, s);  // s - alpha * b
    NEED(false, "scalar first operand");
  }
  NEED(f32(a) && f32(b), "float32 only");
  return ew_new(op, &a, &b, alpha);
}

static Tensor& inplace(uint32_t op, Tensor& self, const Tensor& o, float alpha = 1.f) {
  if (is_int(self) && (op == ew::ADD || op == ew::SUB)) {
    if (cpu_scalar(o)) {
      NEED(!o.is_floating_point(), "int tensor with a float scalar");
      Scalar s = o.item<int64_t>() * (int64_t)alpha;
      int_binary(op, self, nullptr, &s, &self);
    } else {
      NEED(alpha == 1.f && o.scalar_type() == self.scalar_type(), "integer add");
      int_binary(op, self, &o, nullptr, &self);
    }
    return self;
  }
  if (cpu_scalar(o)) ew_into(op, &self, nullptr, self, alpha, o.item<float>());
  else ew_into(op, &self, &o, self, alpha);
  return self;
}

static Tensor add_t(const Tensor& a, const Tensor& b, const Scalar& al) { return binary(ew::ADD, a, b, al.to<float>()); }
static Tensor sub_t(const Tensor& a, const Tensor& b, const Scalar& al) { return binary(ew::SUB, a, b, al.to<float>()); }
static Tensor mul_t(const Tensor& a, const Tensor& b) { return binary(ew::MUL, a, b); }
static Tensor div_t(const Tensor& a, const Tensor& b) { return binary(ew::DIV, a, b); }
static Tensor max_t(const Tensor& a, const Tensor& b) { NEED(is_dev(a) && is_dev(b), "device tensors"); return binary(ew::MAX, a, b); }
static Tensor min_t(const Tensor& a, const Tensor& b) { NEED(is_dev(a) && is_dev(b), "device tensors"); return binary(ew::MIN, a, b); }
static Tensor xlogy_t(const Tensor& a, const Tensor& b) { return binary(ew::XLOGY, a, b); }
static Tensor add_s(const Tensor& a, const Scalar& b, const Scalar& al) {
  if (is_int(a)) { Scalar s = b.to<int64_t>() * al.to<int64_t>(); NEED(!b.isFloatingPoint(), "promotion"); return int_binary(ew::ADD, a, nullptr, &s, nullptr); }
  return ew_new(ew::ADD, &a, nullptr, al.to<float>(), b.to<float>());
}
static Tensor sub_s(const Tensor& a, const Scalar& b, const Scalar& al) { return ew_new(ew::SUB, &a, nullptr, al.to<float>(), b.to<float>()); }
static Tensor mul_s(const Tensor& a, const Scalar& b) { return ew_new(ew::MUL, &a, nullptr, 1.f, b.to<float>()); }
static Tensor div_s(const Tensor& a, const Scalar& b) { return ew_new(ew::DIV, &a, nullptr, 1.f, b.to<float>()); }
static Tensor rsub_s(const Tensor& a, const Scalar& b, const Scalar& al) { return ew_new(ew::RSUB, &a, nullptr, al.to<float>(), b.to<float>()); }
static Tensor rsub_t(const Tensor& a, const Tensor& b, const Scalar& al) {
  if (cpu_scalar(b)) return ew_new(ew::RSUB, &a, nullptr, al.to<float>(), b.item<float>());
  return binary(ew::SUB, b, a, al.to<float>());
}
static Tensor pow_s(const Tensor& a, const Scalar& e) { return ew_new(ew::POW, &a, nullptr, e.to<float>()); }
static Tensor lerp_s(const Tensor& a, const Tensor& b, const Scalar& w) { NEED(f32(a) && f32(b), "float32"); return ew_new(ew::LERP, &a, &b, w.to<float>()); }
static Tensor clamp_(const Tensor& x, const std::optional<Scalar>& lo, const std::optional<Scalar>& hi) {
  float inf = std::numeric_limits<float>::infinity();
  return ew_new(ew::CLAMP, &x, nullptr, lo ? lo->to<float>() : -inf, hi ? hi->to<float>() : inf);
}
static Tensor clamp_min_(const Tensor& x, const Scalar& lo) { return clamp_(x, lo, std::nullopt); }
static Tensor clamp_max_(const Tensor& x, const Scalar& hi) { return clamp_(x, std::nullopt, hi); }
static Tensor threshold_bwd(const Tensor& g, const Tensor& x, const Scalar& th) { NEED(f32(g) && f32(x), "float32"); return ew_new(ew::THRESH_BWD, &g, &x, th.to<float>()); }
static Tensor gelu_(const Tensor& x, c10::string_view approx) { return ew_new(ew::GELU, &x, nullptr, approx == "tanh" ? 1.f : 0.f); }
static Tensor gelu_bwd(const Tensor& g, const Tensor& x, c10::string_view approx) {
  NEED(f32(g) && f32(x), "float32");
  return ew_new(ew::GELU_BWD, &g, &x, approx == "tanh" ? 1.f : 0.f);
}

static Tensor& add__t(Tensor& s, const Tensor& o, const Scalar& al) { return inplace(ew::ADD, s, o, al.to<float>()); }
static Tensor& sub__t(Tensor& s, const Tensor& o, const Scalar& al) { return inplace(ew::SUB, s, o, al.to<float>()); }
static Tensor& mul__t(Tensor& s, const Tensor& o) { return inplace(ew::MUL, s, o); }
static Tensor& div__t(Tensor& s, const Tensor& o) { return inplace(ew::DIV, s, o); }
static Tensor& add__s(Tensor& s, const Scalar& o, const Scalar& al) {
  if (is_int(s)) { NEED(!o.isFloatingPoint(), "promotion"); Scalar v = o.to<int64_t>() * al.to<int64_t>(); int_binary(ew::ADD, s, nullptr, &v, &s); return s; }
  ew_into(ew::ADD, &s, nullptr, s, al.to<float>(), o.to<float>());
  return s;
}
static Tensor& sub__s(Tensor& s, const Scalar& o, const Scalar& al) { ew_into(ew::SUB, &s, nullptr, s, al.to<float>(), o.to<float>()); return s; }
static Tensor& mul__s(Tensor& s, const Scalar& o) { ew_into(ew::MUL, &s, nullptr, s, 1.f, o.to<float>()); return s; }
static Tensor& div__s(Tensor& s, const Scalar& o) { ew_into(ew::DIV, &s, nullptr, s, 1.f, o.to<float>()); return s; }
static Tensor& lerp__s(Tensor& s, const Tensor& e, const Scalar& w) { ew_into(ew::LERP, &s, &e, s, w.to<float>()); return s; }
static Tensor& clamp__(Tensor& s, const std::optional<Scalar>& lo, const std::optional<Scalar>& hi) {
  float inf = std::numeric_limits<float>::infinity();
  ew_into(ew::CLAMP, &s, nullptr, s, lo ? lo->to<float>() : -inf, hi ? hi->to<float>() : inf);
  return s;
}
static Tensor& clamp_min__(Tensor& s, const Scalar& lo) { return clamp__(s, lo, std::nullopt); }
static Tensor& clamp_max__(Tensor& s, const Scalar& hi) { return clamp__(s, std::nullopt, hi); }
static Tensor& pow__s(Tensor& s, const Scalar& e) { ew_into(ew::POW, &s, nullptr, s, e.to<float>()); return s; }
static Tensor& addcmul__(Tensor& s, const Tensor& t1, const Tensor& t2, const Scalar& v) {
  NEED(f32(s) && f32(t1) && f32(t2), "float32");
  if (t1.is_same(t2)) { ew_into(ew::ADD_SQ, &s, &t1, s, v.to<float>()); return s; }  // s + v * t1^2 (Adam's second moment)
  Tensor p = ew_new(ew::MUL, &t1, &t2);
  ew_into(ew::ADD, &s, &p, s, v.to<float>());
  return s;
}
static Tensor& addcdiv__(Tensor& s, const Tensor& t1, const Tensor& t2, const Scalar& v) {
  NEED(f32(s) && f32(t1) && f32(t2), "float32");
  Tensor q = ew_new(ew::DIV, &t1, &t2);
  ew_into(ew::ADD, &s, &q, s, v.to<float>());
  return s;
}
static Tensor addcmul_(const Tensor& s, const Tensor& t1, const Tensor& t2, const Scalar& v) {
  NEED(f32(s), "float32");
  Tensor out = empty_dev(at::infer_size(s.sizes(), at::infer_size(t1.sizes(), t2.sizes())), at::kFloat);
  copy_dev(s, out);
  return addcmul__(out, t1, t2, v);
}
static Tensor addcdiv_(const Tensor& s, const Tensor& t1, const Tensor& t2, const Scalar& v) {
  NEED(f32(s), "float32");
  Tensor out = empty_dev(at::infer_size(s.sizes(), at::infer_size(t1.sizes(), t2.sizes())), at::kFloat);
  copy_dev(s, out);
  return addcdiv__(out, t1, t2, v);
}

// .out variants (decompositions and torch.compile-less code paths call these)
static Tensor& binary_out(uint32_t op, const Tensor& a, const Tensor& b, float alpha, Tensor& out) {
  NEED(f32(out) && f32(a) && (f32(b) || cpu_scalar(b)), "float32");
  at::native::resize_output(out, at::infer_size(a.sizes(), b.sizes()));
  if (cpu_scalar(b)) ew_into(op, &a, nullptr, out, alpha, b.item<float>());
  else ew_into(op, &a, &b, out, alpha);
  return out;
}
static Tensor& add_out(const Tensor& a, const Tensor& b, const Scalar& al, Tensor& o) { return binary_out(ew::ADD, a, b, al.to<float>(), o); }
static Tensor& sub_out(const Tensor& a, const Tensor& b, const Scalar& al, Tensor& o) { return binary_out(ew::SUB, a, b, al.to<float>(), o); }
static Tensor& mul_out(const Tensor& a, const Tensor& b, Tensor& o) { return binary_out(ew::MUL, a, b, 1.f, o); }
static Tensor& div_out(const Tensor& a, const Tensor& b, Tensor& o) { return binary_out(ew::DIV, a, b, 1.f, o); }

#define UNARY(NAME, OP)                                                                            \
  static Tensor NAME(const Tensor& a) { return ew_new(OP, &a, nullptr); }                          \
  static Tensor& NAME##_ip(Tensor& a) { ew_into(OP, &a, nullptr, a); return a; }
UNARY(neg_, ew::NEG)
UNARY(relu_, ew::RELU)
UNARY(exp_, ew::EXP)
UNARY(sqrt_, ew::SQRT)
UNARY(sigmoid_, ew::SIGMOID)
UNARY(silu_, ew::SILU)
UNARY(hardswish_, ew::HARDSWISH)
UNARY(hardsigmoid_, ew::HARDSIGMOID)
UNARY(tanh_, ew::TANH)
UNARY(log_, ew::LOG)
UNARY(abs_, ew::ABS)
UNARY(rsqrt_, ew::RSQRT)
UNARY(recip_, ew::RECIP)
UNARY(sin_, ew::SIN)
UNARY(cos_, ew::COS)

#define BWD(NAME, OP) \
  static Tensor NAME(const Tensor& g, const Tensor& y) { NEED(f32(g) && f32(y), "float32"); return ew_new(OP, &g, &y); }
BWD(sigmoid_bwd, ew::SIGMOID_BWD)
BWD(silu_bwd, ew::SILU_BWD)
BWD(hardswish_bwd, ew::HARDSWISH_BWD)
BWD(hardsigmoid_bwd, ew::HARDSIGMOID_BWD)
BWD(tanh_bwd, ew::TANH_BWD)

// ------------------------------------------------------------------------------------------------
// integer add / subtract (contiguous)
// ------------------------------------------------------------------------------------------------

static Tensor int_binary(uint32_t op, const Tensor& a_, const Tensor* b_, const Scalar* s, const Tensor* out_) {
  NEED(is_int(a_), "integer");
  NEED(!b_ || (is_dev(*b_) && b_->scalar_type() == a_.scalar_type() && b_->sizes() == a_.sizes()),
       "integer op needs same dtype and shape");
  Tensor a = contig(a_);
  Tensor b = b_ ? contig(*b_) : Tensor();
  Tensor out = out_ ? *out_ : empty_dev(a.sizes(), a.scalar_type());
  NEED(out.is_contiguous(), "contiguous output");
  int64_t n = a.numel();
  if (!n) return out;
  bool wide = a.scalar_type() == at::kLong;
  uint64_t sv = s ? (uint64_t)s->to<int64_t>() : 0;
  bool fa = bufof(a) == bufof(out), fb = b_ && bufof(b) == bufof(out);
  Buf* ub = uniform({u(n), op == ew::SUB ? 1u : 0u, wide ? 1u : 0u, b_ ? 0u : 1u, u(off_bytes(a) / 4),
                     b_ ? u(off_bytes(b) / 4) : 0u, u(off_bytes(out) / 4), fa ? 1u : 0u, (uint32_t)sv,
                     (uint32_t)(sv >> 32), fb ? 1u : 0u, 0u});
  uint32_t gx, gy;
  grid(n, 256, gx, gy);
  launch("iarith", {bind_buf(fa ? dummy_buf() : bufof(a)), bind_buf((b_ && !fb) ? bufof(b) : dummy_buf()), bind(out),
                    bind_buf(ub)}, gx, gy);
  return out;
}

// ------------------------------------------------------------------------------------------------
// comparisons, where, masked_fill
// ------------------------------------------------------------------------------------------------

static int cmp_kind(const Tensor& t) {
  switch (t.scalar_type()) {
    case at::kFloat: return 0;
    case at::kInt: return 1;
    case at::kLong: return 2;
    case at::kBool:
    case at::kByte: return 3;
    default: return -1;
  }
}

static Tensor cmp(uint32_t op, const Tensor& a, const Tensor* b, const Scalar* s) {
  int kind = cmp_kind(a);
  NEED(kind >= 0 && is_dev(a), "comparison dtype");
  uint64_t sv = 0;
  if (b && cpu_scalar(*b)) {
    Scalar v = b->item();
    s = nullptr;
    b = nullptr;
    NEED(kind == 0 || !v.isFloatingPoint(), "integer tensor vs float scalar (type promotion)");
    sv = kind == 0 ? fbits(v.to<float>()) : (uint64_t)v.to<int64_t>();
  } else if (s) {
    NEED(kind == 0 || !s->isFloatingPoint(), "integer tensor vs float scalar (type promotion)");
    sv = kind == 0 ? fbits(s->to<float>()) : (uint64_t)s->to<int64_t>();
  } else {
    NEED(b && is_dev(*b), "comparison with a non-device tensor");
    if (b->scalar_type() != a.scalar_type()) {  // float vs int/bool: compare as float32
      NEED((kind == 0 || cmp_kind(*b) == 0) && cast_kind(a.scalar_type()) >= 0 && cast_kind(b->scalar_type()) >= 0, "comparison dtypes");
      Tensor a2 = cast_dev(a, at::kFloat), b2 = cast_dev(*b, at::kFloat);
      return cmp(op, a2, &b2, nullptr);
    }
  }
  std::vector<int64_t> shape = b ? at::infer_size(a.sizes(), b->sizes()) : a.sizes().vec();
  Tensor out = empty_dev(shape, at::kBool);
  int64_t n = out.numel();
  if (!n) return out;
  std::vector<std::vector<int64_t>> st = {out.strides().vec(), bstrides(a, shape),
                                          b ? bstrides(*b, shape) : std::vector<int64_t>(shape.size(), 0)};
  Layout L = coalesce(shape, st);
  std::vector<uint32_t> w = {u(n), op, b ? 0u : 1u, u(L.nd), u(kind), (uint32_t)sv, (uint32_t)(sv >> 32),
                             u(a.storage_offset()), b ? u(b->storage_offset()) : 0u, 0, 0, 0};
  push8(w, L.sh);
  push8(w, L.st[1]);
  push8(w, L.st[2]);
  uint32_t gx, gy;
  grid(cdiv(n, 4), 256, gx, gy);
  launch("cmp", {bind(a), bind_buf(b ? bufof(*b) : dummy_buf()), bind(out), bind_buf(uniform(w))}, gx, gy);
  return out;
}
#define CMP_OPS(NAME, CODE)                                                                       \
  static Tensor NAME##_s(const Tensor& a, const Scalar& v) { return cmp(CODE, a, nullptr, &v); } \
  static Tensor NAME##_t(const Tensor& a, const Tensor& b) { return cmp(CODE, a, &b, nullptr); }
CMP_OPS(lt, 0)
CMP_OPS(le, 1)
CMP_OPS(gt, 2)
CMP_OPS(ge, 3)
CMP_OPS(ne, 4)
CMP_OPS(eq, 5)

// out = cond ? a : b (a, b float32 device tensors or CPU scalars); out may be b (masked_fill_)
static void where_into(const Tensor& cond, const Tensor* a, float av, const Tensor* b, float bv, const Tensor& out) {
  NEED(is_dev(cond) && cond.scalar_type() == at::kBool, "where needs a bool device condition");
  NEED(f32(out) && (!a || f32(*a)) && (!b || f32(*b)), "float32 only");
  int64_t n = out.numel();
  if (!n) return;
  IntArrayRef shape = out.sizes();
  std::vector<std::vector<int64_t>> st = {out.strides().vec(), bstrides(cond, shape),
                                          a ? bstrides(*a, shape) : std::vector<int64_t>(shape.size(), 0),
                                          b ? bstrides(*b, shape) : std::vector<int64_t>(shape.size(), 0)};
  Layout L = coalesce(shape, st);
  bool fa = a && bufof(*a) == bufof(out), fb = b && bufof(*b) == bufof(out);
  std::vector<uint32_t> w = {u(n), u(L.nd), u(cond.storage_offset()), a ? u(a->storage_offset()) : 0u,
                             b ? u(b->storage_offset()) : 0u, a ? 0u : 1u, b ? 0u : 1u, fbits(av),
                             fbits(bv), fa ? 1u : 0u, fb ? 1u : 0u, u(out.storage_offset())};
  push8(w, L.sh);
  push8(w, L.st[1]);
  push8(w, L.st[2]);
  push8(w, L.st[3]);
  push8(w, L.st[0]);
  uint32_t gx, gy;
  grid(n, 256, gx, gy);
  launch("where3", {bind(cond), bind_buf((a && !fa) ? bufof(*a) : dummy_buf()),
                    bind_buf((b && !fb) ? bufof(*b) : dummy_buf()), bind(out), bind_buf(uniform(w))}, gx, gy);
}
static Tensor where_(const Tensor& cond, const Tensor& a, const Tensor& b) {
  NEED((f32(a) || cpu_scalar(a)) && (f32(b) || cpu_scalar(b)) && (is_dev(a) || is_dev(b)), "where on float32");
  auto shape = at::infer_size(cond.sizes(), at::infer_size(a.sizes(), b.sizes()));
  Tensor out = empty_dev(shape, at::kFloat);
  where_into(cond, is_dev(a) ? &a : nullptr, cpu_scalar(a) ? a.item<float>() : 0.f, is_dev(b) ? &b : nullptr,
             cpu_scalar(b) ? b.item<float>() : 0.f, out);
  return out;
}
static Tensor& masked_fill__s(Tensor& self, const Tensor& mask, const Scalar& v) {
  at::assert_no_internal_overlap(self);
  where_into(mask.expand(self.sizes()), nullptr, v.to<float>(), &self, 0.f, self);
  return self;
}
static Tensor& masked_fill__t(Tensor& self, const Tensor& mask, const Tensor& v) {
  NEED(cpu_scalar(v), "masked_fill with a device value");
  return masked_fill__s(self, mask, v.item());
}

// ------------------------------------------------------------------------------------------------
// fills and copies as ops
// ------------------------------------------------------------------------------------------------

static Tensor& zero_(Tensor& self) { fill_dev(self, 0); return self; }
static Tensor& fill__s(Tensor& self, const Scalar& v) { fill_dev(self, v); return self; }
static Tensor& fill__t(Tensor& self, const Tensor& v) {
  NEED(v.dim() == 0, "fill_ with a non-scalar");
  if (is_dev(v) && self.scalar_type() == v.scalar_type() && v.element_size() == 4) {  // no sync: broadcast copy
    copy_dev(v, self);
    return self;
  }
  fill_dev(self, to_host(v).item());
  return self;
}
static Tensor ones_like_(const Tensor& self, std::optional<at::ScalarType> dt, std::optional<at::Layout>,
                         std::optional<at::Device>, std::optional<bool>, std::optional<at::MemoryFormat>) {
  Tensor out = empty_dev(self.sizes(), dt.value_or(self.scalar_type()));
  fill_dev(out, 1);
  return out;
}
static Tensor zeros_like_(const Tensor& self, std::optional<at::ScalarType> dt, std::optional<at::Layout>,
                          std::optional<at::Device>, std::optional<bool>, std::optional<at::MemoryFormat>) {
  Tensor out = empty_dev(self.sizes(), dt.value_or(self.scalar_type()));
  fill_dev(out, 0);
  return out;
}
static Tensor clone_(const Tensor& self, std::optional<at::MemoryFormat>) {
  Tensor out = empty_dev(self.sizes(), self.scalar_type());
  copy_dev(self, out);
  return out;
}
static Tensor& copy__(Tensor& self, const Tensor& src, bool) {
  if (!is_dev(self)) {  // device -> host (dispatch picks this kernel whenever either side is on the device)
    self.copy_(download(src));
    return self;
  }
  if (is_dev(src) && src.scalar_type() == self.scalar_type()) copy_dev(src, self);
  else if (is_dev(src) && cast_kind(src.scalar_type()) >= 0 && cast_kind(self.scalar_type()) >= 0) copy_dev(cast_dev(src, self.scalar_type()), self);
  else if (!is_dev(src)) host_copy_into(src, self);
  else host_copy_into(download(src), self);
  return self;
}
static Tensor _to_copy_(const Tensor& self, std::optional<at::ScalarType> dtype, std::optional<at::Layout>,
                        std::optional<at::Device> device, std::optional<bool>, bool, std::optional<at::MemoryFormat>) {
  at::ScalarType dt = dtype.value_or(self.scalar_type());
  if (!is_dev(self)) {  // CPU -> device (this is how .to("omarchy_mac_torch") arrives)
    Tensor out = empty_dev(self.sizes(), dt);
    host_copy_into(self, out);
    return out;
  }
  if (device && device->type() == c10::DeviceType::CPU) return download(self).to(dt);
  Tensor out = empty_dev(self.sizes(), dt);
  if (dt == self.scalar_type()) copy_dev(self, out);
  else if (cast_kind(dt) >= 0 && cast_kind(self.scalar_type()) >= 0 && !self.is_conj() && !self.is_neg()) return cast_dev(self, dt);
  else host_copy_into(download(self), out);  // other dtype conversions run on the host
  return out;
}


// ------------------------------------------------------------------------------------------------
// classification (isnan / isinf / ...), out= helpers, where.out, tril / triu, arange
// ------------------------------------------------------------------------------------------------

static Tensor classify(uint32_t op, const Tensor& a) {
  NEED(f32(a), "float32 classification");
  Scalar zero = 0.0;
  return cmp(op, a, nullptr, &zero);
}
static Tensor& into_out(const Tensor& r, Tensor& out) {
  NEED(is_dev(out) && out.scalar_type() == r.scalar_type(), "out= with a different dtype or device");
  at::native::resize_output(out, r.sizes());
  copy_dev(r, out);
  return out;
}
static Tensor isnan_(const Tensor& a) { return classify(6, a); }
static Tensor isinf_(const Tensor& a) { return classify(7, a); }
static Tensor isneginf_(const Tensor& a) { return classify(8, a); }
static Tensor isposinf_(const Tensor& a) { return classify(9, a); }
static Tensor& isneginf_out(const Tensor& a, Tensor& out) { return into_out(classify(8, a), out); }
static Tensor& isposinf_out(const Tensor& a, Tensor& out) { return into_out(classify(9, a), out); }

static Tensor& where_out(const Tensor& cond, const Tensor& a, const Tensor& b, Tensor& out) {
  NEED(f32(out), "float32 where");
  return into_out(where_(cond, a, b), out);
}

static Tensor tri_impl(const Tensor& x_, int64_t k, uint32_t mode) {
  NEED(is_dev(x_) && x_.dim() >= 2 && (x_.element_size() == 4 || x_.element_size() == 1), "tril/triu of a 4- or 1-byte dtype, >= 2-D");
  Tensor x = contig(x_);
  Tensor out = empty_dev(x.sizes(), x.scalar_type());
  int64_t n = x.numel();
  if (!n) return out;
  uint32_t es = (uint32_t)x.element_size();
  NEED(es == 4 || off_bytes(x) % 4 == 0, "aligned byte tensor");
  Buf* ub = uniform({u(n), u(x.size(-2)), u(x.size(-1)), (uint32_t)(int32_t)k, mode, es, u(es == 4 ? off_bytes(x) / 4 : off_bytes(x)), 0});
  uint32_t gx, gy;
  grid(es == 4 ? n : cdiv(n, 4), 256, gx, gy);
  launch("tri", {bind(x), bind(out), bind_buf(ub)}, gx, gy);
  return out;
}
static Tensor tril_(const Tensor& x, int64_t k) { return tri_impl(x, k, 0); }
static Tensor triu_(const Tensor& x, int64_t k) { return tri_impl(x, k, 1); }
static Tensor& tril_out(const Tensor& x, int64_t k, Tensor& out) { return into_out(tri_impl(x, k, 0), out); }
static Tensor& triu_out(const Tensor& x, int64_t k, Tensor& out) { return into_out(tri_impl(x, k, 1), out); }
static Tensor& tril_ip(Tensor& x, int64_t k) { Tensor r = tri_impl(x, k, 0); copy_dev(r, x); return x; }
static Tensor& triu_ip(Tensor& x, int64_t k) { Tensor r = tri_impl(x, k, 1); copy_dev(r, x); return x; }

static Tensor& arange_out(const Scalar& start, const Scalar& end, const Scalar& step, Tensor& out) {
  NEED(is_dev(out) && (out.scalar_type() == at::kFloat || out.scalar_type() == at::kLong), "float32 / int64 arange");
  bool wide = out.scalar_type() == at::kLong;
  NEED(!wide || (!start.isFloatingPoint() && !end.isFloatingPoint() && !step.isFloatingPoint()), "integer arange bounds");
  double st = start.to<double>(), en = end.to<double>(), sp = step.to<double>();
  TORCH_CHECK(sp != 0, "arange: step must be nonzero");
  int64_t n = std::max<int64_t>(0, (int64_t)std::ceil((en - st) / sp));
  NEED(n < (int64_t(1) << 31), "arange length");
  at::native::resize_output(out, {n});
  NEED(out.is_contiguous(), "contiguous out");
  if (!n) return out;
  Buf* ub = uniform({u(n), wide ? 1u : 0u, u(out.storage_offset()), 0, fbits((float)st), fbits((float)sp),
                     (uint32_t)(int32_t)start.to<int64_t>(), (uint32_t)(int32_t)step.to<int64_t>()});
  uint32_t gx, gy;
  grid(n, 256, gx, gy);
  launch("arange", {bind(out), bind_buf(ub)}, gx, gy);
  return out;
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  OMT_IMPL(m, add_Tensor, add_t);
  OMT_IMPL(m, add_Scalar, add_s);
  OMT_IMPL(m, sub_Tensor, sub_t);
  OMT_IMPL(m, sub_Scalar, sub_s);
  OMT_IMPL(m, mul_Tensor, mul_t);
  OMT_IMPL(m, mul_Scalar, mul_s);
  OMT_IMPL(m, div_Tensor, div_t);
  OMT_IMPL(m, div_Scalar, div_s);
  OMT_IMPL(m, rsub_Scalar, rsub_s);
  OMT_IMPL(m, rsub_Tensor, rsub_t);
  OMT_IMPL(m, maximum, max_t);
  OMT_IMPL(m, minimum, min_t);
  OMT_IMPL(m, xlogy_Tensor, xlogy_t);
  OMT_IMPL(m, pow_Tensor_Scalar, pow_s);
  OMT_IMPL(m, lerp_Scalar, lerp_s);
  OMT_IMPL(m, clamp, clamp_);
  OMT_IMPL(m, clamp_min, clamp_min_);
  OMT_IMPL(m, clamp_max, clamp_max_);
  OMT_IMPL(m, threshold_backward, threshold_bwd);
  OMT_IMPL(m, gelu, gelu_);
  OMT_IMPL(m, gelu_backward, gelu_bwd);
  OMT_IMPL(m, addcmul, addcmul_);
  OMT_IMPL(m, addcdiv, addcdiv_);
  OMT_IMPL(m, add__Tensor, add__t);
  OMT_IMPL(m, add__Scalar, add__s);
  OMT_IMPL(m, sub__Tensor, sub__t);
  OMT_IMPL(m, sub__Scalar, sub__s);
  OMT_IMPL(m, mul__Tensor, mul__t);
  OMT_IMPL(m, mul__Scalar, mul__s);
  OMT_IMPL(m, div__Tensor, div__t);
  OMT_IMPL(m, div__Scalar, div__s);
  OMT_IMPL(m, lerp__Scalar, lerp__s);
  OMT_IMPL(m, clamp_, clamp__);
  OMT_IMPL(m, clamp_min_, clamp_min__);
  OMT_IMPL(m, clamp_max_, clamp_max__);
  OMT_IMPL(m, pow__Scalar, pow__s);
  OMT_IMPL(m, addcmul_, addcmul__);
  OMT_IMPL(m, addcdiv_, addcdiv__);
  OMT_IMPL(m, add_out, add_out);
  OMT_IMPL(m, sub_out, sub_out);
  OMT_IMPL(m, mul_out, mul_out);
  OMT_IMPL(m, div_out, div_out);
  OMT_IMPL(m, neg, neg_);
  OMT_IMPL(m, relu, relu_);
  OMT_IMPL(m, exp, exp_);
  OMT_IMPL(m, sqrt, sqrt_);
  OMT_IMPL(m, sigmoid, sigmoid_);
  OMT_IMPL(m, silu, silu_);
  OMT_IMPL(m, hardswish, hardswish_);
  OMT_IMPL(m, hardsigmoid, hardsigmoid_);
  OMT_IMPL(m, tanh, tanh_);
  OMT_IMPL(m, log, log_);
  OMT_IMPL(m, abs, abs_);
  OMT_IMPL(m, rsqrt, rsqrt_);
  OMT_IMPL(m, reciprocal, recip_);
  OMT_IMPL(m, sin, sin_);
  OMT_IMPL(m, cos, cos_);
  OMT_IMPL(m, neg_, neg__ip);
  OMT_IMPL(m, relu_, relu__ip);
  OMT_IMPL(m, exp_, exp__ip);
  OMT_IMPL(m, sqrt_, sqrt__ip);
  OMT_IMPL(m, sigmoid_, sigmoid__ip);
  OMT_IMPL(m, silu_, silu__ip);
  OMT_IMPL(m, hardswish_, hardswish__ip);
  OMT_IMPL(m, hardsigmoid_, hardsigmoid__ip);
  OMT_IMPL(m, tanh_, tanh__ip);
  OMT_IMPL(m, log_, log__ip);
  OMT_IMPL(m, abs_, abs__ip);
  OMT_IMPL(m, rsqrt_, rsqrt__ip);
  OMT_IMPL(m, reciprocal_, recip__ip);
  OMT_IMPL(m, sigmoid_backward, sigmoid_bwd);
  OMT_IMPL(m, silu_backward, silu_bwd);
  OMT_IMPL(m, hardswish_backward, hardswish_bwd);
  OMT_IMPL(m, hardsigmoid_backward, hardsigmoid_bwd);
  OMT_IMPL(m, tanh_backward, tanh_bwd);
  OMT_IMPL(m, lt_Scalar, lt_s);
  OMT_IMPL(m, le_Scalar, le_s);
  OMT_IMPL(m, gt_Scalar, gt_s);
  OMT_IMPL(m, ge_Scalar, ge_s);
  OMT_IMPL(m, ne_Scalar, ne_s);
  OMT_IMPL(m, eq_Scalar, eq_s);
  OMT_IMPL(m, lt_Tensor, lt_t);
  OMT_IMPL(m, le_Tensor, le_t);
  OMT_IMPL(m, gt_Tensor, gt_t);
  OMT_IMPL(m, ge_Tensor, ge_t);
  OMT_IMPL(m, ne_Tensor, ne_t);
  OMT_IMPL(m, eq_Tensor, eq_t);
  OMT_IMPL(m, where_self, where_);
  OMT_IMPL(m, masked_fill__Scalar, masked_fill__s);
  OMT_IMPL(m, masked_fill__Tensor, masked_fill__t);
  OMT_IMPL(m, zero_, zero_);
  OMT_IMPL(m, fill__Scalar, fill__s);
  OMT_IMPL(m, fill__Tensor, fill__t);
  OMT_IMPL(m, ones_like, ones_like_);
  OMT_IMPL(m, zeros_like, zeros_like_);
  OMT_IMPL(m, clone, clone_);
  OMT_IMPL(m, copy_, copy__);
  OMT_IMPL(m, _to_copy, _to_copy_);
  OMT_IMPL(m, isnan, isnan_);
  OMT_IMPL(m, isinf, isinf_);
  OMT_IMPL(m, isneginf, isneginf_);
  OMT_IMPL(m, isposinf, isposinf_);
  OMT_IMPL(m, isneginf_out, isneginf_out);
  OMT_IMPL(m, isposinf_out, isposinf_out);
  OMT_IMPL(m, where_self_out, where_out);
  OMT_IMPL(m, tril, tril_);
  OMT_IMPL(m, triu, triu_);
  OMT_IMPL(m, tril_out, tril_out);
  OMT_IMPL(m, triu_out, triu_out);
  OMT_IMPL(m, tril_, tril_ip);
  OMT_IMPL(m, triu_, triu_ip);
  OMT_IMPL(m, arange_start_out, arange_out);
}

}  // namespace omt
