// Fused conv / BatchNorm+ReLU / pool stacks: an nn.Sequential compiled once per input shape into a fixed list of
// dispatches over plan-owned buffers, driven from Python by a small autograd Function (stacks.py).
#include "omt.h"

#include <pybind11/stl.h>

namespace omt {

struct BRef {
  Buf* buf = nullptr;
  uint64_t off = 0, size = 0;
  int gslot = -1;  // >= 0: a range of the plan's gradient storage
};
struct Step {
  std::string kernel;
  std::vector<BRef> b;
  uint32_t x, y, z;
};
struct GradRef {
  int param;
  int64_t slot;
  bool bn;
};
struct Plan {
  bool train, input_grad, busy = false;
  int64_t n, c, h, w;
  std::vector<Tensor> params, rms, rvs;
  Buf* xbuf = nullptr;
  Buf* outbuf = nullptr;
  std::vector<int64_t> out_shape;
  int64_t out_numel = 0, in_numel = 0;
  std::vector<Step> fwd, bwd;
  Buf* gout = nullptr;
  Buf* gin = nullptr;
  std::vector<std::pair<int64_t, int64_t>> gslots;  // (word offset, words)
  int64_t gwords = 0;
  std::vector<GradRef> grads;
  Tensor gstore;
  std::vector<Buf*> owned, pinned;
  ~Plan() {
    OMT_LOCK;
    for (Buf* b : owned) free_buf(b);
    for (Buf* b : pinned) unpin(b);
  }
};

static Buf* nb(Plan& p, int64_t words) {
  Buf* b = alloc_buf(std::max<int64_t>(words, 1) * 4);
  p.owned.push_back(b);
  return b;
}
static BRef R_(Buf* b) { return BRef{b, 0, b->bytes, -1}; }
static BRef RT(const Tensor& t) { return BRef{bufof(t), 0, bufof(t)->bytes, -1}; }
static BRef GS(Plan& p, int64_t numel, int64_t& slot) {
  slot = (int64_t)p.gslots.size();
  p.gslots.push_back({p.gwords, numel});
  p.gwords += (std::max<int64_t>(numel, 1) + 63) / 64 * 64;
  BRef r;
  r.gslot = (int)slot;
  return r;
}
static void add_step(Plan& p, std::vector<Step>& v, const std::string& k, std::vector<BRef> b, uint32_t x, uint32_t y = 1, uint32_t z = 1) {
  for (auto& r : b)
    if (r.buf && r.buf->uniform) { pin(r.buf); p.pinned.push_back(r.buf); }  // never evicted while the plan lives
  v.push_back({k, std::move(b), x, y, z});
}
static void add_ew(Plan& p, std::vector<Step>& v, const std::string& k, std::vector<BRef> b, int64_t total) {
  uint32_t gx, gy;
  grid(total, 256, gx, gy);
  add_step(p, v, k, std::move(b), gx, gy);
}

struct PlanLayer {
  std::string kind;  // conv dw bn pool apool gap
  int64_t ci = 0, co = 0, ks = 0;
  double momentum = 0.1;
};

static void gemm_steps(Plan& p, std::vector<Step>& v, int kind, const BRef& a, const BRef& b, const BRef& out, int64_t n, int64_t ci,
                       int64_t co, int64_t h, int64_t w, int64_t ks, const std::vector<BRef>& extra = {}) {
  Geo g = same_geo(n, ci, co, h, w, ks);
  int64_t M, N, K;
  mnk(kind, g, M, N, K);
  ConvPlan pl = conv_plan(kind, g);
  if (kind == 2) {
    int64_t tiles = cdiv(M, pl.bm) * cdiv(N, pl.bn);
    int64_t slices = std::max<int64_t>(1, std::min<int64_t>((int64_t)(512 / tiles * pl.mult), K / 64));
    int64_t kslice = cdiv(cdiv(K, slices), 16) * 16;
    slices = cdiv(K, kslice);
    Buf* part = nb(p, slices * M * N);
    Buf* pr = pbuf_geo(g, M, N, K, kslice, slices);
    add_step(p, v, pl.kernel, {a, b, R_(part), R_(pr)}, u(cdiv(N, pl.bn)), u(cdiv(M, pl.bm)), u(slices));
    add_step(p, v, "gemm_reduce", {R_(part), out, R_(pbuf_reduce(M, N, slices))}, u(cdiv(M * N, 256)));
  } else {
    Buf* pr = pbuf_geo(g, M, N, K, cdiv(K, 16) * 16, 1);
    std::vector<BRef> bs = {a, b, out, R_(pr)};
    if (kind == 0) bs.push_back(R_(dummy_buf()));  // the forward kernel declares a bias binding
    for (auto& e : extra) bs.push_back(e);
    add_step(p, v, pl.kernel, bs, u(cdiv(N, pl.bn)), u(cdiv(M, pl.bm)), 1);
  }
}

struct BackRec {
  std::string kind;
  Buf *x = nullptr, *y = nullptr, *stats = nullptr, *idx = nullptr, *q = nullptr;
  int param = -1;
  int64_t n, c, co, h, w, ks;
};

static std::shared_ptr<Plan> build_plan(const std::vector<PlanLayer>& layers, std::vector<int64_t> shape, bool train, bool input_grad,
                                        std::vector<Tensor> params, std::vector<Tensor> rms, std::vector<Tensor> rvs) {
  OMT_LOCK;
  init_ctx();
  for (auto& t : params) NEED(f32(t) && t.is_contiguous() && t.storage_offset() == 0, "plan parameters must be whole contiguous tensors");
  auto P = std::make_shared<Plan>();
  Plan& p = *P;
  p.train = train;
  p.input_grad = input_grad;
  p.params = params;
  p.rms = rms;
  p.rvs = rvs;
  int64_t n = shape[0], c = shape[1], h = shape[2], w = shape[3];
  p.n = n; p.c = c; p.h = h; p.w = w;
  p.xbuf = nb(p, n * c * h * w);
  Buf* cur = p.xbuf;
  std::vector<BackRec> recs;
  int pi = 0, bi = 0;
  // eval mode: BatchNorm+ReLU followed by a conv is folded into that conv's loader
  struct Pending { bool on = false; int bn = -1, gamma = -1; } pend;
  for (size_t li = 0; li < layers.size(); li++) {
    const PlanLayer& L = layers[li];
    if (L.kind == "conv") {
      int64_t co = L.co, ks = L.ks;
      Buf* out = nb(p, n * co * h * w);
      if (pend.on) {
        gemm_steps(p, p.fwd, 3, R_(cur), RT(params[pi]), R_(out), n, c, co, h, w, ks,
                   {RT(rms[pend.bn]), RT(rvs[pend.bn]), RT(params[pend.gamma]), RT(params[pend.gamma + 1])});
        pend.on = false;
      } else {
        gemm_steps(p, p.fwd, 0, R_(cur), RT(params[pi]), R_(out), n, c, co, h, w, ks);
      }
      BackRec r; r.kind = "conv"; r.x = cur; r.param = pi; r.n = n; r.c = c; r.co = co; r.h = h; r.w = w; r.ks = ks;
      recs.push_back(r);
      cur = out; c = co; pi++;
    } else if (L.kind == "dw") {
      int64_t ks = L.ks;
      Buf* out = nb(p, n * c * h * w);
      Buf* q = uniform({u(n), u(c), u(h), u(w), 0, 0, u(ks), 0});
      if (pend.on) {
        add_ew(p, p.fwd, "dw_fwd_bn", {R_(cur), RT(params[pi]), R_(out), R_(q), RT(rms[pend.bn]), RT(rvs[pend.bn]),
                                       RT(params[pend.gamma]), RT(params[pend.gamma + 1])}, n * c * h * w);
        pend.on = false;
      } else {
        add_ew(p, p.fwd, "dw_fwd", {R_(cur), RT(params[pi]), R_(out), R_(q)}, n * c * h * w);
      }
      BackRec r; r.kind = "dw"; r.x = cur; r.q = q; r.param = pi; r.n = n; r.c = c; r.h = h; r.w = w; r.ks = ks;
      recs.push_back(r);
      cur = out; pi++;
    } else if (L.kind == "bn") {
      Buf* q = uniform({u(n), u(c), u(h), u(w), 0, 0, 0, 0});
      bool next_is_conv = false;
      if (li + 1 < layers.size() && g_fold > 0) {
        const PlanLayer& N = layers[li + 1];
        next_is_conv = g_fold >= 2 ? (N.kind == "conv" || N.kind == "dw") : (N.kind == "conv" && N.ks == 1);
      }
      if (!train && next_is_conv) {  // fold into the consumer
        pend.on = true; pend.bn = bi; pend.gamma = pi;
        bi++; pi += 2;
        continue;
      }
      Buf* y = nb(p, n * c * h * w);
      Buf* stats = nullptr;
      if (train) {
        stats = nb(p, 2 * c);
        int64_t count = n * h * w;
        Buf* r = uniform({fbits((float)L.momentum), fbits((float)count / std::max<int64_t>(count - 1, 1)), 0, 0});
        int64_t S = n >= 8 ? 8 : 1;  // each channel's reduction is split over S workgroups
        Buf* qs = uniform({u(n), u(c), u(h), u(w), 0, 0, u(S), 0});
        Buf* part = nb(p, c * S * 2);
        add_step(p, p.fwd, "bn_stats_part", {R_(cur), R_(part), R_(qs)}, u(c), u(S));
        add_step(p, p.fwd, "bn_stats_fin", {R_(cur), R_(part), R_(stats), RT(rms[bi]), RT(rvs[bi]), R_(qs), R_(r)}, u(cdiv(c, 64)));
        add_ew(p, p.fwd, "bn_relu", {R_(cur), R_(stats), RT(params[pi]), RT(params[pi + 1]), R_(y), R_(q)}, n * c * h * w);
      } else {
        add_ew(p, p.fwd, "bn_relu_rv", {R_(cur), RT(rms[bi]), RT(params[pi]), RT(params[pi + 1]), R_(y), R_(q), RT(rvs[bi])}, n * c * h * w);
      }
      BackRec r; r.kind = "bn"; r.x = cur; r.y = y; r.stats = stats; r.q = q; r.param = pi; r.n = n; r.c = c; r.h = h; r.w = w;
      recs.push_back(r);
      cur = y; bi++; pi += 2;
    } else if (L.kind == "pool") {
      int64_t oh = h / 2, ow = w / 2;
      Buf* out = nb(p, n * c * oh * ow);
      Buf* idx = nb(p, n * c * oh * ow);
      Buf* q = uniform({u(n), u(c), u(h), u(w), u(oh), u(ow), 0, 0});
      add_ew(p, p.fwd, "pool_fwd", {R_(cur), R_(out), R_(idx), R_(q)}, n * c * oh * ow);
      BackRec r; r.kind = "pool"; r.idx = idx; r.q = q; r.n = n; r.c = c; r.h = h; r.w = w;
      recs.push_back(r);
      cur = out; h = oh; w = ow;
    } else if (L.kind == "apool") {
      int64_t oh = h / 2, ow = w / 2;
      Buf* out = nb(p, n * c * oh * ow);
      Buf* q = uniform({u(n), u(c), u(h), u(w), u(oh), u(ow), 0, 0});
      add_ew(p, p.fwd, "avgpool_fwd", {R_(cur), R_(out), R_(q)}, n * c * oh * ow);
      BackRec r; r.kind = "apool"; r.q = q; r.n = n; r.c = c; r.h = h; r.w = w;
      recs.push_back(r);
      cur = out; h = oh; w = ow;
    } else {  // gap
      Buf* out = nb(p, n * c);
      Buf* q = uniform({u(n), u(c), u(h), u(w), 0, 0, 0, 0});
      add_ew(p, p.fwd, "gap_fwd", {R_(cur), R_(out), R_(q)}, n * c);
      BackRec r; r.kind = "gap"; r.q = q; r.n = n; r.c = c; r.h = h; r.w = w;
      recs.push_back(r);
      cur = out; h = 1; w = 1;
    }
  }
  p.outbuf = cur;
  p.out_shape = {n, c, h, w};
  p.out_numel = n * c * h * w;
  if (!train) return P;

  // ---- backward ----
  p.gout = nb(p, p.out_numel);
  Buf* g = p.gout;
  for (int ri = (int)recs.size() - 1; ri >= 0; ri--) {
    BackRec& r = recs[ri];
    bool first = ri == 0;
    if (r.kind == "pool") {
      Buf* gx = nb(p, r.n * r.c * r.h * r.w);
      add_ew(p, p.bwd, "pool_bwd", {R_(g), R_(r.idx), R_(gx), R_(r.q)}, r.n * r.c * r.h * r.w);
      g = gx;
    } else if (r.kind == "apool") {
      Buf* gx = nb(p, r.n * r.c * r.h * r.w);
      add_ew(p, p.bwd, "avgpool_bwd", {R_(g), R_(gx), R_(r.q)}, r.n * r.c * r.h * r.w);
      g = gx;
    } else if (r.kind == "gap") {
      Buf* gx = nb(p, r.n * r.c * r.h * r.w);
      add_ew(p, p.bwd, "gap_bwd", {R_(g), R_(gx), R_(r.q)}, r.n * r.c * r.h * r.w);
      g = gx;
    } else if (r.kind == "bn") {
      int64_t slot;
      BRef dgb = GS(p, 2 * r.c, slot);
      p.grads.push_back({r.param, slot, true});
      Buf* gx = nb(p, r.n * r.c * r.h * r.w);
      int64_t S = r.n >= 8 ? 8 : 1;
      Buf* qs = uniform({u(r.n), u(r.c), u(r.h), u(r.w), 0, 0, u(S), 0});
      Buf* part = nb(p, r.c * S * 2);
      add_step(p, p.bwd, "bn_bwd_part", {R_(g), R_(r.x), R_(r.stats), R_(r.y), R_(part), R_(qs)}, u(r.c), u(S));
      add_step(p, p.bwd, "bn_bwd_fin", {R_(part), dgb, R_(qs)}, u(cdiv(r.c, 64)));
      add_ew(p, p.bwd, "bn_bwd_apply", {R_(g), R_(r.x), R_(r.stats), R_(r.y), RT(params[r.param]), dgb, R_(gx), R_(r.q)},
             r.n * r.c * r.h * r.w);
      g = gx;
    } else if (r.kind == "dw") {
      int64_t slot;
      BRef gw = GS(p, r.c * r.ks * r.ks, slot);
      p.grads.push_back({r.param, slot, false});
      add_step(p, p.bwd, "dw_gwt", {R_(g), R_(r.x), gw, R_(r.q)}, u(r.c * r.ks * r.ks));
      if (!first || input_grad) {
        Buf* gx = nb(p, r.n * r.c * r.h * r.w);
        add_ew(p, p.bwd, "dw_gin", {R_(g), RT(params[r.param]), R_(gx), R_(r.q)}, r.n * r.c * r.h * r.w);
        g = gx;
      }
      p.in_numel = r.n * r.c * r.h * r.w;
    } else {  // conv
      int64_t slot;
      BRef gw = GS(p, r.co * r.c * r.ks * r.ks, slot);
      p.grads.push_back({r.param, slot, false});
      gemm_steps(p, p.bwd, 2, R_(g), R_(r.x), gw, r.n, r.c, r.co, r.h, r.w, r.ks);
      if (!first || input_grad) {
        Buf* gx = nb(p, r.n * r.c * r.h * r.w);
        gemm_steps(p, p.bwd, 1, R_(g), RT(params[r.param]), R_(gx), r.n, r.c, r.co, r.h, r.w, r.ks);
        g = gx;
      }
      p.in_numel = r.n * r.c * r.h * r.w;
    }
  }
  p.gin = g;
  p.gstore = empty_dev({std::max<int64_t>(p.gwords, 64)}, at::kFloat);
  return P;
}

static void run_steps(Plan& p, const std::vector<Step>& steps, Buf* gbuf) {
  std::vector<Binding> b;
  for (const Step& s : steps) {
    b.clear();
    for (const BRef& r : s.b) {
      if (r.gslot >= 0) {
        auto& sl = p.gslots[r.gslot];
        b.push_back({gbuf, (uint64_t)sl.first * 4, (uint64_t)std::max<int64_t>(sl.second, 1) * 4});
      } else {
        b.push_back({r.buf, r.off, r.size});
      }
    }
    launch(s.kernel.c_str(), b, s.x, s.y, s.z);
  }
}
static void copy_words(Buf* src, uint64_t src_off, Buf* dst, uint64_t dst_off, int64_t n) {
  uint32_t gx, gy;
  grid(n, 256, gx, gy);
  launch("icopy", {bind_buf(src), bind_buf(dst), bind_buf(uniform({u(n), u(src_off), u(dst_off), 0}))}, gx, gy);
}

static Tensor plan_forward(Plan& p, const Tensor& x_) {
  OMT_LOCK;
  Tensor x = contig(x_);
  TORCH_CHECK(f32(x) && x.numel() == p.n * p.c * p.h * p.w, "omarchy-mac-torch plan: bad input");
  copy_words(bufof(x), off_words(x), p.xbuf, 0, x.numel());
  run_steps(p, p.fwd, nullptr);
  Tensor out = empty_dev(p.out_shape, at::kFloat);
  copy_words(p.outbuf, 0, bufof(out), 0, p.out_numel);
  return out;
}

static std::pair<std::optional<Tensor>, std::vector<Tensor>> plan_backward(Plan& p, const Tensor& g_) {
  OMT_LOCK;
  Tensor g = contig(g_);
  if (p.gstore.storage().use_count() > 1) p.gstore = empty_dev({std::max<int64_t>(p.gwords, 64)}, at::kFloat);  // previous grads alive
  copy_words(bufof(g), off_words(g), p.gout, 0, g.numel());
  run_steps(p, p.bwd, bufof(p.gstore));
  std::vector<Tensor> grads(p.params.size());
  for (auto& gr : p.grads) {
    int64_t off = p.gslots[gr.slot].first;
    if (gr.bn) {
      int64_t half = p.gslots[gr.slot].second / 2;
      grads[gr.param] = p.gstore.as_strided({half}, {1}, off);
      grads[gr.param + 1] = p.gstore.as_strided({half}, {1}, off + half);
    } else {
      grads[gr.param] = p.gstore.as_strided(p.params[gr.param].sizes(), p.params[gr.param].strides(), off);
    }
  }
  std::optional<Tensor> gx;
  if (p.input_grad) {
    Tensor t = empty_dev({p.n, p.c, p.h, p.w}, at::kFloat);
    copy_words(p.gin, 0, bufof(t), 0, p.in_numel);
    gx = t;
  }
  return {gx, grads};
}

void bind_plans(py::module_& m) {
  py::class_<Plan, std::shared_ptr<Plan>>(m, "Plan")
      .def_readwrite("busy", &Plan::busy)
      .def("forward", [](Plan& p, const Tensor& x) { return plan_forward(p, x); })
      .def("backward", [](Plan& p, const Tensor& g) {
        auto r = plan_backward(p, g);
        return py::make_tuple(r.first ? py::cast(*r.first) : py::none(), r.second);
      });
  m.def("make_plan", [](std::vector<std::tuple<std::string, int64_t, int64_t, int64_t, double>> layers, std::vector<int64_t> shape,
                        bool train, bool input_grad, std::vector<Tensor> params, std::vector<Tensor> rms, std::vector<Tensor> rvs) {
    std::vector<PlanLayer> L;
    for (auto& t : layers) {
      PlanLayer l;
      l.kind = std::get<0>(t); l.ci = std::get<1>(t); l.co = std::get<2>(t); l.ks = std::get<3>(t); l.momentum = std::get<4>(t);
      L.push_back(l);
    }
    try {
      return build_plan(L, shape, train, input_grad, params, rms, rvs);
    } catch (const Unsupported& e) {
      throw std::runtime_error(std::string("omarchy-mac-torch: cannot build a fused stack: ") + e.why);
    }
  });
}

}  // namespace omt
