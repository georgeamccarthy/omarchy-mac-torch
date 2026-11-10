// omarchy-mac-torch native backend: a PrivateUse1 PyTorch backend in C++ on wgpu-native (WGSL kernels over
// Vulkan/Honeykrisp). Python only registers kernel sources and runs the model; every op, the memory
// pool, command recording and the CPU fallback live here.

#include <ATen/ATen.h>
#include <ATen/EmptyTensor.h>
#include <ATen/CPUGeneratorImpl.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/native/CPUFallback.h>
#include <ATen/ops/_copy_from.h>
#include <ATen/ops/_copy_from_and_resize.h>
#include <c10/core/Allocator.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <ATen/detail/PrivateUse1HooksInterface.h>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/extension.h>
#include <torch/library.h>

#include <webgpu.h>
#include <wgpu.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <deque>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace omt {

using at::Tensor;
using c10::IntArrayRef;
using c10::Scalar;
constexpr auto kDev = c10::DeviceType::PrivateUse1;

// ------------------------------------------------------------------------------------------------
// wgpu context
// ------------------------------------------------------------------------------------------------

static WGPUStringView sv(const char* s) { return WGPUStringView{s, s ? std::strlen(s) : 0}; }

struct Ctx {
  WGPUInstance instance = nullptr;
  WGPUAdapter adapter = nullptr;
  WGPUDevice device = nullptr;
  WGPUQueue queue = nullptr;
  bool ready = false;
};
static Ctx C;

struct Flag {
  std::atomic<bool> done{false};
};
static void on_adapter(WGPURequestAdapterStatus st, WGPUAdapter a, WGPUStringView, void* u1, void* u2) {
  if (st == WGPURequestAdapterStatus_Success) *(WGPUAdapter*)u1 = a;
  ((Flag*)u2)->done = true;
}
static void on_device(WGPURequestDeviceStatus st, WGPUDevice d, WGPUStringView, void* u1, void* u2) {
  if (st == WGPURequestDeviceStatus_Success) *(WGPUDevice*)u1 = d;
  ((Flag*)u2)->done = true;
}
static void on_uncaptured(WGPUDevice const*, WGPUErrorType, WGPUStringView msg, void*, void*) {
  fprintf(stderr, "[omarchy-mac-torch] wgpu error: %.*s\n", (int)msg.length, msg.data);
}

static void wait_flag(Flag* f) {
  while (!f->done) {
    if (C.device) wgpuDevicePoll(C.device, 0, nullptr);
    wgpuInstanceProcessEvents(C.instance);
    if (!f->done) std::this_thread::sleep_for(std::chrono::microseconds(20));
  }
}

static void init_ctx() {
  if (C.ready) return;
  WGPUInstanceDescriptor idesc = {};
  C.instance = wgpuCreateInstance(&idesc);
  WGPURequestAdapterOptions opts = {};
  opts.powerPreference = WGPUPowerPreference_HighPerformance;
  WGPURequestAdapterCallbackInfo ci = {};
  ci.mode = WGPUCallbackMode_AllowSpontaneous;
  ci.callback = on_adapter;
  Flag af;
  ci.userdata1 = &C.adapter;
  ci.userdata2 = &af;
  wgpuInstanceRequestAdapter(C.instance, &opts, ci);
  wait_flag(&af);
  TORCH_CHECK(C.adapter, "omarchy-mac-torch: no GPU adapter");
  WGPULimits lim = {};
  wgpuAdapterGetLimits(C.adapter, &lim);
  WGPUDeviceDescriptor dd = {};
  dd.requiredLimits = &lim;
  dd.uncapturedErrorCallbackInfo.callback = on_uncaptured;
  WGPURequestDeviceCallbackInfo dci = {};
  dci.mode = WGPUCallbackMode_AllowSpontaneous;
  dci.callback = on_device;
  Flag df;
  dci.userdata1 = &C.device;
  dci.userdata2 = &df;
  wgpuAdapterRequestDevice(C.adapter, &dd, dci);
  wait_flag(&df);
  TORCH_CHECK(C.device, "omarchy-mac-torch: could not create device");
  C.queue = wgpuDeviceGetQueue(C.device);
  C.ready = true;
}

// ------------------------------------------------------------------------------------------------
// GPU buffers: pooled by size class (a freed buffer is reused; the queue is in-order, so this is safe)
// ------------------------------------------------------------------------------------------------

struct Buf {
  WGPUBuffer b;
  size_t bytes;
};

static std::unordered_map<size_t, std::vector<Buf*>> g_free;
static size_t g_reserved = 0, g_inuse = 0;
static std::mutex g_pool_mu;

static size_t size_class(size_t bytes) {
  size_t c = 256;
  while (c < bytes) c <<= 1;
  return c;
}

static Buf* alloc_buf(size_t bytes) {
  init_ctx();
  size_t cls = size_class(bytes);
  {
    std::lock_guard<std::mutex> g(g_pool_mu);
    auto& v = g_free[cls];
    if (!v.empty()) {
      Buf* b = v.back();
      v.pop_back();
      g_inuse += cls;
      return b;
    }
  }
  WGPUBufferDescriptor d = {};
  d.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc | WGPUBufferUsage_CopyDst;
  d.size = cls;
  Buf* b = new Buf{wgpuDeviceCreateBuffer(C.device, &d), cls};
  g_reserved += cls;
  g_inuse += cls;
  return b;
}

static void free_buf(void* p) {
  Buf* b = (Buf*)p;
  std::lock_guard<std::mutex> g(g_pool_mu);
  g_inuse -= b->bytes;
  g_free[b->bytes].push_back(b);
}

// ------------------------------------------------------------------------------------------------
// Pipelines, bind groups, recorder
// ------------------------------------------------------------------------------------------------

struct Pipe {
  WGPUComputePipeline p;
  WGPUBindGroupLayout layout;
};
static std::vector<Pipe> g_pipes;
static std::unordered_map<std::string, int> g_pipe_ids;
static std::unordered_map<std::string, std::string> g_sources;  // name -> WGSL (registered from Python)

static int make_pipe(const std::string& name, const std::string& wgsl) {
  init_ctx();
  WGPUShaderSourceWGSL src = {};
  src.chain.sType = WGPUSType_ShaderSourceWGSL;
  src.code = WGPUStringView{wgsl.data(), wgsl.size()};
  WGPUShaderModuleDescriptor md = {};
  md.nextInChain = &src.chain;
  WGPUShaderModule mod = wgpuDeviceCreateShaderModule(C.device, &md);
  WGPUComputePipelineDescriptor pd = {};
  pd.compute.module = mod;
  pd.compute.entryPoint = sv("main");
  WGPUComputePipeline p = wgpuDeviceCreateComputePipeline(C.device, &pd);
  g_pipes.push_back({p, wgpuComputePipelineGetBindGroupLayout(p, 0)});
  int id = (int)g_pipes.size() - 1;
  g_pipe_ids[name] = id;
  return id;
}

static int pipe_id(const std::string& name) {
  auto it = g_pipe_ids.find(name);
  if (it != g_pipe_ids.end()) return it->second;
  auto s = g_sources.find(name);
  TORCH_CHECK(s != g_sources.end(), "omarchy-mac-torch: unknown kernel ", name);
  return make_pipe(name, s->second);
}

struct Binding {
  Buf* buf;
  uint64_t off;   // bytes
  uint64_t size;  // bytes
};

struct Dispatch {
  int pipe;
  WGPUBindGroup group;
  uint32_t x, y, z;
};

struct KeyHash {
  size_t operator()(const std::vector<uint64_t>& v) const {
    size_t h = 1469598103934665603ull;
    for (auto x : v) h = (h ^ x) * 1099511628211ull;
    return h;
  }
};

static std::unordered_map<std::vector<uint64_t>, WGPUBindGroup, KeyHash> g_groups;
static std::unordered_map<std::vector<uint32_t>, Buf*, KeyHash> g_unis_dummy;  // (unused placeholder)

struct VecU32Hash {
  size_t operator()(const std::vector<uint32_t>& v) const {
    size_t h = 1469598103934665603ull;
    for (auto x : v) h = (h ^ x) * 1099511628211ull;
    return h;
  }
};
static std::unordered_map<std::vector<uint32_t>, Buf*, VecU32Hash> g_uniforms;

static std::vector<Dispatch> g_pending;
static bool g_skip = false;  // debug: record but never submit (measures CPU-side cost)
static std::map<std::string, int64_t> g_kcount;
static bool g_time_each = false;  // debug: run each dispatch alone and time it
static std::map<std::string, double> g_ktime;
static int g_chunk = 16, g_chunk_max = 64, g_depth = 3;
static double g_target_backlog_ms = 8.0;  // keep (queued GPU work) around this, so the compositor never waits long
static double g_chunk_ms = 0.0;           // EMA of GPU time per chunk, measured while the GPU is the bottleneck
static std::chrono::steady_clock::time_point g_last_submit = std::chrono::steady_clock::now();
static std::deque<Flag*> g_inflight;

static Buf* uniform(const std::vector<uint32_t>& words) {
  auto it = g_uniforms.find(words);
  if (it != g_uniforms.end()) return it->second;
  init_ctx();
  std::vector<uint32_t> w = words;
  while (w.size() % 4) w.push_back(0);
  WGPUBufferDescriptor d = {};
  d.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
  d.size = w.size() * 4;
  Buf* b = new Buf{wgpuDeviceCreateBuffer(C.device, &d), (size_t)d.size};
  wgpuQueueWriteBuffer(C.queue, b->b, 0, w.data(), w.size() * 4);
  g_uniforms[words] = b;
  return b;
}

static void submit_pending() {
  if (g_pending.empty()) return;
  if (g_skip) { g_pending.clear(); return; }
  WGPUCommandEncoderDescriptor ed = {};
  WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(C.device, &ed);
  WGPUComputePassDescriptor pd = {};
  WGPUComputePassEncoder cp = wgpuCommandEncoderBeginComputePass(enc, &pd);
  int last = -1;
  for (auto& d : g_pending) {
    if (d.pipe != last) {
      wgpuComputePassEncoderSetPipeline(cp, g_pipes[d.pipe].p);
      last = d.pipe;
    }
    wgpuComputePassEncoderSetBindGroup(cp, 0, d.group, 0, nullptr);
    wgpuComputePassEncoderDispatchWorkgroups(cp, d.x, d.y, d.z);
  }
  wgpuComputePassEncoderEnd(cp);
  WGPUCommandBufferDescriptor cd = {};
  WGPUCommandBuffer cb = wgpuCommandEncoderFinish(enc, &cd);
  g_pending.clear();
  bool blocked = false;
  while ((int)g_inflight.size() >= g_depth) {  // wait for the oldest chunk only
    auto w0 = std::chrono::steady_clock::now();
    wait_flag(g_inflight.front());
    delete g_inflight.front();
    g_inflight.pop_front();
    if (std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - w0).count() > 0.2) blocked = true;
  }
  {  // adapt the chunk size: when we had to wait, the gap between submits is the GPU time of one chunk
    auto now = std::chrono::steady_clock::now();
    double dt = std::chrono::duration<double, std::milli>(now - g_last_submit).count();
    g_last_submit = now;
    if (blocked) {
      g_chunk_ms = g_chunk_ms == 0.0 ? dt : 0.7 * g_chunk_ms + 0.3 * dt;
      double backlog = g_chunk_ms * g_depth;
      if (backlog > g_target_backlog_ms) g_chunk = std::max(4, (int)(g_chunk * 0.7));
      else if (backlog < 0.5 * g_target_backlog_ms) g_chunk = std::min(g_chunk_max, g_chunk + 4);
    }
  }
  wgpuQueueSubmit(C.queue, 1, &cb);
  WGPUQueueWorkDoneCallbackInfo wi = {};
  wi.mode = WGPUCallbackMode_AllowSpontaneous;
  Flag* fl = new Flag();
  wi.callback = [](WGPUQueueWorkDoneStatus, WGPUStringView, void*, void* u2) { ((Flag*)u2)->done = true; };
  wi.userdata2 = fl;
  wgpuQueueOnSubmittedWorkDone(C.queue, wi);
  g_inflight.push_back(fl);
  wgpuCommandEncoderRelease(enc);
  wgpuCommandBufferRelease(cb);
}

static void sync_all() {
  submit_pending();
  while (!g_inflight.empty()) {
    wait_flag(g_inflight.front());
    delete g_inflight.front();
    g_inflight.pop_front();
  }
}

static void launch(const char* kernel, const std::vector<Binding>& b, uint32_t x, uint32_t y = 1, uint32_t z = 1) {
  int p = pipe_id(kernel);
  g_kcount[kernel]++;
  std::vector<uint64_t> key;
  key.push_back((uint64_t)p);
  for (auto& e : b) {
    key.push_back((uint64_t)(uintptr_t)e.buf);
    key.push_back(e.off);
    key.push_back(e.size);
  }
  WGPUBindGroup grp;
  auto it = g_groups.find(key);
  if (it != g_groups.end()) {
    grp = it->second;
  } else {
    std::vector<WGPUBindGroupEntry> es(b.size());
    for (size_t i = 0; i < b.size(); i++) {
      es[i] = {};
      es[i].binding = (uint32_t)i;
      es[i].buffer = b[i].buf->b;
      es[i].offset = b[i].off;
      es[i].size = b[i].size;
    }
    WGPUBindGroupDescriptor gd = {};
    gd.layout = g_pipes[p].layout;
    gd.entryCount = es.size();
    gd.entries = es.data();
    grp = wgpuDeviceCreateBindGroup(C.device, &gd);
    g_groups[key] = grp;
  }
  g_pending.push_back({p, grp, x, y, z});
  if (g_time_each) {
    submit_pending();
    sync_all();
    auto t0 = std::chrono::steady_clock::now();
    g_pending.push_back({p, grp, x, y, z});
    for (int r = 0; r < 5; r++) g_pending.push_back({p, grp, x, y, z});  // repeat: time = (6 runs) / 6, minus launch overhead
    submit_pending();
    sync_all();
    g_ktime[kernel] += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / 6.0;
    return;
  }
  if ((int)g_pending.size() >= g_chunk) submit_pending();
}

// grid for one thread per element with `wg` threads per workgroup (2-D split beyond 65535 groups)
static void grid(int64_t total, int wg, uint32_t& gx, uint32_t& gy) {
  int64_t groups = std::max<int64_t>((total + wg - 1) / wg, 1);
  gx = (uint32_t)std::min<int64_t>(groups, 65535);
  gy = (uint32_t)((groups + gx - 1) / gx);
}

// ------------------------------------------------------------------------------------------------
// Allocator and tensor helpers
// ------------------------------------------------------------------------------------------------

struct Allocator final : c10::Allocator {
  size_t factor = 1;  // bool tensors keep one 4-byte word per element on the GPU
  explicit Allocator(size_t f = 1) : factor(f) {}
  c10::DataPtr allocate(size_t n) override {
    Buf* b = alloc_buf(std::max<size_t>(n * factor, 4));
    return {(void*)b, (void*)b, &free_buf, c10::Device(kDev, 0)};
  }
  void copy_data(void*, const void*, std::size_t) const override { TORCH_CHECK(false, "omarchy-mac-torch: copy_data"); }
  c10::DeleterFnPtr raw_deleter() const override { return &free_buf; }
};
static Allocator g_alloc;
static Allocator g_alloc_bool(4);
REGISTER_ALLOCATOR(c10::DeviceType::PrivateUse1, &g_alloc);

static Buf* bufof(const Tensor& t) { return (Buf*)t.storage().data_ptr().get_context(); }
static uint64_t offb(const Tensor& t) { return (uint64_t)t.storage_offset() * t.element_size(); }
static uint64_t off_words(const Tensor& t) { return (uint64_t)t.storage_offset(); }
static bool is_dev(const Tensor& t) { return t.device().type() == kDev; }
static Binding bind(const Tensor& t) { return {bufof(t), 0, bufof(t)->bytes}; }  // whole buffer; offsets go in uniforms
static Binding bind_buf(Buf* b) { return {b, 0, b->bytes}; }

static Tensor empty_dev(IntArrayRef size, c10::ScalarType dt) {
  c10::DispatchKeySet ks(c10::DispatchKey::PrivateUse1);
  return at::detail::empty_generic(size, dt == at::kBool ? (c10::Allocator*)&g_alloc_bool : (c10::Allocator*)&g_alloc, ks, dt, std::nullopt);
}

static bool f32(const Tensor& t) { return is_dev(t) && t.scalar_type() == at::kFloat; }
static uint32_t u(int64_t v) { return (uint32_t)v; }
static uint32_t fbits(float f) {
  uint32_t x;
  std::memcpy(&x, &f, 4);
  return x;
}

static Tensor download(const Tensor& t);
static void upload(const Tensor& cpu, Tensor& dst);
static Tensor to_host(const Tensor& t) { return is_dev(t) ? download(t) : t; }
static std::optional<Tensor> to_host(const std::optional<Tensor>& t) {
  return (t && t->defined()) ? std::optional<Tensor>(to_host(*t)) : std::nullopt;
}
static Tensor to_dev(const Tensor& cpu) {
  Tensor out = empty_dev(cpu.sizes(), cpu.scalar_type());
  upload(cpu, out);
  return out;
}
static Tensor contig(const Tensor& t);
static Tensor base0(const Tensor& t) {  // contiguous with storage offset 0 (kernels that can't take offsets)
  Tensor c = contig(t);
  if (c.storage_offset() != 0) {
    Tensor o = empty_dev(c.sizes(), c.scalar_type());
    o.copy_(c);
    return o;
  }
  return c;
}

// ------------------------------------------------------------------------------------------------
// Kernel sources (registered from Python at import)
// ------------------------------------------------------------------------------------------------

static std::map<std::string, int64_t> g_fallbacks;
static void register_source(const std::string& name, const std::string& wgsl) { g_sources[name] = wgsl; }

// ------------------------------------------------------------------------------------------------
// Elementwise (broadcasting) kernel: one kernel "ew" (source from Python), uniform layout as in ops.py
// ------------------------------------------------------------------------------------------------

static std::vector<int64_t> bstrides(const Tensor& t, IntArrayRef shape) {
  int nd = (int)shape.size();
  std::vector<int64_t> st(4, 0);
  for (int i = 0; i < t.dim(); i++) {
    int di = nd - t.dim() + i;
    st[4 - nd + di] = t.size(i) > 1 ? t.stride(i) : 0;
  }
  for (int i = 0; i < nd; i++)
    if (shape[i] == 1) st[4 - nd + i] = 0;
  return st;
}

static void launch_ew(int op, const Tensor* a, const Tensor* b, Tensor& out, IntArrayRef shape, float alpha = 1.f,
                      float sval = 0.f, int64_t oo = 0) {
  int64_t n = 1;
  for (auto s : shape) n *= s;
  if (n == 0) return;
  TORCH_CHECK(shape.size() <= 4, "omarchy-mac-torch: rank > 4");
  std::vector<int64_t> sh(4, 1);
  for (size_t i = 0; i < shape.size(); i++) sh[4 - shape.size() + i] = shape[i];
  std::vector<int64_t> sa = a ? bstrides(*a, shape) : std::vector<int64_t>(4, 0);
  std::vector<int64_t> sb = b ? bstrides(*b, shape) : std::vector<int64_t>(4, 0);
  std::vector<uint32_t> w = {u(n), u(op), (uint32_t)(b ? 0 : 1), fbits(alpha), fbits(sval), 0, 0, 0};
  for (auto x : sh) w.push_back(u(x));
  for (auto x : sa) w.push_back(u(x));
  for (auto x : sb) w.push_back(u(x));
  w.push_back(a ? u(off_words(*a)) : 0);
  w.push_back(b ? u(off_words(*b)) : 0);
  w.push_back(u(oo));
  w.push_back(0);
  Buf* ub = uniform(w);
  g_kcount["ew#" + std::to_string(op) + ":" + std::to_string(n)]++;
  const Tensor& abuf = a ? *a : out;
  uint32_t gx, gy;
  grid(n, 256, gx, gy);
  Buf* dummy_a = a ? bufof(abuf) : nullptr;
  static Buf* dummy = nullptr;
  if (!dummy) dummy = alloc_buf(256);
  launch("ew", {bind_buf(a ? dummy_a : dummy), bind_buf(b ? bufof(*b) : (a ? dummy_a : dummy)), bind(out), bind_buf(ub)}, gx, gy);
}

static Tensor contig(const Tensor& t) {
  if (t.is_contiguous()) return t;
  Tensor out = empty_dev(t.sizes(), t.scalar_type());
  TORCH_CHECK(t.scalar_type() == at::kFloat && t.dim() <= 4, "omarchy-mac-torch: non-contiguous non-float copy");
  launch_ew(9, &t, nullptr, out, t.sizes());
  return out;
}

static Tensor ew_scalar(int op, const Tensor& a, double s, float alpha = 1.f);
static Tensor ew_binary(int op, const Tensor& a, const Tensor& b, float alpha = 1.f) {
  if (!is_dev(b) && b.dim() == 0) return ew_scalar(op, a, b.item<double>(), alpha);  // wrapped python number
  if (!is_dev(a) && a.dim() == 0 && (op == 0 || op == 2)) return ew_scalar(op, b, a.item<double>(), alpha);
  auto shape = at::infer_size(a.sizes(), b.sizes());
  Tensor out = empty_dev(shape, at::kFloat);
  launch_ew(op, &a, &b, out, shape, alpha);
  return out;
}

static Tensor ew_scalar(int op, const Tensor& a, double s, float alpha) {
  Tensor out = empty_dev(a.sizes(), at::kFloat);
  launch_ew(op, &a, nullptr, out, a.sizes(), alpha, (float)s);
  return out;
}

static Tensor ew_unary(int op, const Tensor& a, float alpha = 1.f) {
  Tensor out = empty_dev(a.sizes(), at::kFloat);
  launch_ew(op, &a, nullptr, out, a.sizes(), alpha);
  return out;
}

// scalar operand that may be a CPU 0-dim tensor or an device tensor: returns true if device tensor
static Tensor add_t(const Tensor& a, const Tensor& b, const Scalar& alpha) {
  return ew_binary(0, a, b, alpha.to<float>());
}
static Tensor add_s(const Tensor& a, const Scalar& b, const Scalar& alpha) { return ew_scalar(0, a, b.to<double>(), alpha.to<float>()); }
static Tensor sub_t(const Tensor& a, const Tensor& b, const Scalar& alpha) { return ew_binary(1, a, b, alpha.to<float>()); }
static Tensor sub_s(const Tensor& a, const Scalar& b, const Scalar& alpha) { return ew_scalar(1, a, b.to<double>(), alpha.to<float>()); }
static Tensor mul_t(const Tensor& a, const Tensor& b) {
  if (!is_dev(b) && b.dim() == 0) return ew_scalar(2, a, b.item<double>());
  if (!is_dev(a) && a.dim() == 0) return ew_scalar(2, b, a.item<double>());
  return ew_binary(2, a, b);
}
static Tensor mul_s(const Tensor& a, const Scalar& b) { return ew_scalar(2, a, b.to<double>()); }
static Tensor div_t(const Tensor& a, const Tensor& b) {
  if (!is_dev(b) && b.dim() == 0) return ew_scalar(3, a, b.item<double>());
  return ew_binary(3, a, b);
}
static Tensor div_s(const Tensor& a, const Scalar& b) { return ew_scalar(3, a, b.to<double>()); }

#define UNARY(NAME, CODE) static Tensor NAME(const Tensor& a) { return ew_unary(CODE, a); }
UNARY(neg_, 4)
UNARY(relu_, 5)
UNARY(exp_, 6)
UNARY(sqrt_, 7)
UNARY(sigmoid_, 13)
UNARY(silu_, 14)
UNARY(hardswish_, 15)
UNARY(hardsigmoid_, 19)
UNARY(tanh_, 21)

static Tensor bwd_ew(int op, const Tensor& g, const Tensor& y) {
  Tensor out = empty_dev(g.sizes(), at::kFloat);
  launch_ew(op, &g, &y, out, g.sizes());
  return out;
}
static Tensor sigmoid_bwd(const Tensor& g, const Tensor& y) { return bwd_ew(16, g, y); }
static Tensor silu_bwd(const Tensor& g, const Tensor& x) { return bwd_ew(17, g, x); }
static Tensor hardswish_bwd(const Tensor& g, const Tensor& x) { return bwd_ew(18, g, x); }
static Tensor hardsigmoid_bwd(const Tensor& g, const Tensor& x) { return bwd_ew(20, g, x); }
static Tensor tanh_bwd(const Tensor& g, const Tensor& y) { return bwd_ew(22, g, y); }
static Tensor threshold_bwd(const Tensor& g, const Tensor& x, const Scalar& th) {
  Tensor out = empty_dev(g.sizes(), at::kFloat);
  launch_ew(8, &g, &x, out, g.sizes(), th.to<float>());
  return out;
}
static Tensor xlogy_t(const Tensor& a, const Tensor& b) { return ew_binary(12, a, b); }

// in-place variants: compute out-of-place then copy back
static Tensor& copy_into(Tensor& self, const Tensor& src) {
  TORCH_CHECK(self.is_contiguous() && f32(self), "omarchy-mac-torch: in-place on non-contiguous / non-float");
  Tensor s = src;
  launch_ew(9, &s, nullptr, self, self.sizes(), 1.f, 0.f, self.storage_offset());
  return self;
}
static Tensor& add__t(Tensor& self, const Tensor& o, const Scalar& alpha) { Tensor r = add_t(self, o, alpha); return copy_into(self, r); }
static Tensor& add__s(Tensor& self, const Scalar& o, const Scalar& alpha) { Tensor r = add_s(self, o, alpha); return copy_into(self, r); }
static Tensor& sub__t(Tensor& self, const Tensor& o, const Scalar& alpha) { Tensor r = sub_t(self, o, alpha); return copy_into(self, r); }
static Tensor& mul__t(Tensor& self, const Tensor& o) { Tensor r = mul_t(self, o); return copy_into(self, r); }
static Tensor& mul__s(Tensor& self, const Scalar& o) { Tensor r = mul_s(self, o); return copy_into(self, r); }
static Tensor& div__t(Tensor& self, const Tensor& o) { Tensor r = div_t(self, o); return copy_into(self, r); }
static Tensor& div__s(Tensor& self, const Scalar& o) { Tensor r = div_s(self, o); return copy_into(self, r); }
static Tensor& omt_zero_(Tensor& self) {
  launch_ew(10, nullptr, nullptr, self, self.sizes(), 1.f, 0.f, self.storage_offset());
  return self;
}
static Tensor& fill_s(Tensor& self, const Scalar& v) {
  launch_ew(10, nullptr, nullptr, self, self.sizes(), 1.f, v.to<float>(), self.storage_offset());
  return self;
}

// ------------------------------------------------------------------------------------------------
// Memory / copies
// ------------------------------------------------------------------------------------------------

static size_t wire_bytes(const Tensor& t) {  // bytes per element as stored on the GPU
  auto d = t.scalar_type();
  return (d == at::kLong || d == at::kBool || d == at::kInt || d == at::kFloat) ? 4 : t.element_size();
}

static void upload(const Tensor& cpu, Tensor& dst) {  // dst contiguous, same dtype word size
  Tensor c = cpu.contiguous();
  Tensor conv = c;
  if (c.scalar_type() == at::kLong) conv = c.to(at::kInt);
  if (c.scalar_type() == at::kBool) conv = c.to(at::kInt);
  sync_all();  // earlier recorded work may still read the old contents
  size_t eb = wire_bytes(dst);
  size_t nb = conv.numel() * eb;
  size_t padded = (nb + 3) / 4 * 4;
  std::vector<char> tmp(padded, 0);
  std::memcpy(tmp.data(), conv.data_ptr(), nb);
  wgpuQueueWriteBuffer(C.queue, bufof(dst)->b, dst.storage_offset() * eb, tmp.data(), padded);
}

static Tensor download(const Tensor& t) {
  Tensor c = contig(t);
  sync_all();
  size_t eb = wire_bytes(c);
  size_t nbytes = c.numel() * eb;
  static Buf* stage = nullptr;
  static size_t stage_bytes = 0;
  size_t need = std::max<size_t>(nbytes + 8, 4);
  if (!stage || stage_bytes < need) {
    WGPUBufferDescriptor d = {};
    d.usage = WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst;
    d.size = size_class(need);
    stage = new Buf{wgpuDeviceCreateBuffer(C.device, &d), (size_t)d.size};
    stage_bytes = d.size;
  }
  WGPUCommandEncoderDescriptor ed = {};
  WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(C.device, &ed);
  wgpuCommandEncoderCopyBufferToBuffer(enc, bufof(c)->b, c.storage_offset() * eb / 4 * 4, stage->b, 0, ((c.storage_offset() * eb % 4) + nbytes + 3) / 4 * 4);
  WGPUCommandBufferDescriptor cd = {};
  WGPUCommandBuffer cb = wgpuCommandEncoderFinish(enc, &cd);
  wgpuQueueSubmit(C.queue, 1, &cb);
  WGPUBufferMapCallbackInfo mi = {};
  mi.mode = WGPUCallbackMode_AllowSpontaneous;
  Flag mf;
  mi.callback = [](WGPUMapAsyncStatus, WGPUStringView, void*, void* u2) { ((Flag*)u2)->done = true; };
  mi.userdata2 = &mf;
  wgpuBufferMapAsync(stage->b, WGPUMapMode_Read, 0, ((c.storage_offset() * eb % 4) + nbytes + 3) / 4 * 4, mi);
  wait_flag(&mf);
  const void* p = wgpuBufferGetConstMappedRange(stage->b, 0, ((c.storage_offset() * eb % 4) + nbytes + 3) / 4 * 4);
  Tensor out;
  if (c.scalar_type() != at::kLong && c.scalar_type() != at::kBool && c.scalar_type() != at::kInt) {
    out = at::empty(c.sizes(), c.scalar_type());
    std::memcpy(out.data_ptr(), (const char*)p + (c.storage_offset() * eb % 4), nbytes);
  } else {
    Tensor tmp = at::empty(c.sizes(), at::kInt);
    std::memcpy(tmp.data_ptr(), p, nbytes);
    out = (c.scalar_type() == at::kBool) ? tmp.ne(0) : tmp.to(c.scalar_type());
  }
  wgpuBufferUnmap(stage->b);
  wgpuCommandEncoderRelease(enc);
  wgpuCommandBufferRelease(cb);
  return out;
}

static Tensor empty_memory_format(IntArrayRef size, std::optional<at::ScalarType> dtype, std::optional<at::Layout>,
                                  std::optional<at::Device>, std::optional<bool>, std::optional<at::MemoryFormat>) {
  return empty_dev(size, dtype.value_or(at::kFloat));
}
static Tensor empty_strided(IntArrayRef size, IntArrayRef stride, std::optional<at::ScalarType> dtype,
                            std::optional<at::Layout>, std::optional<at::Device>, std::optional<bool>) {
  c10::DispatchKeySet ks(c10::DispatchKey::PrivateUse1);
  return at::detail::empty_strided_generic(size, stride, &g_alloc, ks, dtype.value_or(at::kFloat));
}
static Tensor empty_like_(const Tensor& self, std::optional<at::ScalarType> dtype, std::optional<at::Layout>,
                          std::optional<at::Device>, std::optional<bool>, std::optional<at::MemoryFormat>) {
  return empty_dev(self.sizes(), dtype.value_or(self.scalar_type()));
}

static Tensor _copy_from_(const Tensor& self, const Tensor& dst, bool) {  // copy self -> dst
  if (!is_dev(self) && is_dev(dst)) {
    Tensor s = self;
    if (dst.is_contiguous() && s.sizes() == dst.sizes()) {
      Tensor d = dst;
      upload(s, d);
    } else {
      Tensor tmp = empty_dev(s.sizes(), dst.scalar_type());
      upload(s.to(dst.scalar_type()), tmp);
      Tensor dd = dst;
      Tensor b = tmp.expand(dd.sizes());
      launch_ew(9, &b, nullptr, dd, dd.sizes(), 1.f, 0.f, dd.storage_offset());
    }
  } else if (is_dev(self) && !is_dev(dst)) {
    Tensor cpu = download(self);
    const_cast<Tensor&>(dst).copy_(cpu);
  } else {  // device -> device
    Tensor d = dst;
    if (f32(self) && f32(d) && d.is_contiguous()) {
      Tensor src = self.sizes() == d.sizes() ? self : self.expand(d.sizes());
      launch_ew(9, &src, nullptr, d, d.sizes(), 1.f, 0.f, d.storage_offset());
    } else {  // integer / byte tensors: through the host (rare: storage clones, state dict copies)
      Tensor cpu = download(self);
      Tensor conv = cpu.to(d.scalar_type());
      upload(conv.expand(d.sizes()).contiguous(), d);
    }
  }
  return dst;
}
static Tensor _copy_from_and_resize_(const Tensor& self, const Tensor& dst) {
  const_cast<Tensor&>(dst).resize_(self.sizes());
  return _copy_from_(self, dst, false);
}
static Tensor& copy_(Tensor& self, const Tensor& src, bool nb) {
  _copy_from_(src, self, nb);
  return self;
}
static Tensor _to_copy_(const Tensor& self, std::optional<at::ScalarType> dtype, std::optional<at::Layout>,
                        std::optional<at::Device> device, std::optional<bool>, bool, std::optional<at::MemoryFormat>) {
  at::ScalarType dt = dtype.value_or(self.scalar_type());
  if (!is_dev(self)) {  // CPU -> device (this is how .to("omarchy_mac_torch") arrives)
    Tensor out = empty_dev(self.sizes(), dt);
    upload(self.to(dt), out);
    return out;
  }
  if (device && device->type() == c10::DeviceType::CPU) {
    Tensor cpu = download(self);
    return cpu.to(dt);
  }
  Tensor out = empty_dev(self.sizes(), dt);
  if (dt == self.scalar_type() && f32(self)) {
    Tensor d = out;
    Tensor s = self;
    launch_ew(9, &s, nullptr, d, d.sizes());
  } else {
    Tensor cpu = download(self).to(dt);
    upload(cpu, out);
  }
  return out;
}
static Tensor clone_(const Tensor& self, std::optional<at::MemoryFormat>) {
  if (!f32(self)) return _to_copy_(self, std::nullopt, std::nullopt, std::nullopt, std::nullopt, false, std::nullopt);
  Tensor out = empty_dev(self.sizes(), at::kFloat);
  launch_ew(9, &self, nullptr, out, self.sizes());
  return out;
}
static c10::Scalar _local_scalar_dense_(const Tensor& self) { return download(self).item(); }
static Tensor ones_like_(const Tensor& self, std::optional<at::ScalarType> dtype, std::optional<at::Layout>,
                         std::optional<at::Device>, std::optional<bool>, std::optional<at::MemoryFormat>) {
  Tensor out = empty_dev(self.sizes(), dtype.value_or(self.scalar_type()));
  return fill_s(out, 1);
}
static Tensor zeros_like_(const Tensor& self, std::optional<at::ScalarType> dtype, std::optional<at::Layout>,
                          std::optional<at::Device>, std::optional<bool>, std::optional<at::MemoryFormat>) {
  Tensor out = empty_dev(self.sizes(), dtype.value_or(self.scalar_type()));
  return omt_zero_(out);
}
static Tensor& set_storage_(Tensor& self, c10::Storage src, int64_t off, IntArrayRef size, IntArrayRef stride) {
  at::native::set_storage_cpu_(self, src, off, size, stride);  // metadata-only
  return self;
}

// ------------------------------------------------------------------------------------------------
// Reductions, matmul
// ------------------------------------------------------------------------------------------------

static Tensor reduce_dims(const Tensor& x, IntArrayRef dims_in, bool keepdim, bool mean) {
  TORCH_CHECK(f32(x), "omarchy-mac-torch: reduce float only");
  int nd = (int)x.dim();
  std::vector<int64_t> dims;
  if (dims_in.empty()) for (int i = 0; i < nd; i++) dims.push_back(i);
  else for (auto d : dims_in) dims.push_back(d < 0 ? d + nd : d);
  std::sort(dims.begin(), dims.end());
  if (dims.empty()) return clone_(x, std::nullopt);
  for (size_t i = 1; i < dims.size(); i++) TORCH_CHECK(dims[i] == dims[i - 1] + 1, "omarchy-mac-torch: non-contiguous reduce dims");
  Tensor xc = contig(x);
  int64_t outer = 1, red = 1, inner = 1;
  for (int i = 0; i < dims.front(); i++) outer *= x.size(i);
  for (int64_t i = dims.front(); i <= dims.back(); i++) red *= x.size(i);
  for (int i = (int)dims.back() + 1; i < nd; i++) inner *= x.size(i);
  std::vector<int64_t> os;
  for (int i = 0; i < nd; i++) {
    bool r = std::find(dims.begin(), dims.end(), i) != dims.end();
    if (r) { if (keepdim) os.push_back(1); } else os.push_back(x.size(i));
  }
  Tensor out = empty_dev(os, at::kFloat);
  Buf* ub = uniform({u(outer), u(red), u(inner), u(off_words(xc)), 0, fbits(mean ? 1.f / red : 1.f), 0, 0});
  uint32_t gx, gy;
  grid(outer * inner, 64, gx, gy);
  launch("reduce", {bind(xc), bind(out), bind_buf(ub)}, gx, gy);
  return out;
}
static Tensor sum_default(const Tensor& x, std::optional<at::ScalarType> dt) {
  if (x.scalar_type() == at::kBool || x.scalar_type() == at::kLong || x.scalar_type() == at::kInt) {
    Tensor xc = contig(x);
    Tensor out = empty_dev({}, at::kLong);
    Buf* ub = uniform({u(xc.numel()), u(off_words(xc)), 0, 0});
    launch("isum", {bind(xc), bind(out), bind_buf(ub)}, 1);
    return out;
  }
  return reduce_dims(x, {}, false, false);
}
static Tensor sum_dim(const Tensor& x, at::OptionalIntArrayRef dim, bool keepdim, std::optional<at::ScalarType>) {
  return reduce_dims(x, dim.value_or(IntArrayRef{}), keepdim, false);
}
static Tensor mean_dim(const Tensor& x, at::OptionalIntArrayRef dim, bool keepdim, std::optional<at::ScalarType>) {
  return reduce_dims(x, dim.value_or(IntArrayRef{}), keepdim, true);
}
static Tensor mean_default(const Tensor& x, std::optional<at::ScalarType>) { return reduce_dims(x, {}, false, true); }

static int64_t cdiv(int64_t a, int64_t b);
static py::function g_mm_planner;  // (a_k_fast, b_n_fast, tile) -> (kernel, bm, bn)
struct MmKernel { std::string name; int bm, bn; };
static std::map<std::tuple<bool, bool, int>, MmKernel> g_mm_kernels;
static MmKernel mm_kernel(bool a_kfast, bool b_nfast, int tile) {
  auto key = std::make_tuple(a_kfast, b_nfast, tile);
  auto it = g_mm_kernels.find(key);
  if (it != g_mm_kernels.end()) return it->second;
  py::gil_scoped_acquire gil;
  py::tuple r = g_mm_planner(a_kfast, b_nfast, tile);
  MmKernel k{r[0].cast<std::string>(), r[1].cast<int>(), r[2].cast<int>()};
  g_mm_kernels[key] = k;
  return k;
}

static Tensor mm_impl(const Tensor& a, const Tensor& b, const Tensor* bias, Tensor* outp = nullptr, int64_t co = 0) {
  TORCH_CHECK(f32(a) && f32(b) && a.dim() == 2 && b.dim() == 2, "omarchy-mac-torch: mm float 2-D");
  int64_t M = a.size(0), K = a.size(1), N = b.size(1);
  Tensor out = outp ? *outp : empty_dev({M, N}, at::kFloat);
  const Tensor& bt = bias ? *bias : a;
  // tile: skinny problems (a batch of rows against a big weight matrix) want a tall-narrow-free tile and split-K
  int tile = M <= 32 ? 2 : (N <= 32 ? 1 : (M >= 256 && N >= 256 ? 5 : 0));
  MmKernel kn = mm_kernel(a.stride(1) == 1, b.stride(1) == 1, tile);
  int64_t tiles = cdiv(M, kn.bm) * cdiv(N, kn.bn);
  int64_t slices = std::max<int64_t>(1, std::min<int64_t>(192 / tiles, K / 128));
  int64_t kslice = cdiv(cdiv(K, slices), 16) * 16;
  slices = cdiv(K, kslice);
  uint32_t hb = bias ? 1 : 0;
  if (slices == 1) {
    Buf* pr = uniform({u(M), u(N), u(K), hb, u(off_words(a)), u(off_words(b)), u(co), bias ? u(off_words(*bias)) : 0u,
                       u(a.stride(0)), u(a.stride(1)), u(b.stride(0)), u(b.stride(1)), u(kslice), 1, 0, 0});
    launch(kn.name.c_str(), {bind(a), bind(b), bind(out), bind_buf(pr), bind(bt)}, u(cdiv(N, kn.bn)), u(cdiv(M, kn.bm)), 1);
  } else {
    Tensor part = empty_dev({slices * M * N}, at::kFloat);
    Buf* pr = uniform({u(M), u(N), u(K), hb, u(off_words(a)), u(off_words(b)), 0, bias ? u(off_words(*bias)) : 0u,
                       u(a.stride(0)), u(a.stride(1)), u(b.stride(0)), u(b.stride(1)), u(kslice), u(slices), 0, 0});
    launch(kn.name.c_str(), {bind(a), bind(b), bind(part), bind_buf(pr), bind(bt)}, u(cdiv(N, kn.bn)), u(cdiv(M, kn.bm)), u(slices));
    Buf* rr = uniform({u(M), u(N), 0, hb, 0, 0, u(co), bias ? u(off_words(*bias)) : 0u, 0, 0, 0, 0, 0, u(slices), 0, 0});
    uint32_t gx, gy;
    grid(M * N, 256, gx, gy);
    launch("mm_reduce", {bind(part), bind(out), bind(bt), bind_buf(rr)}, gx, gy);
  }
  return out;
}
static Tensor mm_(const Tensor& a, const Tensor& b) { return mm_impl(a, b, nullptr); }
static Tensor addmm_(const Tensor& bias, const Tensor& a, const Tensor& b, const Scalar& beta, const Scalar& alpha) {
  TORCH_CHECK(beta.to<double>() == 1 && alpha.to<double>() == 1 && bias.dim() == 1 && bias.is_contiguous(), "omarchy-mac-torch: addmm");
  return mm_impl(a, b, &bias);
}

// ------------------------------------------------------------------------------------------------
// Convolution (GEMM kernels generated in Python: tile choice via callback), batch norm, pooling, softmax
// ------------------------------------------------------------------------------------------------

static py::function g_conv_planner;  // (kind, n, ci, co, ih, iw, oh, ow, ks, stride, pad, tile) -> (kernel, bm, bn)

struct Geo {  // a convolution: input n x ci x ih x iw, kernel ks x ks, output n x co x oh x ow
  int64_t n, ci, co, ih, iw, oh, ow, ks, stride, pad;
};
static Geo same_geo(int64_t n, int64_t ci, int64_t co, int64_t h, int64_t w, int64_t ks) { return {n, ci, co, h, w, h, w, ks, 1, ks / 2}; }
static Geo conv_geo(int64_t n, int64_t ci, int64_t co, int64_t ih, int64_t iw, int64_t ks, int64_t stride, int64_t pad) {
  return {n, ci, co, ih, iw, (ih + 2 * pad - ks) / stride + 1, (iw + 2 * pad - ks) / stride + 1, ks, stride, pad};
}
static void mnk(int kind, const Geo& g, int64_t& M, int64_t& N, int64_t& K) {
  if (kind == 0 || kind == 3) { M = g.n * g.oh * g.ow; N = g.co; K = g.ci * g.ks * g.ks; }
  else if (kind == 1) { M = g.n * g.ih * g.iw; N = g.ci; K = g.co * g.ks * g.ks; }
  else { M = g.co; N = g.ci * g.ks * g.ks; K = g.n * g.oh * g.ow; }
}

struct ConvPlan {
  std::string kernel;
  int bm = 64, bn = 64;
  int tile = -1;
  double mult = 1.0;  // split-K multiplier for the weight gradient
};
static std::map<std::vector<int64_t>, ConvPlan> g_conv_cache;
static const ConvPlan* g_force = nullptr;
static bool g_tune = true, g_in_tune = false;
static int g_fold = 1;  // BN folding in eval plans: 0 none, 1 only into 1x1 convs, 2 into every conv / depthwise
static int g_num_tiles = 5;

static ConvPlan plan_for(int kind, const Geo& g, int tile) {
  py::gil_scoped_acquire gil;
  py::tuple r = g_conv_planner(kind, g.n, g.ci, g.co, g.ih, g.iw, g.oh, g.ow, g.ks, g.stride, g.pad, tile);
  ConvPlan p;
  p.kernel = r[0].cast<std::string>();
  p.bm = r[1].cast<int>();
  p.bn = r[2].cast<int>();
  p.tile = tile;
  return p;
}

static ConvPlan autotune(int kind, const Geo& g);

static std::vector<int64_t> geo_key(int kind, const Geo& g) {
  return {kind, g.n, g.ci, g.co, g.ih, g.iw, g.oh, g.ow, g.ks, g.stride, g.pad};
}

static ConvPlan conv_plan(int kind, const Geo& g) {
  if (g_force) return *g_force;
  auto key = geo_key(kind, g);
  auto it = g_conv_cache.find(key);
  if (it != g_conv_cache.end()) return it->second;
  ConvPlan p;
  if (kind == 3) {  // forward with a BatchNorm-folding loader: same tile as the plain forward conv
    ConvPlan base = conv_plan(0, g);
    p = plan_for(3, g, base.tile >= 0 ? base.tile : -1);
  } else {
    p = plan_for(kind, g, -1);
    if (g_tune && !g_in_tune) p = autotune(kind, g);
  }
  g_conv_cache[key] = p;
  return p;
}

static Buf* pbuf_geo(const Geo& g, int64_t M, int64_t N, int64_t K, int64_t kslice, int64_t slices) {
  return uniform({u(g.n), u(g.ci), u(g.co), u(g.ih), u(g.iw), u(g.oh), u(g.ow), u(g.ks), u(g.stride), u(g.pad), u(M), u(N), u(K),
                  u(kslice), u(slices), 0});
}
static Buf* pbuf_reduce(int64_t M, int64_t N, int64_t slices) {  // the 12-word struct the split-K reduce kernel reads
  return uniform({0, 0, 0, 0, 0, u(M), u(N), 0, 0, u(slices), 0, 0});
}
static int64_t cdiv(int64_t a, int64_t b) { return (a + b - 1) / b; }

static void gemm_forward(const Tensor& x, const Tensor& w, Tensor& out, const Geo& g) {
  int64_t M, N, K;
  mnk(0, g, M, N, K);
  ConvPlan p = conv_plan(0, g);
  Buf* pr = pbuf_geo(g, M, N, K, cdiv(K, 16) * 16, 1);
  launch(p.kernel.c_str(), {bind(x), bind(w), bind(out), bind_buf(pr)}, u(cdiv(N, p.bn)), u(cdiv(M, p.bm)), 1);
}
static void gemm_grad_input(const Tensor& gy, const Tensor& w, Tensor& out, const Geo& g) {
  int64_t M, N, K;
  mnk(1, g, M, N, K);
  ConvPlan p = conv_plan(1, g);
  Buf* pr = pbuf_geo(g, M, N, K, cdiv(K, 16) * 16, 1);
  launch(p.kernel.c_str(), {bind(gy), bind(w), bind(out), bind_buf(pr)}, u(cdiv(N, p.bn)), u(cdiv(M, p.bm)), 1);
}
static void gemm_grad_weight(const Tensor& x, const Tensor& gy, Tensor& out, const Geo& g) {
  int64_t M, N, K;
  mnk(2, g, M, N, K);
  ConvPlan p = conv_plan(2, g);
  int64_t tiles = cdiv(M, p.bm) * cdiv(N, p.bn);
  int64_t slices = std::max<int64_t>(1, std::min<int64_t>((int64_t)(512 / tiles * p.mult), K / 64));
  int64_t kslice = cdiv(cdiv(K, slices), 16) * 16;
  slices = cdiv(K, kslice);
  Tensor part = empty_dev({slices * M * N}, at::kFloat);
  Buf* pr = pbuf_geo(g, M, N, K, kslice, slices);
  launch(p.kernel.c_str(), {bind(gy), bind(x), bind(part), bind_buf(pr)}, u(cdiv(N, p.bn)), u(cdiv(M, p.bm)), u(slices));
  launch("gemm_reduce", {bind(part), bind(out), bind_buf(pbuf_reduce(M, N, slices))}, u(cdiv(M * N, 256)));
}

static Buf* qbuf(int64_t n, int64_t c, int64_t h, int64_t w, int64_t oh = 0, int64_t ow = 0, int64_t p0 = 0, int64_t p1 = 0) {
  return uniform({u(n), u(c), u(h), u(w), u(oh), u(ow), u(p0), u(p1)});
}
static void ew_launch(const char* k, const std::vector<Binding>& b, int64_t total) {
  uint32_t gx, gy;
  grid(total, 256, gx, gy);
  launch(k, b, gx, gy);
}

static std::string conv_kind(const Tensor& x, const Tensor& w, bool has_bias, IntArrayRef stride, IntArrayRef padding,
                             IntArrayRef dilation, bool transposed, int64_t groups) {
  if (has_bias || transposed || !f32(x) || !f32(w) || x.dim() != 4 || w.dim() != 4) return "";
  if (dilation[0] != 1 || dilation[1] != 1 || w.size(2) != w.size(3)) return "";
  if (stride[0] != stride[1] || padding[0] != padding[1]) return "";
  int64_t ks = w.size(2), s = stride[0], p = padding[0];
  if (groups == 1 && ks <= 11 && s >= 1 && s <= 4 && p <= ks) return "dense";  // any stride 1-4 and padding
  if (groups == x.size(1) && groups == w.size(0) && groups > 1 && (ks == 3 || ks == 5 || ks == 7) && s == 1 && p == ks / 2) return "dw";
  return "";
}

static Tensor convolution_(const Tensor& x_, const Tensor& w_, const std::optional<Tensor>& bias, IntArrayRef stride,
                           IntArrayRef padding, IntArrayRef dilation, bool transposed, IntArrayRef, int64_t groups) {
  bool has_bias = bias.has_value() && bias->defined();
  std::string kind = conv_kind(x_, w_, false, stride, padding, dilation, transposed, groups);
  if (kind.empty()) {  // not a configuration the GPU kernels cover: run on the host
    g_fallbacks["host:convolution (dilation/transposed/grouped)"]++;
    Tensor r = at::convolution(to_host(x_), to_host(w_), to_host(bias), stride, padding, dilation, transposed, {0, 0}, groups);
    return to_dev(r);
  }
  Tensor x = base0(x_), w = base0(w_);
  int64_t n = x.size(0), ci = x.size(1), h = x.size(2), wd = x.size(3), co = w.size(0), ks = w.size(2);
  Geo g = conv_geo(n, ci, co, h, wd, ks, stride[0], padding[0]);
  Tensor out = empty_dev({n, co, g.oh, g.ow}, at::kFloat);
  if (kind == "dense") gemm_forward(x, w, out, g);
  else ew_launch("dw_fwd", {bind(x), bind(w), bind(out), bind_buf(qbuf(n, ci, h, wd, 0, 0, ks))}, n * ci * h * wd);
  if (has_bias) return ew_binary(0, out, bias->view({1, co, 1, 1}));
  return out;
}

static std::tuple<Tensor, Tensor, Tensor> convolution_backward_(const Tensor& gy_, const Tensor& x_, const Tensor& w_,
                                                                at::OptionalIntArrayRef, IntArrayRef stride,
                                                                IntArrayRef padding, IntArrayRef dilation,
                                                                bool transposed, IntArrayRef, int64_t groups,
                                                                std::array<bool, 3> mask) {
  std::string kind = conv_kind(x_, w_, false, stride, padding, dilation, transposed, groups);
  if (kind.empty() || (kind == "dense" && mask[0] && stride[0] != 1)) {  // input gradient of a strided conv: host
    g_fallbacks["host:convolution_backward"]++;
    auto r = at::convolution_backward(to_host(gy_), to_host(x_), to_host(w_), std::nullopt, stride, padding, dilation,
                                      transposed, {0, 0}, groups, mask);
    return {mask[0] ? to_dev(std::get<0>(r)) : Tensor(), mask[1] ? to_dev(std::get<1>(r)) : Tensor(),
            mask[2] ? to_dev(std::get<2>(r)) : Tensor()};
  }
  Tensor gy = base0(gy_), x = base0(x_), w = base0(w_);
  int64_t n = x.size(0), ci = x.size(1), h = x.size(2), wd = x.size(3), co = w.size(0), ks = w.size(2);
  int64_t oh = gy.size(2), ow = gy.size(3);
  Tensor gx, gw;
  if (kind == "dense") {
    Geo g = conv_geo(n, ci, co, h, wd, ks, stride[0], padding[0]);
    if (mask[0]) { gx = empty_dev(x.sizes(), at::kFloat); gemm_grad_input(gy, w, gx, g); }
    if (mask[1]) { gw = empty_dev(w.sizes(), at::kFloat); gemm_grad_weight(x, gy, gw, g); }
  } else {
    Buf* q = qbuf(n, ci, h, wd, 0, 0, ks);
    if (mask[0]) { gx = empty_dev(x.sizes(), at::kFloat); ew_launch("dw_gin", {bind(gy), bind(w), bind(gx), bind_buf(q)}, n * ci * h * wd); }
    if (mask[1]) { gw = empty_dev(w.sizes(), at::kFloat); launch("dw_gwt", {bind(gy), bind(x), bind(gw), bind_buf(q)}, u(ci * ks * ks)); }
  }
  Tensor gb;
  if (mask[2]) {
    gb = empty_dev({co}, at::kFloat);
    launch("csum_nchw", {bind(gy), bind(gb), bind_buf(qbuf(n, co, oh, ow))}, u(co));
  }
  return {gx, gw, gb};
}

static std::tuple<Tensor, Tensor, Tensor> batch_norm_(const Tensor& x_, const std::optional<Tensor>& weight,
                                                      const std::optional<Tensor>& bias, const std::optional<Tensor>& rm,
                                                      const std::optional<Tensor>& rv, bool training, double momentum,
                                                      double eps) {
  if (!(weight && bias && rm && rv && f32(x_) && x_.dim() == 4)) {  // 1-D/2-D batch norm, no affine, ...: host
    auto hx = to_host(x_);
    auto hrm = to_host(rm), hrv = to_host(rv);
    auto r = at::native_batch_norm(hx, to_host(weight), to_host(bias), hrm, hrv, training, momentum, eps);
    if (rm && rm->defined() && training) { Tensor d = *rm; upload(*hrm, d); Tensor d2 = *rv; upload(*hrv, d2); }
    return {to_dev(std::get<0>(r)), to_dev(std::get<1>(r)), to_dev(std::get<2>(r))};
  }
  Tensor x = base0(x_);
  int64_t n = x.size(0), c = x.size(1), h = x.size(2), w = x.size(3);
  Buf* q = qbuf(n, c, h, w);
  Tensor y = empty_dev(x.sizes(), at::kFloat);
  Tensor sm, si;
  if (training) {
    Tensor stats = empty_dev({2 * c}, at::kFloat);
    sm = empty_dev({c}, at::kFloat);
    si = empty_dev({c}, at::kFloat);
    int64_t count = n * h * w;
    launch("bn_stats", {bind(x), bind(stats), bind_buf(q)}, u(c));
    Buf* r = uniform({u(c), fbits((float)momentum), fbits((float)count / std::max<int64_t>(count - 1, 1)), fbits((float)eps), 1, 0, 0, 0});
    launch("bn_final", {bind(stats), bind(*rm), bind(*rv), bind(sm), bind(si), bind_buf(r)}, u(cdiv(c, 64)));
  } else {
    sm = *rm;
    si = empty_dev({c}, at::kFloat);
    Buf* r = uniform({u(c), fbits((float)eps), 0, 0});
    launch("bn_eval_inv", {bind(*rv), bind(si), bind_buf(r)}, u(cdiv(c, 64)));
  }
  ew_launch("bn_apply", {bind(x), bind(sm), bind(si), bind(*weight), bind(*bias), bind(y), bind_buf(q)}, n * c * h * w);
  if (!training) return {y, empty_dev({0}, at::kFloat), empty_dev({0}, at::kFloat)};
  return {y, sm, si};
}

static std::tuple<Tensor, Tensor, Tensor> batch_norm_backward_(const Tensor& gy_, const Tensor& x_,
                                                               const std::optional<Tensor>& weight,
                                                               const std::optional<Tensor>& rmn, const std::optional<Tensor>& rvn,
                                                               const std::optional<Tensor>& sm,
                                                               const std::optional<Tensor>& si, bool train, double eps,
                                                               std::array<bool, 3> mask) {
  if (!(train && weight && sm && si && f32(x_) && x_.dim() == 4)) {
    auto r = at::native_batch_norm_backward(to_host(gy_), to_host(x_), to_host(weight), to_host(rmn), to_host(rvn),
                                            to_host(sm), to_host(si), train, eps, mask);
    return {mask[0] ? to_dev(std::get<0>(r)) : Tensor(), mask[1] ? to_dev(std::get<1>(r)) : Tensor(),
            mask[2] ? to_dev(std::get<2>(r)) : Tensor()};
  }
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
  return {gx, dgb.select(0, 0), dgb.select(0, 1)};
}

static bool pool_2x2(const Tensor& x, IntArrayRef k, IntArrayRef s, IntArrayRef p, IntArrayRef d, bool ceil) {
  return f32(x) && x.dim() == 4 && k[0] == 2 && k[k.size() - 1] == 2 && (s.empty() || s[0] == 2) && p[0] == 0 && d[0] == 1 && !ceil;
}
static bool pool_general(const Tensor& x, IntArrayRef k, IntArrayRef s, IntArrayRef p, IntArrayRef d, bool ceil) {
  if (!(f32(x) && x.dim() == 4 && d[0] == 1 && d[d.size() - 1] == 1 && !ceil && k[0] == k[k.size() - 1])) return false;
  int64_t st = s.empty() ? k[0] : s[0];
  return (s.empty() || s[0] == s[s.size() - 1]) && p[0] == p[p.size() - 1] && p[0] <= k[0] / 2 && st >= 1;
}
static Buf* pool_params(const Tensor& x, int64_t oh, int64_t ow, int64_t k, int64_t st, int64_t pd) {
  return uniform({u(x.size(0)), u(x.size(1)), u(x.size(2)), u(x.size(3)), u(oh), u(ow), u(k), u(st), u(pd), 0, 0, 0});
}

static std::tuple<Tensor, Tensor> max_pool2d_(const Tensor& x_, IntArrayRef k, IntArrayRef s, IntArrayRef p, IntArrayRef d,
                                              bool ceil) {
  if (pool_2x2(x_, k, s, p, d, ceil)) {
    Tensor x = base0(x_);
    int64_t n = x.size(0), c = x.size(1), h = x.size(2), w = x.size(3);
    Tensor out = empty_dev({n, c, h / 2, w / 2}, at::kFloat);
    Tensor idx = empty_dev({n, c, h / 2, w / 2}, at::kLong);
    ew_launch("pool_fwd", {bind(x), bind(out), bind(idx), bind_buf(qbuf(n, c, h, w, h / 2, w / 2))}, n * c * (h / 2) * (w / 2));
    return {out, idx};
  }
  if (pool_general(x_, k, s, p, d, ceil)) {  // any window / stride / padding; indices in PyTorch's format
    Tensor x = base0(x_);
    int64_t kk = k[0], st = s.empty() ? k[0] : s[0], pd = p[0];
    int64_t n = x.size(0), c = x.size(1), h = x.size(2), w = x.size(3);
    int64_t oh = (h + 2 * pd - kk) / st + 1, ow = (w + 2 * pd - kk) / st + 1;
    Tensor out = empty_dev({n, c, oh, ow}, at::kFloat);
    Tensor idx = empty_dev({n, c, oh, ow}, at::kLong);
    ew_launch("maxpool_gen_fwd", {bind(x), bind(out), bind(idx), bind_buf(pool_params(x, oh, ow, kk, st, pd))}, n * c * oh * ow);
    return {out, idx};
  }
  g_fallbacks["host:max_pool2d"]++;
  auto r = at::max_pool2d_with_indices(to_host(x_), k, s, p, d, ceil);
  return {to_dev(std::get<0>(r)), to_dev(std::get<1>(r))};
}
static Tensor max_pool2d_backward_(const Tensor& gy_, const Tensor& x, IntArrayRef k, IntArrayRef s, IntArrayRef p, IntArrayRef d,
                                   bool ceil, const Tensor& idx) {
  if (pool_2x2(x, k, s, p, d, ceil)) {
    Tensor gy = base0(gy_);
    int64_t n = x.size(0), c = x.size(1), h = x.size(2), w = x.size(3);
    Tensor gx = empty_dev(x.sizes(), at::kFloat);
    ew_launch("pool_bwd", {bind(gy), bind(idx), bind(gx), bind_buf(qbuf(n, c, h, w, h / 2, w / 2))}, n * c * h * w);
    return gx;
  }
  if (pool_general(x, k, s, p, d, ceil)) {
    Tensor gy = base0(gy_);
    int64_t kk = k[0], st = s.empty() ? k[0] : s[0], pd = p[0];
    int64_t n = x.size(0), c = x.size(1), h = x.size(2), w = x.size(3);
    int64_t oh = gy.size(2), ow = gy.size(3);
    Tensor gx = empty_dev(x.sizes(), at::kFloat);
    ew_launch("maxpool_gen_bwd", {bind(gy), bind(idx), bind(gx), bind_buf(pool_params(x, oh, ow, kk, st, pd))}, n * c * h * w);
    return gx;
  }
  g_fallbacks["host:max_pool2d_backward"]++;
  return to_dev(at::max_pool2d_with_indices_backward(to_host(gy_), to_host(x), k, s, p, d, ceil, to_host(idx)));
}
static Tensor adaptive_avg_pool2d_(const Tensor& x_, IntArrayRef os) {
  if (!(f32(x_) && x_.dim() == 4)) { g_fallbacks["host:adaptive_avg_pool2d"]++; return to_dev(at::_adaptive_avg_pool2d(to_host(x_), os)); }
  Tensor x = base0(x_);
  int64_t oh = os[0], ow = os[1];
  Tensor out = empty_dev({x.size(0), x.size(1), oh, ow}, at::kFloat);
  ew_launch("adapool_fwd", {bind(x), bind(out), bind_buf(pool_params(x, oh, ow, 0, 0, 0))}, out.numel());
  return out;
}
static Tensor adaptive_avg_pool2d_backward_(const Tensor& gy_, const Tensor& x) {
  if (!(f32(x) && f32(gy_) && x.dim() == 4)) { g_fallbacks["host:adaptive_avg_pool2d_backward"]++; return to_dev(at::_adaptive_avg_pool2d_backward(to_host(gy_), to_host(x))); }
  Tensor gy = base0(gy_);
  Tensor gx = empty_dev(x.sizes(), at::kFloat);
  ew_launch("adapool_bwd", {bind(gy), bind(gx), bind_buf(pool_params(x, gy.size(2), gy.size(3), 0, 0, 0))}, gx.numel());
  return gx;
}
static Tensor& relu_inplace(Tensor& self) {
  Tensor r = ew_unary(5, self);
  return copy_into(self, r);
}

static Tensor avg_pool2d_(const Tensor& x_, IntArrayRef, IntArrayRef, IntArrayRef, bool, bool, std::optional<int64_t>) {
  Tensor x = base0(x_);
  int64_t n = x.size(0), c = x.size(1), h = x.size(2), w = x.size(3);
  Tensor out = empty_dev({n, c, h / 2, w / 2}, at::kFloat);
  ew_launch("avgpool_fwd", {bind(x), bind(out), bind_buf(qbuf(n, c, h, w, h / 2, w / 2))}, n * c * (h / 2) * (w / 2));
  return out;
}
static Tensor avg_pool2d_backward_(const Tensor& gy_, const Tensor& x, IntArrayRef, IntArrayRef, IntArrayRef, bool, bool,
                                   std::optional<int64_t>) {
  Tensor gy = base0(gy_);
  int64_t n = x.size(0), c = x.size(1), h = x.size(2), w = x.size(3);
  Tensor gx = empty_dev(x.sizes(), at::kFloat);
  ew_launch("avgpool_bwd", {bind(gy), bind(gx), bind_buf(qbuf(n, c, h, w, h / 2, w / 2))}, n * c * h * w);
  return gx;
}

static Tensor softmax_impl(const Tensor& x_, int64_t mode) {
  Tensor x = base0(x_);
  int64_t n = x.size(0), c = x.size(1);
  Tensor out = empty_dev({n, c}, at::kFloat);
  launch("smax", {bind(x), bind(out), bind_buf(uniform({u(n), u(c), u(mode), 0}))}, u(cdiv(n, 64)));
  return out;
}
static Tensor softmax_(const Tensor& x, int64_t dim, bool) { TORCH_CHECK(x.dim() == 2 && (dim == 1 || dim == -1)); return softmax_impl(x, 0); }
static Tensor log_softmax_(const Tensor& x, int64_t dim, bool) { TORCH_CHECK(x.dim() == 2 && (dim == 1 || dim == -1)); return softmax_impl(x, 1); }
static Tensor smax_bwd_impl(const Tensor& g_, const Tensor& y_, int64_t mode) {
  Tensor g = base0(g_), y = base0(y_);
  int64_t n = g.size(0), c = g.size(1);
  Tensor out = empty_dev({n, c}, at::kFloat);
  launch("smax_bwd", {bind(g), bind(y), bind(out), bind_buf(uniform({u(n), u(c), u(mode), 0}))}, u(cdiv(n, 64)));
  return out;
}
static Tensor softmax_bwd_(const Tensor& g, const Tensor& y, int64_t, at::ScalarType) { return smax_bwd_impl(g, y, 0); }
static Tensor log_softmax_bwd_(const Tensor& g, const Tensor& y, int64_t, at::ScalarType) { return smax_bwd_impl(g, y, 1); }

static Tensor argmax_(const Tensor& x_, std::optional<int64_t> dim, bool keepdim) {
  TORCH_CHECK(f32(x_) && x_.dim() == 2 && dim && (*dim == 1 || *dim == -1) && !keepdim, "omarchy-mac-torch: argmax(dim=1)");
  Tensor x = contig(x_);
  int64_t n = x.size(0), c = x.size(1);
  Tensor out = empty_dev({n}, at::kLong);
  launch("argmax_row", {bind(x), bind(out), bind_buf(uniform({u(n), u(c), u(off_words(x)), 0}))}, u(cdiv(n, 64)));
  return out;
}
static Tensor eq_t(const Tensor& a_, const Tensor& b_) {
  Tensor a = contig(a_), b = contig(b_);
  TORCH_CHECK(a.sizes() == b.sizes() && !a.is_floating_point(), "omarchy-mac-torch: eq on same-shape integer tensors");
  Tensor out = empty_dev(a.sizes(), at::kBool);
  launch("eqw", {bind(a), bind(b), bind(out), bind_buf(uniform({u(a.numel()), u(off_words(a)), u(off_words(b)), 0, 0, 0, 0, 0}))}, u(cdiv(a.numel(), 64)));
  return out;
}
static Tensor eq_s(const Tensor& a_, const Scalar& v) {
  Tensor a = contig(a_);
  Tensor out = empty_dev(a.sizes(), at::kBool);
  launch("eqw", {bind(a), bind(a), bind(out), bind_buf(uniform({u(a.numel()), u(off_words(a)), 0, 0, 1, (uint32_t)v.to<int64_t>(), 0, 0}))}, u(cdiv(a.numel(), 64)));
  return out;
}
static Tensor iadd_t(const Tensor& a_, const Tensor& b_, const Scalar&) {
  if (!is_dev(b_) && b_.dim() == 0) {  // python int operand
    Tensor a = contig(a_);
    Tensor out = empty_dev(a.sizes(), at::kLong);
    launch("iaddw", {bind(a), bind(a), bind(out), bind_buf(uniform({u(a.numel()), u(off_words(a)), 0, 0, 1, (uint32_t)b_.item<int64_t>(), 0, 0}))}, u(cdiv(a.numel(), 64)));
    return out;
  }
  Tensor a = contig(a_), b = contig(b_);
  Tensor out = empty_dev(a.sizes(), at::kLong);
  launch("iaddw", {bind(a), bind(b), bind(out), bind_buf(uniform({u(a.numel()), u(off_words(a)), u(off_words(b)), 0, 0, 0, 0, 0}))}, u(cdiv(a.numel(), 64)));
  return out;
}

// dispatch helpers that route on dtype (float vs integer) for ops registered once
static Tensor add_dispatch(const Tensor& a, const Tensor& b, const Scalar& alpha) {
  if (a.scalar_type() == at::kLong || a.scalar_type() == at::kInt) return iadd_t(a, b, alpha);
  return add_t(a, b, alpha);
}
static Tensor& add__dispatch(Tensor& self, const Tensor& o, const Scalar& alpha) {
  if (self.scalar_type() == at::kLong || self.scalar_type() == at::kInt) {
    Tensor r = iadd_t(self, o, alpha);
    launch("icopy", {bind(r), bind(self), bind_buf(uniform({u(self.numel()), 0, u(off_words(self)), 0}))}, u(cdiv(self.numel(), 64)));
    return self;
  }
  return add__t(self, o, alpha);
}
static Tensor& add__s_dispatch(Tensor& self, const Scalar& o, const Scalar& alpha) {
  if (self.scalar_type() == at::kLong || self.scalar_type() == at::kInt) {
    launch("iadd", {bind(self), bind_buf(uniform({u(self.numel()), 0, u(off_words(self)), (uint32_t)(o.to<int64_t>() * alpha.to<int64_t>())}))}, u(cdiv(self.numel(), 64)));
    return self;
  }
  return add__s(self, o, alpha);
}

// ------------------------------------------------------------------------------------------------
// Fused cross-entropy (autograd Function; hooked from Python via torch_function)
// ------------------------------------------------------------------------------------------------

struct CrossEntropyFn : public torch::autograd::Function<CrossEntropyFn> {
  static Tensor forward(torch::autograd::AutogradContext* ctx, const Tensor& logits_, const Tensor& target, double eps) {
    Tensor logits = base0(logits_);
    int64_t n = logits.size(0), c = logits.size(1);
    Tensor gout = empty_dev({n, c}, at::kFloat), losses = empty_dev({n}, at::kFloat);
    launch("ce", {bind(logits), bind(target), bind(gout), bind(losses), bind_buf(uniform({u(n), u(c), fbits((float)eps), 0}))}, u(cdiv(n, 64)));
    ctx->saved_data["gout"] = gout;
    return mean_default(losses, std::nullopt);
  }
  static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx, torch::autograd::variable_list g) {
    Tensor gout = ctx->saved_data["gout"].toTensor();
    return {mul_t(gout, g[0]), Tensor(), Tensor()};
  }
};
static Tensor cross_entropy_fused(const Tensor& logits, const Tensor& target, double eps) {
  return CrossEntropyFn::apply(logits, target, eps);
}

// ------------------------------------------------------------------------------------------------
// Fused Adam (4 params per dispatch)
// ------------------------------------------------------------------------------------------------

static Buf* g_sgd_hyper = nullptr;
static void fused_sgd_(at::TensorList params, at::TensorList grads, at::TensorList momentum_bufs, double weight_decay, double momentum,
                       double lr, double dampening, bool nesterov, bool maximize, bool is_first_step,
                       const std::optional<Tensor>& grad_scale, const std::optional<Tensor>& found_inf) {
  TORCH_CHECK(!grad_scale && !found_inf, "omarchy-mac-torch: fused sgd grad_scale/found_inf");
  float h[8] = {(float)lr, (float)momentum, (float)dampening, (float)weight_decay, 0, 0, 0, 0};
  uint32_t flags[3] = {nesterov ? 1u : 0u, is_first_step ? 1u : 0u, maximize ? 1u : 0u};
  std::memcpy(&h[4], flags, 12);
  if (!g_sgd_hyper) {
    WGPUBufferDescriptor d = {};
    d.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
    d.size = 32;
    g_sgd_hyper = new Buf{wgpuDeviceCreateBuffer(C.device, &d), 32};
  }
  submit_pending();
  wgpuQueueWriteBuffer(C.queue, g_sgd_hyper->b, 0, h, 32);
  for (size_t i = 0; i < params.size(); i++) {
    const Tensor& mbuf = momentum >= 0 && momentum_bufs.size() > i && momentum_bufs[i].defined() ? momentum_bufs[i] : params[i];
    Buf* a = uniform({u(params[i].numel()), u(off_words(params[i])), u(off_words(grads[i])), u(off_words(mbuf))});
    ew_launch("sgd", {bind(params[i]), bind(grads[i]), bind(mbuf), bind_buf(a), bind_buf(g_sgd_hyper)}, params[i].numel());
  }
}

static Buf* g_hyper = nullptr;

static void fused_adam_(at::TensorList params, at::TensorList grads, at::TensorList exp_avgs, at::TensorList exp_avg_sqs,
                        at::TensorList, at::TensorList state_steps, double lr, double beta1, double beta2,
                        double weight_decay, double eps, bool amsgrad, bool maximize,
                        const std::optional<Tensor>& grad_scale, const std::optional<Tensor>& found_inf) {
  TORCH_CHECK(!amsgrad && !maximize && !grad_scale && !found_inf, "omarchy-mac-torch: fused adam options");
  double step = state_steps[0].item<double>();
  float hyper[8] = {(float)lr, (float)beta1, (float)beta2, (float)eps, (float)(1 - std::pow(beta1, step)),
                    (float)std::sqrt(1 - std::pow(beta2, step)), (float)weight_decay, 0.f};
  if (!g_hyper) {
    WGPUBufferDescriptor d = {};
    d.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
    d.size = 32;
    g_hyper = new Buf{wgpuDeviceCreateBuffer(C.device, &d), 32};
  }
  submit_pending();  // earlier work may still read the old hyper-parameters
  wgpuQueueWriteBuffer(C.queue, g_hyper->b, 0, hyper, 32);
  size_t i = 0, n = params.size();
  while (i + 4 <= n) {
    std::vector<uint32_t> w;
    std::vector<Binding> bs;
    int64_t total = 0;
    for (int k = 0; k < 4; k++) { w.push_back(u(params[i + k].numel())); total += params[i + k].numel(); }
    for (int k = 0; k < 4; k++) w.push_back(u(off_words(params[i + k])));
    for (int k = 0; k < 4; k++) w.push_back(u(off_words(grads[i + k])));
    for (int k = 0; k < 4; k++) w.push_back(u(off_words(exp_avgs[i + k])));
    for (int k = 0; k < 4; k++) w.push_back(u(off_words(exp_avg_sqs[i + k])));
    for (int k = 0; k < 4; k++) bs.push_back(bind(params[i + k]));
    for (int k = 0; k < 4; k++) bs.push_back(bind(grads[i + k]));
    for (int k = 0; k < 4; k++) bs.push_back(bind(exp_avgs[i + k]));
    for (int k = 0; k < 4; k++) bs.push_back(bind(exp_avg_sqs[i + k]));
    bs.push_back(bind_buf(uniform(w)));
    bs.push_back(bind_buf(g_hyper));
    ew_launch("adam4", bs, total);
    i += 4;
  }
  for (; i < n; i++) {
    Buf* a = uniform({u(params[i].numel()), u(off_words(params[i])), u(off_words(grads[i])), u(off_words(exp_avgs[i])), u(off_words(exp_avg_sqs[i])), 0, 0, 0});
    ew_launch("adam", {bind(params[i]), bind(grads[i]), bind(exp_avgs[i]), bind(exp_avg_sqs[i]), bind_buf(a), bind_buf(g_hyper)}, params[i].numel());
  }
}

// ------------------------------------------------------------------------------------------------
// Device guard, hooks, CPU fallback
// ------------------------------------------------------------------------------------------------

struct Guard final : c10::impl::DeviceGuardImplInterface {
  c10::DeviceType type() const override { return kDev; }
  c10::Device exchangeDevice(c10::Device) const override { return {kDev, 0}; }
  c10::Device getDevice() const override { return {kDev, 0}; }
  void setDevice(c10::Device) const override {}
  void uncheckedSetDevice(c10::Device) const noexcept override {}
  c10::Stream getStream(c10::Device d) const noexcept override { return c10::Stream(c10::Stream::DEFAULT, d); }
  c10::Stream getDefaultStream(c10::Device d) const override { return c10::Stream(c10::Stream::DEFAULT, d); }
  c10::Stream exchangeStream(c10::Stream s) const noexcept override { return s; }
  c10::DeviceIndex deviceCount() const noexcept override { return 1; }
  void record(void**, const c10::Stream&, const c10::DeviceIndex, const c10::EventFlag) const override {}
  void block(void*, const c10::Stream&) const override {}
  bool queryEvent(void*) const override { return true; }
  void destroyEvent(void*, const c10::DeviceIndex) const noexcept override {}
  bool queryStream(const c10::Stream&) const override { return true; }
  void synchronizeStream(const c10::Stream&) const override { sync_all(); }
  void synchronizeDevice(const c10::DeviceIndex) const override { sync_all(); }
};
C10_REGISTER_GUARD_IMPL(PrivateUse1, Guard);

struct Hooks final : at::PrivateUse1HooksInterface {
  bool isBuilt() const override { return true; }
  bool isAvailable() const override { return true; }
  bool hasPrimaryContext(c10::DeviceIndex) const override { return true; }
  const at::Generator& getDefaultGenerator(c10::DeviceIndex) const override {
    static at::Generator g = at::detail::createCPUGenerator();
    return g;
  }
};
struct HooksRegistrar {
  HooksRegistrar() {
    static Hooks hooks;
    at::RegisterPrivateUse1HooksInterface(&hooks);
  }
} g_registrar;

static void cpu_fallback_(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  g_fallbacks[op.schema().name() + "." + op.schema().overload_name()]++;
  at::native::cpu_fallback(op, stack);
}

// ------------------------------------------------------------------------------------------------
// Registration
// ------------------------------------------------------------------------------------------------

TORCH_LIBRARY_IMPL(_, PrivateUse1, m) { m.fallback(torch::CppFunction::makeFromBoxedFunction<&cpu_fallback_>()); }

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("empty.memory_format", &empty_memory_format);
  m.impl("empty_strided", &empty_strided);
  m.impl("empty_like", &empty_like_);
  m.impl("_copy_from", &_copy_from_);
  m.impl("_copy_from_and_resize", &_copy_from_and_resize_);
  m.impl("copy_", &copy_);
  m.impl("_to_copy", &_to_copy_);
  m.impl("clone", &clone_);
  m.impl("_local_scalar_dense", &_local_scalar_dense_);
  m.impl("ones_like", &ones_like_);
  m.impl("zeros_like", &zeros_like_);
  m.impl("set_.source_Storage_storage_offset", &set_storage_);
  m.impl("view", &at::native::view);
  m.impl("_unsafe_view", &at::native::_unsafe_view);
  m.impl("as_strided", &at::native::as_strided_tensorimpl);
  m.impl("_reshape_alias", &at::native::_reshape_alias);
  m.impl("resize_", &at::native::resize_);
  m.impl("add.Tensor", &add_dispatch);
  m.impl("add.Scalar", &add_s);
  m.impl("sub.Tensor", &sub_t);
  m.impl("sub.Scalar", &sub_s);
  m.impl("mul.Tensor", &mul_t);
  m.impl("mul.Scalar", &mul_s);
  m.impl("div.Tensor", &div_t);
  m.impl("div.Scalar", &div_s);
  m.impl("neg", &neg_);
  m.impl("relu", &relu_);
  m.impl("exp", &exp_);
  m.impl("sqrt", &sqrt_);
  m.impl("sigmoid", &sigmoid_);
  m.impl("silu", &silu_);
  m.impl("hardswish", &hardswish_);
  m.impl("hardsigmoid", &hardsigmoid_);
  m.impl("tanh", &tanh_);
  m.impl("sigmoid_backward", &sigmoid_bwd);
  m.impl("silu_backward", &silu_bwd);
  m.impl("hardswish_backward", &hardswish_bwd);
  m.impl("hardsigmoid_backward", &hardsigmoid_bwd);
  m.impl("tanh_backward", &tanh_bwd);
  m.impl("threshold_backward", &threshold_bwd);
  m.impl("xlogy.Tensor", &xlogy_t);
  m.impl("add_.Tensor", &add__dispatch);
  m.impl("add_.Scalar", &add__s_dispatch);
  m.impl("sub_.Tensor", &sub__t);
  m.impl("mul_.Tensor", &mul__t);
  m.impl("mul_.Scalar", &mul__s);
  m.impl("div_.Tensor", &div__t);
  m.impl("div_.Scalar", &div__s);
  m.impl("zero_", &omt_zero_);
  m.impl("fill_.Scalar", &fill_s);
  m.impl("sum", &sum_default);
  m.impl("sum.dim_IntList", &sum_dim);
  m.impl("mean.dim", &mean_dim);
  m.impl("mean", &mean_default);
  m.impl("mm", &mm_);
  m.impl("addmm", &addmm_);
  m.impl("convolution", &convolution_);
  m.impl("convolution_backward", &convolution_backward_);
  m.impl("native_batch_norm", &batch_norm_);
  m.impl("native_batch_norm_backward", &batch_norm_backward_);
  m.impl("max_pool2d_with_indices", &max_pool2d_);
  m.impl("max_pool2d_with_indices_backward", &max_pool2d_backward_);
  m.impl("avg_pool2d", &avg_pool2d_);
  m.impl("avg_pool2d_backward", &avg_pool2d_backward_);
  m.impl("_softmax", &softmax_);
  m.impl("_log_softmax", &log_softmax_);
  m.impl("_softmax_backward_data", &softmax_bwd_);
  m.impl("_log_softmax_backward_data", &log_softmax_bwd_);
  m.impl("argmax", &argmax_);
  m.impl("eq.Tensor", &eq_t);
  m.impl("eq.Scalar", &eq_s);
  m.impl("_fused_adam_", &fused_adam_);
}


// ------------------------------------------------------------------------------------------------
// More elementwise ops, comparisons, where, clamp, cat/stack/index, bmm, dropout, layer norm, RNG
// ------------------------------------------------------------------------------------------------

static Tensor clamp_(const Tensor& x, const std::optional<Scalar>& lo, const std::optional<Scalar>& hi) {
  Tensor out = empty_dev(x.sizes(), at::kFloat);
  launch_ew(23, &x, nullptr, out, x.sizes(), lo ? lo->to<float>() : -3.4e38f, hi ? hi->to<float>() : 3.4e38f);
  return out;
}
static Tensor log_(const Tensor& a) { return ew_unary(24, a); }
static Tensor abs_(const Tensor& a) { return ew_unary(25, a); }
static Tensor rsqrt_(const Tensor& a) { return ew_unary(26, a); }
static Tensor pow_s(const Tensor& a, const Scalar& e) { return ew_unary(27, a, e.to<float>()); }

static Tensor cmp_impl(int op, const Tensor& a_, const Tensor* b_, double sval) {
  Tensor a = contig(a_);
  Tensor out = empty_dev(a.sizes(), at::kBool);
  Tensor b;
  if (b_) b = contig(*b_);
  Buf* ub = uniform({u(a.numel()), u(op), (uint32_t)(b_ ? 0 : 1), u(off_words(a)), b_ ? u(off_words(b)) : 0u, 0, fbits((float)sval), 0});
  launch("cmpw", {bind(a), bind(b_ ? b : a), bind(out), bind_buf(ub)}, u(cdiv(a.numel(), 64)));
  return out;
}
#define CMP_OPS(NAME, CODE) \
  static Tensor NAME##_s(const Tensor& a, const Scalar& v) { return cmp_impl(CODE, a, nullptr, v.to<double>()); } \
  static Tensor NAME##_t(const Tensor& a, const Tensor& b) { return cmp_impl(CODE, a, &b, 0); }
CMP_OPS(lt, 0)
CMP_OPS(le, 1)
CMP_OPS(gt, 2)
CMP_OPS(ge, 3)
CMP_OPS(ne, 4)
static Tensor eq_float_s(const Tensor& a, const Scalar& v) {
  if (a.is_floating_point() || v.isFloatingPoint()) return cmp_impl(5, a, nullptr, v.to<double>());
  return eq_s(a, v);
}

static Tensor where_(const Tensor& cond_, const Tensor& a_, const Tensor& b_) {
  TORCH_CHECK(cond_.scalar_type() == at::kBool, "omarchy-mac-torch: where needs a bool condition");
  auto shape = at::infer_size(cond_.sizes(), at::infer_size(a_.sizes(), b_.sizes()));
  Tensor cond = contig(cond_.expand(shape));
  Tensor a = is_dev(a_) ? contig(a_.expand(shape)) : to_dev(a_.expand(shape).contiguous());
  Tensor b = is_dev(b_) ? contig(b_.expand(shape)) : to_dev(b_.expand(shape).contiguous());
  Tensor out = empty_dev(shape, a.scalar_type());
  int64_t n = out.numel();
  Buf* ub = uniform({u(n), u(off_words(cond)), u(off_words(a)), u(off_words(b)), 0, 0, 0, 0});
  launch("where3", {bind(cond), bind(a), bind(b), bind(out), bind_buf(ub)}, u(cdiv(n, 64)));
  return out;
}

static Tensor cat_(const at::ITensorListRef& list, int64_t dim) {
  std::vector<Tensor> ts;
  for (const Tensor& t : list) ts.push_back(t);
  bool ok = dim == 0 && !ts.empty();
  for (auto& t : ts) ok = ok && f32(t) && t.dim() >= 1 && t.numel() > 0;
  if (!ok) {
    std::vector<Tensor> h;
    for (auto& t : ts) h.push_back(to_host(t));
    return to_dev(at::cat(h, dim));
  }
  int64_t rows = 0;
  for (auto& t : ts) rows += t.size(0);
  std::vector<int64_t> shape(ts[0].sizes().begin(), ts[0].sizes().end());
  shape[0] = rows;
  Tensor out = empty_dev(shape, at::kFloat);
  int64_t off = 0;
  for (auto& t : ts) {
    Tensor tc = contig(t);
    launch_ew(9, &tc, nullptr, out, tc.sizes(), 1.f, 0.f, off);
    off += tc.numel();
  }
  return out;
}
static Tensor stack_(at::TensorList list, int64_t dim) {
  bool ok = dim == 0 && !list.empty();
  for (auto& t : list) ok = ok && f32(t) && t.numel() > 0;
  if (!ok) {
    std::vector<Tensor> h;
    for (auto& t : list) h.push_back(to_host(t));
    return to_dev(at::stack(h, dim));
  }
  std::vector<int64_t> shape = {(int64_t)list.size()};
  for (auto d : list[0].sizes()) shape.push_back(d);
  Tensor out = empty_dev(shape, at::kFloat);
  int64_t off = 0;
  for (auto& t : list) {
    Tensor tc = contig(t);
    launch_ew(9, &tc, nullptr, out, tc.sizes(), 1.f, 0.f, off);
    off += tc.numel();
  }
  return out;
}

static Tensor gather0(const Tensor& x_, const Tensor& idx_) {
  Tensor x = contig(x_);
  Tensor idx = is_dev(idx_) ? contig(idx_) : to_dev(idx_.contiguous());
  int64_t n = idx.numel(), row = x.numel() / std::max<int64_t>(x.size(0), 1);
  std::vector<int64_t> shape = {n};
  for (int i = 1; i < x.dim(); i++) shape.push_back(x.size(i));
  Tensor out = empty_dev(shape, x.scalar_type());
  Buf* ub = uniform({u(n), u(row), u(off_words(x)), u(off_words(idx))});
  uint32_t gx, gy;
  grid(n * row, 256, gx, gy);
  launch("gather0", {bind(x), bind(idx), bind(out), bind_buf(ub)}, gx, gy);
  return out;
}
static Tensor index_(const Tensor& x, const c10::List<std::optional<Tensor>>& indices) {
  if (indices.size() == 1 && indices[0].has_value() && indices[0]->defined() &&
      (indices[0]->scalar_type() == at::kLong || indices[0]->scalar_type() == at::kInt) && indices[0]->dim() == 1 && x.dim() >= 1)
    return gather0(x, *indices[0]);
  c10::List<std::optional<Tensor>> h;
  for (size_t i = 0; i < indices.size(); i++) {
    std::optional<Tensor> t = indices[i];
    h.push_back((t && t->defined()) ? std::optional<Tensor>(to_host(*t)) : std::nullopt);
  }
  return to_dev(at::index(to_host(x), h));
}
static Tensor index_select_(const Tensor& x, int64_t dim, const Tensor& idx) {
  if (dim == 0 || dim == -x.dim()) return gather0(x, idx);
  return to_dev(at::index_select(to_host(x), dim, to_host(idx)));
}

static Tensor bmm_(const Tensor& a, const Tensor& b) {
  TORCH_CHECK(f32(a) && f32(b) && a.dim() == 3 && b.dim() == 3, "omarchy-mac-torch: bmm float 3-D");
  int64_t B = a.size(0), m = a.size(1), n = b.size(2);
  Tensor out = empty_dev({B, m, n}, at::kFloat);
  for (int64_t i = 0; i < B; i++) {
    Tensor ai = a.select(0, i), bi = b.select(0, i);
    mm_impl(ai, bi, nullptr, &out, i * m * n);
  }
  return out;
}

static std::tuple<Tensor, Tensor> dropout_(const Tensor& x_, double p, std::optional<bool> train) {
  Tensor x = contig(x_);
  Tensor mask = empty_dev(x.sizes(), at::kBool);
  Tensor out = empty_dev(x.sizes(), at::kFloat);
  int64_t n = x.numel();
  if (!train.value_or(true) || p == 0) {
    launch_ew(9, &x, nullptr, out, x.sizes());
    Tensor one = fill_s(mask, 1);
    return {out, mask};
  }
  uint32_t seed = (uint32_t)at::detail::getDefaultCPUGenerator().get<at::CPUGeneratorImpl>()->random();
  Buf* ub = uniform({u(n), seed, fbits((float)p), fbits((float)(1.0 / (1.0 - p))), u(off_words(x)), 0, 0, 0});
  launch("dropout", {bind(x), bind(out), bind(mask), bind_buf(ub)}, u(cdiv(n, 64)));
  return {out, mask};
}
static Tensor dropout_backward_(const Tensor& g_, const Tensor& mask, double scale) {
  Tensor g = contig(g_);
  Tensor out = empty_dev(g.sizes(), at::kFloat);
  int64_t n = g.numel();
  launch("mulmask", {bind(g), bind(mask), bind(out), bind_buf(uniform({u(n), fbits((float)scale), u(off_words(g)), 0}))}, u(cdiv(n, 64)));
  return out;
}

static std::tuple<Tensor, Tensor, Tensor> layer_norm_(const Tensor& x_, IntArrayRef normalized, const std::optional<Tensor>& w,
                                                      const std::optional<Tensor>& b, double eps) {
  Tensor x = base0(x_);
  int64_t D = 1;
  for (auto d : normalized) D *= d;
  int64_t R = x.numel() / D;
  Tensor out = empty_dev(x.sizes(), at::kFloat), mean = empty_dev({R}, at::kFloat), rstd = empty_dev({R}, at::kFloat);
  bool aff = w && w->defined() && b && b->defined();
  Tensor g = aff ? base0(*w) : x, bt = aff ? base0(*b) : x;
  launch("ln_fwd", {bind(x), bind(g), bind(bt), bind(out), bind(mean), bind(rstd),
                    bind_buf(uniform({u(R), u(D), fbits((float)eps), (uint32_t)(aff ? 1 : 0)}))}, u(R));
  return {out, mean, rstd};
}
static std::tuple<Tensor, Tensor, Tensor> layer_norm_backward_(const Tensor& gy_, const Tensor& x_, IntArrayRef normalized,
                                                               const Tensor& mean, const Tensor& rstd,
                                                               const std::optional<Tensor>& w, const std::optional<Tensor>&,
                                                               std::array<bool, 3> mask) {
  Tensor gy = base0(gy_), x = base0(x_);
  int64_t D = 1;
  for (auto d : normalized) D *= d;
  int64_t R = x.numel() / D;
  bool aff = w && w->defined();
  Tensor g = aff ? base0(*w) : x;
  Tensor dx, dg, db;
  Buf* ub = uniform({u(R), u(D), 0, (uint32_t)(aff ? 1 : 0)});
  if (mask[0]) {
    dx = empty_dev(x.sizes(), at::kFloat);
    launch("ln_bwd_x", {bind(gy), bind(x), bind(g), bind(mean), bind(rstd), bind(dx), bind_buf(ub)}, u(R));
  }
  if (mask[1] || mask[2]) {
    dg = empty_dev({D}, at::kFloat);
    db = empty_dev({D}, at::kFloat);
    launch("ln_bwd_w", {bind(gy), bind(x), bind(mean), bind(rstd), bind(dg), bind(db), bind_buf(ub)}, u(cdiv(D, 64)));
  }
  return {dx, mask[1] ? dg : Tensor(), mask[2] ? db : Tensor()};
}

static Tensor& uniform_(Tensor& self, double from, double to, std::optional<at::Generator> gen) {
  Tensor h = at::empty(self.sizes(), at::kFloat).uniform_(from, to, gen);
  Tensor tmp = to_dev(h);
  return copy_into(self, tmp);
}
static Tensor& normal_(Tensor& self, double mean, double std, std::optional<at::Generator> gen) {
  Tensor h = at::empty(self.sizes(), at::kFloat).normal_(mean, std, gen);
  Tensor tmp = to_dev(h);
  return copy_into(self, tmp);
}

static void empty_cache() {
  sync_all();
  std::lock_guard<std::mutex> g(g_pool_mu);
  for (auto& kv : g_free)
    for (Buf* b : kv.second) {
      g_reserved -= b->bytes;
      wgpuBufferDestroy(b->b);
      wgpuBufferRelease(b->b);
      delete b;
    }
  g_free.clear();
  g_groups.clear();  // bind groups may reference freed buffers
}

// ------------------------------------------------------------------------------------------------
// Native fused conv stacks (the C++ counterpart of fused.py's _Plan)
// ------------------------------------------------------------------------------------------------

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
  std::vector<Buf*> owned;
  ~Plan() {
    for (Buf* b : owned) free_buf(b);
  }
};

static Buf* nb(Plan& p, int64_t numel) {
  Buf* b = alloc_buf(std::max<int64_t>(numel, 1) * 4);
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
static std::pair<uint32_t, uint32_t> g2(int64_t total) {
  uint32_t gx, gy;
  grid(total, 256, gx, gy);
  return {gx, gy};
}
static void add_step(std::vector<Step>& v, const std::string& k, std::vector<BRef> b, uint32_t x, uint32_t y = 1, uint32_t z = 1) {
  v.push_back({k, std::move(b), x, y, z});
}
static void add_ew(std::vector<Step>& v, const std::string& k, std::vector<BRef> b, int64_t total) {
  auto g = g2(total);
  add_step(v, k, std::move(b), g.first, g.second);
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
    add_step(v, pl.kernel, {a, b, R_(part), R_(pr)}, u(cdiv(N, pl.bn)), u(cdiv(M, pl.bm)), u(slices));
    add_step(v, "gemm_reduce", {R_(part), out, R_(pbuf_reduce(M, N, slices))}, u(cdiv(M * N, 256)));
  } else {
    Buf* pr = pbuf_geo(g, M, N, K, cdiv(K, 16) * 16, 1);
    std::vector<BRef> bs = {a, b, out, R_(pr)};
    for (auto& e : extra) bs.push_back(e);
    add_step(v, pl.kernel, bs, u(cdiv(N, pl.bn)), u(cdiv(M, pl.bm)), 1);
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
  init_ctx();
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
        add_ew(p.fwd, "dw_fwd_bn", {R_(cur), RT(params[pi]), R_(out), R_(q), RT(rms[pend.bn]), RT(rvs[pend.bn]),
                                    RT(params[pend.gamma]), RT(params[pend.gamma + 1])}, n * c * h * w);
        pend.on = false;
      } else {
        add_ew(p.fwd, "dw_fwd", {R_(cur), RT(params[pi]), R_(out), R_(q)}, n * c * h * w);
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
        add_step(p.fwd, "bn_stats_part", {R_(cur), R_(part), R_(qs)}, u(c), u(S));
        add_step(p.fwd, "bn_stats_fin", {R_(cur), R_(part), R_(stats), RT(rms[bi]), RT(rvs[bi]), R_(qs), R_(r)}, u(cdiv(c, 64)));
        add_ew(p.fwd, "bn_relu", {R_(cur), R_(stats), RT(params[pi]), RT(params[pi + 1]), R_(y), R_(q)}, n * c * h * w);
      } else {
        add_ew(p.fwd, "bn_relu_rv", {R_(cur), RT(rms[bi]), RT(params[pi]), RT(params[pi + 1]), R_(y), R_(q), RT(rvs[bi])}, n * c * h * w);
      }
      BackRec r; r.kind = "bn"; r.x = cur; r.y = y; r.stats = stats; r.q = q; r.param = pi; r.n = n; r.c = c; r.h = h; r.w = w;
      recs.push_back(r);
      cur = y; bi++; pi += 2;
    } else if (L.kind == "pool") {
      int64_t oh = h / 2, ow = w / 2;
      Buf* out = nb(p, n * c * oh * ow);
      Buf* idx = nb(p, n * c * oh * ow);
      Buf* q = uniform({u(n), u(c), u(h), u(w), u(oh), u(ow), 0, 0});
      add_ew(p.fwd, "pool_fwd", {R_(cur), R_(out), R_(idx), R_(q)}, n * c * oh * ow);
      BackRec r; r.kind = "pool"; r.idx = idx; r.q = q; r.n = n; r.c = c; r.h = h; r.w = w;
      recs.push_back(r);
      cur = out; h = oh; w = ow;
    } else if (L.kind == "apool") {
      int64_t oh = h / 2, ow = w / 2;
      Buf* out = nb(p, n * c * oh * ow);
      Buf* q = uniform({u(n), u(c), u(h), u(w), u(oh), u(ow), 0, 0});
      add_ew(p.fwd, "avgpool_fwd", {R_(cur), R_(out), R_(q)}, n * c * oh * ow);
      BackRec r; r.kind = "apool"; r.q = q; r.n = n; r.c = c; r.h = h; r.w = w;
      recs.push_back(r);
      cur = out; h = oh; w = ow;
    } else {  // gap
      Buf* out = nb(p, n * c);
      Buf* q = uniform({u(n), u(c), u(h), u(w), 0, 0, 0, 0});
      add_ew(p.fwd, "gap_fwd", {R_(cur), R_(out), R_(q)}, n * c);
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
      add_ew(p.bwd, "pool_bwd", {R_(g), R_(r.idx), R_(gx), R_(r.q)}, r.n * r.c * r.h * r.w);
      g = gx;
    } else if (r.kind == "apool") {
      Buf* gx = nb(p, r.n * r.c * r.h * r.w);
      add_ew(p.bwd, "avgpool_bwd", {R_(g), R_(gx), R_(r.q)}, r.n * r.c * r.h * r.w);
      g = gx;
    } else if (r.kind == "gap") {
      Buf* gx = nb(p, r.n * r.c * r.h * r.w);
      add_ew(p.bwd, "gap_bwd", {R_(g), R_(gx), R_(r.q)}, r.n * r.c * r.h * r.w);
      g = gx;
    } else if (r.kind == "bn") {
      int64_t slot;
      BRef dgb = GS(p, 2 * r.c, slot);
      p.grads.push_back({r.param, slot, true});
      Buf* gx = nb(p, r.n * r.c * r.h * r.w);
      int64_t S = r.n >= 8 ? 8 : 1;
      Buf* qs = uniform({u(r.n), u(r.c), u(r.h), u(r.w), 0, 0, u(S), 0});
      Buf* part = nb(p, r.c * S * 2);
      add_step(p.bwd, "bn_bwd_part", {R_(g), R_(r.x), R_(r.stats), R_(r.y), R_(part), R_(qs)}, u(r.c), u(S));
      add_step(p.bwd, "bn_bwd_fin", {R_(part), dgb, R_(qs)}, u(cdiv(r.c, 64)));
      add_ew(p.bwd, "bn_bwd_apply", {R_(g), R_(r.x), R_(r.stats), R_(r.y), RT(params[r.param]), dgb, R_(gx), R_(r.q)},
             r.n * r.c * r.h * r.w);
      g = gx;
    } else if (r.kind == "dw") {
      int64_t slot;
      BRef gw = GS(p, r.c * r.ks * r.ks, slot);
      p.grads.push_back({r.param, slot, false});
      add_step(p.bwd, "dw_gwt", {R_(g), R_(r.x), gw, R_(r.q)}, u(r.c * r.ks * r.ks));
      if (!first || input_grad) {
        Buf* gx = nb(p, r.n * r.c * r.h * r.w);
        add_ew(p.bwd, "dw_gin", {R_(g), RT(params[r.param]), R_(gx), R_(r.q)}, r.n * r.c * r.h * r.w);
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


static ConvPlan autotune(int kind, const Geo& g) {
  g_in_tune = true;
  sync_all();
  Tensor x = empty_dev({g.n, g.ci, g.ih, g.iw}, at::kFloat), wt = empty_dev({g.co, g.ci, g.ks, g.ks}, at::kFloat);
  Tensor gy = empty_dev({g.n, g.co, g.oh, g.ow}, at::kFloat);
  Tensor out = empty_dev({std::max<int64_t>(std::max<int64_t>(x.numel(), gy.numel()), wt.numel())}, at::kFloat);
  auto run = [&]() {
    if (kind == 0) gemm_forward(x, wt, out, g);
    else if (kind == 1) gemm_grad_input(gy, wt, out, g);
    else gemm_grad_weight(x, gy, out, g);
  };
  ConvPlan best = plan_for(kind, g, -1);
  double best_ms = 1e30;
  std::vector<double> mults = kind == 2 ? std::vector<double>{0.25, 0.5, 1.0, 2.0} : std::vector<double>{1.0};
  for (int t = 0; t < g_num_tiles; t++) {
    ConvPlan cand = plan_for(kind, g, t);
    for (double m : mults) {
      cand.mult = m;
      g_force = &cand;
      run();  // warm up (compiles the pipeline)
      sync_all();
      auto t0 = std::chrono::steady_clock::now();
      run();
      sync_all();
      double first = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
      int reps = (int)std::max(1.0, std::min(6.0, 15.0 / std::max(first, 0.05)));  // ~15 ms of timing per candidate
      double ms = first;
      if (reps > 1) {
        auto t1 = std::chrono::steady_clock::now();
        for (int r = 0; r < reps; r++) run();
        sync_all();
        ms = std::min(first, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count() / reps);
      }
      if (ms < best_ms) { best_ms = ms; best = cand; }
    }
  }
  g_force = nullptr;
  g_in_tune = false;
  return best;
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
static void copy_words_(Buf* src, uint64_t src_off_words, Buf* dst, uint64_t dst_off_words, int64_t n) {
  launch("icopy", {bind_buf(src), bind_buf(dst), bind_buf(uniform({u(n), u(src_off_words), u(dst_off_words), 0}))}, u(cdiv(n, 64)));
}

static Tensor plan_forward(Plan& p, const Tensor& x_) {
  Tensor x = contig(x_);
  TORCH_CHECK(f32(x) && x.numel() == p.n * p.c * p.h * p.w, "device plan: bad input");
  copy_words_(bufof(x), off_words(x), p.xbuf, 0, x.numel());
  run_steps(p, p.fwd, nullptr);
  Tensor out = empty_dev(p.out_shape, at::kFloat);
  copy_words_(p.outbuf, 0, bufof(out), 0, p.out_numel);
  return out;
}

static std::pair<std::optional<Tensor>, std::vector<Tensor>> plan_backward(Plan& p, const Tensor& g_) {
  Tensor g = contig(g_);
  if (p.gstore.storage().use_count() > 1) p.gstore = empty_dev({std::max<int64_t>(p.gwords, 64)}, at::kFloat);  // previous grads alive
  copy_words_(bufof(g), off_words(g), p.gout, 0, g.numel());
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
    copy_words_(p.gin, 0, bufof(t), 0, p.in_numel);
    gx = t;
  }
  return {gx, grads};
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m2) {
  m2.impl("eq.Scalar", &eq_float_s);
  m2.impl("clamp", &clamp_);
  m2.impl("log", &log_);
  m2.impl("abs", &abs_);
  m2.impl("rsqrt", &rsqrt_);
  m2.impl("pow.Tensor_Scalar", &pow_s);
  m2.impl("lt.Scalar", &lt_s);
  m2.impl("le.Scalar", &le_s);
  m2.impl("gt.Scalar", &gt_s);
  m2.impl("ge.Scalar", &ge_s);
  m2.impl("ne.Scalar", &ne_s);
  m2.impl("lt.Tensor", &lt_t);
  m2.impl("le.Tensor", &le_t);
  m2.impl("gt.Tensor", &gt_t);
  m2.impl("ge.Tensor", &ge_t);
  m2.impl("ne.Tensor", &ne_t);
  m2.impl("where.self", &where_);
  m2.impl("cat", &cat_);
  m2.impl("stack", &stack_);
  m2.impl("index.Tensor", &index_);
  m2.impl("index_select", &index_select_);
  m2.impl("bmm", &bmm_);
  m2.impl("native_dropout", &dropout_);
  m2.impl("native_dropout_backward", &dropout_backward_);
  m2.impl("native_layer_norm", &layer_norm_);
  m2.impl("native_layer_norm_backward", &layer_norm_backward_);
  m2.impl("uniform_", &uniform_);
  m2.impl("normal_", &normal_);
  m2.impl("_fused_sgd_", &fused_sgd_);
  m2.impl("_adaptive_avg_pool2d", &adaptive_avg_pool2d_);
  m2.impl("_adaptive_avg_pool2d_backward", &adaptive_avg_pool2d_backward_);
  m2.impl("relu_", &relu_inplace);
}

// ------------------------------------------------------------------------------------------------
// Python module
// ------------------------------------------------------------------------------------------------

PYBIND11_MODULE(_native, m) {
  m.def("register_source", &register_source);
  m.def("set_conv_planner", [](py::function f) { g_conv_planner = f; });
  m.def("tune_dump", []() {  // geometry key -> tile index and split-K multiplier, for the on-disk tuning cache
    std::map<std::string, std::pair<int, double>> r;
    for (auto& kv : g_conv_cache) {
      if (kv.first[0] == 3 || kv.second.tile < 0) continue;
      std::string k;
      for (size_t i = 0; i < kv.first.size(); i++) k += (i ? "," : "") + std::to_string(kv.first[i]);
      r[k] = {kv.second.tile, kv.second.mult};
    }
    return r;
  });
  m.def("tune_load", [](std::map<std::string, std::pair<int, double>> entries) {
    for (auto& e : entries) {
      std::vector<int64_t> key;
      size_t pos = 0; std::string str = e.first;
      while (pos <= str.size()) { size_t nx = str.find(',', pos); if (nx == std::string::npos) nx = str.size(); key.push_back(std::stoll(str.substr(pos, nx - pos))); pos = nx + 1; }
      if (key.size() != 11 || e.second.first >= g_num_tiles) continue;
      Geo g{key[1], key[2], key[3], key[4], key[5], key[6], key[7], key[8], key[9], key[10]};
      ConvPlan p = plan_for((int)key[0], g, e.second.first);
      p.mult = e.second.second;
      g_conv_cache[key] = p;
    }
  });
  m.def("set_mm_planner", [](py::function f) { g_mm_planner = f; });
  m.def("set_fold", [](int f) { g_fold = f; });
  m.def("set_tuning", [](bool on, int num_tiles) { g_tune = on; g_num_tiles = num_tiles; });
  m.def("tuned_plans", []() { std::map<std::string, std::string> r; for (auto& kv : g_conv_cache) { std::string k; for (auto v : kv.first) k += std::to_string(v) + ","; r[k] = std::to_string(kv.second.tile) + " x" + std::to_string(kv.second.mult) + " " + kv.second.kernel; } return r; });
  m.def("init", &init_ctx);
  m.def("sync", &sync_all);
  m.def("flush", &submit_pending);
  m.def("set_queue", [](int chunk, int depth) { g_chunk_max = chunk; g_chunk = std::min(g_chunk, chunk); g_depth = depth; });
  m.def("set_target_backlog_ms", [](double ms) { g_target_backlog_ms = ms; });
  m.def("queue_state", []() { return py::make_tuple(g_chunk, g_chunk_max, g_depth, g_chunk_ms); });
  m.def("cross_entropy", &cross_entropy_fused);
  m.def("empty_cache", &empty_cache);
  m.def("debug_time_kernels", [](bool b) { g_time_each = b; });
  m.def("kernel_times", [](bool reset) { auto r = g_ktime; if (reset) g_ktime.clear(); return r; });
  m.def("debug_skip_submit", [](bool b) { g_skip = b; });
  m.def("kernel_counts", [](bool reset) { auto r = g_kcount; if (reset) g_kcount.clear(); return r; });
  m.def("memory_allocated", []() { return (int64_t)g_inuse; });
  m.def("memory_reserved", []() { return (int64_t)g_reserved; });
  m.def("fallback_counts", []() { return g_fallbacks; });
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
    return build_plan(L, shape, train, input_grad, params, rms, rvs);
  });
  m.def("adapter_name", []() { init_ctx(); WGPUAdapterInfo info = {}; wgpuAdapterGetInfo(C.adapter, &info);
    return std::string(info.device.data ? info.device.data : "", info.device.length); });
}

}  // namespace omt
