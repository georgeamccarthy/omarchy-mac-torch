// wgpu context, buffer pool, kernel registry, command recorder and queue, transfers, errors.
#include "omt.h"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <deque>
#include <list>
#include <thread>
#include <unordered_map>

namespace omt {

std::recursive_mutex& big_lock() {
  static std::recursive_mutex m;
  return m;
}

static WGPUStringView sv(const char* s) { return WGPUStringView{s, s ? std::strlen(s) : 0}; }
static std::string str(WGPUStringView s) { return s.data ? std::string(s.data, s.length == WGPU_STRLEN ? std::strlen(s.data) : s.length) : ""; }

// ------------------------------------------------------------------------------------------------
// context and errors
// ------------------------------------------------------------------------------------------------

struct Ctx {
  WGPUInstance instance = nullptr;
  WGPUAdapter adapter = nullptr;
  WGPUDevice device = nullptr;
  WGPUQueue queue = nullptr;
  bool ready = false;
};
static Ctx C;
static std::mutex g_err_mu;
static std::string g_err;            // first asynchronous error since the last check
static std::atomic<bool> g_lost{false};
static std::string g_lost_reason;

WGPUDevice device() { init_ctx(); return C.device; }
WGPUQueue queue() { init_ctx(); return C.queue; }
WGPUAdapter adapter() { init_ctx(); return C.adapter; }

RuntimeStats& stats() {
  static RuntimeStats s;
  return s;
}

static void process_events() {
  if (C.device) wgpuDevicePoll(C.device, 0, nullptr);
  wgpuInstanceProcessEvents(C.instance);
}
// wait for a callback that sets *flag (adapter/device requests, map, error scopes): these complete quickly
static void wait_until(const std::atomic<bool>& flag) {
  while (!flag) {
    process_events();
    if (!flag) std::this_thread::sleep_for(std::chrono::microseconds(10));
  }
}

static void on_uncaptured(WGPUDevice const*, WGPUErrorType type, WGPUStringView msg, void*, void*) {
  std::string m = str(msg);
  fprintf(stderr, "[omarchy-mac-torch] wgpu error (%d): %s\n", (int)type, m.c_str());
  std::lock_guard<std::mutex> g(g_err_mu);
  if (g_err.empty()) g_err = m;
}
static void on_lost(WGPUDevice const*, WGPUDeviceLostReason reason, WGPUStringView msg, void*, void*) {
  if (reason == WGPUDeviceLostReason_Destroyed || reason == WGPUDeviceLostReason_CallbackCancelled) return;  // shutdown
  g_lost_reason = str(msg);
  g_lost = true;
  fprintf(stderr, "[omarchy-mac-torch] GPU device lost: %s\n", g_lost_reason.c_str());
}

void check_errors() {
  TORCH_CHECK(!g_lost, "omarchy-mac-torch: the GPU device was lost (", g_lost_reason, "); restart the process");
  std::string e;
  {
    std::lock_guard<std::mutex> g(g_err_mu);
    e.swap(g_err);
  }
  TORCH_CHECK(e.empty(), "omarchy-mac-torch: GPU error: ", e);
}

void init_ctx() {
  if (C.ready) return;
  OMT_LOCK;
  if (C.ready) return;
  WGPUInstanceDescriptor idesc = {};
  C.instance = wgpuCreateInstance(&idesc);
  WGPURequestAdapterOptions opts = {};
  opts.powerPreference = WGPUPowerPreference_HighPerformance;
  // CI runs on Mesa's software Vulkan driver (lavapipe), which wgpu reports as a fallback adapter
  const char* fb = std::getenv("OMARCHY_MAC_TORCH_FALLBACK_ADAPTER");
  opts.forceFallbackAdapter = fb && fb[0] == '1';
  std::atomic<bool> af{false};
  WGPURequestAdapterCallbackInfo ci = {};
  ci.mode = WGPUCallbackMode_AllowSpontaneous;
  ci.callback = [](WGPURequestAdapterStatus st, WGPUAdapter a, WGPUStringView, void* u1, void* u2) {
    if (st == WGPURequestAdapterStatus_Success) *(WGPUAdapter*)u1 = a;
    *(std::atomic<bool>*)u2 = true;
  };
  ci.userdata1 = &C.adapter;
  ci.userdata2 = &af;
  wgpuInstanceRequestAdapter(C.instance, &opts, ci);
  wait_until(af);
  TORCH_CHECK(C.adapter, "omarchy-mac-torch: no GPU adapter (is the Honeykrisp Vulkan driver installed? check vulkaninfo)");
  WGPULimits lim = {};
  wgpuAdapterGetLimits(C.adapter, &lim);
  WGPUDeviceDescriptor dd = {};
  dd.requiredLimits = &lim;
  dd.uncapturedErrorCallbackInfo.callback = on_uncaptured;
  dd.deviceLostCallbackInfo.mode = WGPUCallbackMode_AllowSpontaneous;
  dd.deviceLostCallbackInfo.callback = on_lost;
  std::atomic<bool> df{false};
  WGPURequestDeviceCallbackInfo dci = {};
  dci.mode = WGPUCallbackMode_AllowSpontaneous;
  dci.callback = [](WGPURequestDeviceStatus st, WGPUDevice d, WGPUStringView, void* u1, void* u2) {
    if (st == WGPURequestDeviceStatus_Success) *(WGPUDevice*)u1 = d;
    *(std::atomic<bool>*)u2 = true;
  };
  dci.userdata1 = &C.device;
  dci.userdata2 = &df;
  wgpuAdapterRequestDevice(C.adapter, &dd, dci);
  wait_until(df);
  TORCH_CHECK(C.device, "omarchy-mac-torch: could not create the GPU device");
  C.queue = wgpuDeviceGetQueue(C.device);
  C.ready = true;
}

std::string adapter_name() {
  init_ctx();
  WGPUAdapterInfo info = {};
  wgpuAdapterGetInfo(C.adapter, &info);
  return str(info.device);
}

// error scopes, used by debug mode and allocation
static void push_scopes() {
  wgpuDevicePushErrorScope(C.device, WGPUErrorFilter_OutOfMemory);
  wgpuDevicePushErrorScope(C.device, WGPUErrorFilter_Validation);
}
static std::string pop_scope() {
  struct R { std::atomic<bool> done{false}; std::string msg; } r;
  WGPUPopErrorScopeCallbackInfo pi = {};
  pi.mode = WGPUCallbackMode_AllowSpontaneous;
  pi.callback = [](WGPUPopErrorScopeStatus, WGPUErrorType type, WGPUStringView msg, void* u1, void*) {
    R* r = (R*)u1;
    if (type != WGPUErrorType_NoError) r->msg = str(msg);
    r->done = true;
  };
  pi.userdata1 = &r;
  wgpuDevicePopErrorScope(C.device, pi);
  wait_until(r.done);
  return r.msg;
}
static std::string pop_scopes() {
  std::string m;
  for (int i = 0; i < 2; i++) {  // validation, out of memory (wgpu-native has no internal-error filter)
    std::string e = pop_scope();
    if (m.empty()) m = e;
  }
  return m;
}
void debug_begin() {
  init_ctx();
  push_scopes();
}
void debug_end(const char* op) {
  sync_all();  // execution errors surface at the op that caused them
  std::string e = pop_scopes();
  TORCH_CHECK(e.empty(), "omarchy-mac-torch: GPU error in ", op, ": ", e);
}

// ------------------------------------------------------------------------------------------------
// buffer pool
// ------------------------------------------------------------------------------------------------

static std::unordered_map<size_t, std::vector<Buf*>> g_free;
static size_t g_reserved = 0, g_inuse = 0;
static std::mutex g_pool_mu;
static std::atomic<uint64_t> g_next_id{1};

// size classes: powers of two up to 1 MiB, then eighths of a power of two (at most 12.5% waste for big tensors)
static size_t size_class(size_t bytes) {
  size_t p = 256;
  while (p < bytes) p <<= 1;
  if (p <= (1u << 20)) return p;
  size_t step = p / 8;
  return (bytes + step - 1) / step * step;
}

static WGPUBuffer create_buffer(size_t bytes, WGPUBufferUsage usage) {
  WGPUBufferDescriptor d = {};
  d.usage = usage;
  d.size = bytes;
  wgpuDevicePushErrorScope(C.device, WGPUErrorFilter_OutOfMemory);
  WGPUBuffer b = wgpuDeviceCreateBuffer(C.device, &d);
  std::string e = pop_scope();
  if (!e.empty()) {
    wgpuBufferRelease(b);
    return nullptr;
  }
  return b;
}

Buf* alloc_buf(size_t bytes) {
  init_ctx();
  size_t cls = size_class(std::max<size_t>(bytes, 4));
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
  auto usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc | WGPUBufferUsage_CopyDst;
  WGPUBuffer wb = create_buffer(cls, usage);
  if (!wb) {  // out of memory: return cached buffers to the driver and retry once
    empty_cache();
    wb = create_buffer(cls, usage);
  }
  TORCH_CHECK_WITH(OutOfMemoryError, wb, "omarchy-mac-torch: out of GPU memory allocating ", cls, " bytes (",
                   g_inuse, " in use, ", g_reserved, " reserved)");
  Buf* b = new Buf{wb, cls, g_next_id++, false, 0};
  std::lock_guard<std::mutex> g(g_pool_mu);
  g_reserved += cls;
  g_inuse += cls;
  return b;
}

// Freed buffers go straight back to the pool. That's safe although queued work may still use them: the queue
// executes in order, so whatever the next owner records runs after the previous owner's work.
void free_buf(void* p) {
  Buf* b = (Buf*)p;
  std::lock_guard<std::mutex> g(g_pool_mu);
  g_inuse -= b->bytes;
  g_free[b->bytes].push_back(b);
}

size_t memory_allocated() { std::lock_guard<std::mutex> g(g_pool_mu); return g_inuse; }
size_t memory_reserved() { std::lock_guard<std::mutex> g(g_pool_mu); return g_reserved; }

Buf* dummy_buf() {
  static Buf* d = nullptr;
  if (!d) d = alloc_buf(256);
  return d;
}

// ------------------------------------------------------------------------------------------------
// kernel registry and pipelines
// ------------------------------------------------------------------------------------------------

struct Pipe {
  WGPUComputePipeline p;
  WGPUBindGroupLayout layout;
  std::vector<int> rw, ro;  // storage bindings declared read_write / read
};
static std::vector<Pipe> g_pipes;
static std::unordered_map<std::string, int> g_pipe_ids;
static std::unordered_map<std::string, std::string>& sources() {
  static std::unordered_map<std::string, std::string> s;
  return s;
}

void register_source(const std::string& name, const std::string& wgsl) { sources()[name] = wgsl; }

// kernels shipped as csrc/kernels/*.wgsl, embedded by tools/build_native.py
struct EmbeddedKernel { const char* name; const char* src; };
static const EmbeddedKernel k_embedded[] = {
#include "kernels_gen.inc"
};
static struct RegisterEmbedded {
  RegisterEmbedded() {
    for (auto& k : k_embedded) register_source(k.name, k.src);
  }
} g_register_embedded;

static void scan_bindings(const std::string& wgsl, Pipe& p) {
  size_t pos = 0;
  while ((pos = wgsl.find("@binding(", pos)) != std::string::npos) {
    int b = std::atoi(wgsl.c_str() + pos + 9);
    size_t eol = wgsl.find(';', pos);
    std::string decl = wgsl.substr(pos, eol - pos);
    if (decl.find("read_write") != std::string::npos) p.rw.push_back(b);
    else if (decl.find("var<storage") != std::string::npos) p.ro.push_back(b);
    pos = eol;
  }
}

static std::string compile(const std::string& wgsl, WGPUComputePipeline* out) {
  WGPUShaderSourceWGSL src = {};
  src.chain.sType = WGPUSType_ShaderSourceWGSL;
  src.code = WGPUStringView{wgsl.data(), wgsl.size()};
  WGPUShaderModuleDescriptor md = {};
  md.nextInChain = &src.chain;
  push_scopes();
  WGPUShaderModule mod = wgpuDeviceCreateShaderModule(C.device, &md);
  WGPUComputePipelineDescriptor pd = {};
  pd.compute.module = mod;
  pd.compute.entryPoint = sv("main");
  WGPUComputePipeline p = wgpuDeviceCreateComputePipeline(C.device, &pd);
  std::string e = pop_scopes();
  wgpuShaderModuleRelease(mod);
  *out = p;
  return e;
}

static int pipe_id(const std::string& name) {
  auto it = g_pipe_ids.find(name);
  if (it != g_pipe_ids.end()) return it->second;
  auto s = sources().find(name);
  TORCH_CHECK(s != sources().end(), "omarchy-mac-torch: unknown kernel ", name);
  init_ctx();
  Pipe pp;
  std::string e = compile(s->second, &pp.p);
  TORCH_CHECK(e.empty(), "omarchy-mac-torch: kernel ", name, " failed to compile: ", e);
  pp.layout = wgpuComputePipelineGetBindGroupLayout(pp.p, 0);
  scan_bindings(s->second, pp);
  g_pipes.push_back(pp);
  int id = (int)g_pipes.size() - 1;
  g_pipe_ids[name] = id;
  return id;
}

std::vector<std::string> validate_kernels() {
  OMT_LOCK;
  init_ctx();
  std::vector<std::string> bad;
  for (auto& kv : sources()) {
    WGPUComputePipeline p;
    std::string e = compile(kv.second, &p);
    if (p) wgpuComputePipelineRelease(p);
    if (!e.empty()) bad.push_back(kv.first + ": " + e);
  }
  return bad;
}

// ------------------------------------------------------------------------------------------------
// LRU caches: uniform buffers (keyed by contents) and bind groups (keyed by pipeline + buffer ids/ranges)
// ------------------------------------------------------------------------------------------------

struct VecHash {
  template <class T>
  size_t operator()(const std::vector<T>& v) const {
    size_t h = 1469598103934665603ull;
    for (auto x : v) h = (h ^ (size_t)x) * 1099511628211ull;
    return h;
  }
};

template <class K, class V>
struct Lru {
  size_t cap;
  std::list<std::pair<K, V>> items;
  std::unordered_map<K, typename std::list<std::pair<K, V>>::iterator, VecHash> index;
  explicit Lru(size_t c) : cap(c) {}
  V* get(const K& k) {
    auto it = index.find(k);
    if (it == index.end()) return nullptr;
    items.splice(items.begin(), items, it->second);
    return &it->second->second;
  }
  void put(const K& k, V v) {
    items.emplace_front(k, v);
    index[k] = items.begin();
  }
};

static Lru<std::vector<uint32_t>, Buf*> g_uniforms(8192);
static Lru<std::vector<uint64_t>, WGPUBindGroup> g_groups(8192);

Buf* uniform(const std::vector<uint32_t>& words) {
  if (Buf** b = g_uniforms.get(words)) return *b;
  init_ctx();
  std::vector<uint32_t> w = words;
  while (w.size() % 4) w.push_back(0);
  WGPUBufferDescriptor d = {};
  d.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
  d.size = w.size() * 4;
  Buf* b = new Buf{wgpuDeviceCreateBuffer(C.device, &d), (size_t)d.size, g_next_id++, true, 0};
  wgpuQueueWriteBuffer(C.queue, b->b, 0, w.data(), w.size() * 4);
  g_uniforms.put(words, b);
  // evict from the cold end; bind groups that use an evicted buffer keep it alive inside wgpu until they go
  for (auto it = std::prev(g_uniforms.items.end()); g_uniforms.items.size() > g_uniforms.cap && it != g_uniforms.items.begin();) {
    auto cur = it--;
    if (cur->second->pins > 0) continue;
    wgpuBufferRelease(cur->second->b);
    delete cur->second;
    g_uniforms.index.erase(cur->first);
    g_uniforms.items.erase(cur);
  }
  return b;
}
void pin(Buf* b) { if (b && b->uniform) b->pins++; }
void unpin(Buf* b) { if (b && b->uniform) b->pins--; }

static void trim_groups() {
  while (g_groups.items.size() > g_groups.cap) {
    auto& last = g_groups.items.back();
    wgpuBindGroupRelease(last.second);  // recorded dispatches hold their own reference
    g_groups.index.erase(last.first);
    g_groups.items.pop_back();
  }
}
static void release_groups() {
  for (auto& kv : g_groups.items) wgpuBindGroupRelease(kv.second);
  g_groups.items.clear();
  g_groups.index.clear();
}

// ------------------------------------------------------------------------------------------------
// recorder and queue
// ------------------------------------------------------------------------------------------------

struct Cmd {
  int pipe;  // -1: buffer copy
  WGPUBindGroup group;
  uint32_t x, y, z;
  WGPUBuffer src, dst;
  uint64_t so, dO, bytes;
};
static std::vector<Cmd> g_pending;
static int g_chunk = 16, g_chunk_max = 64, g_depth = 3;
static double g_target_backlog_ms = 8.0;  // keep about this much GPU work queued, so the compositor never waits long
static double g_chunk_ms = 0.0;           // EMA of GPU time per chunk, measured while the GPU is the bottleneck
static auto g_last_submit = std::chrono::steady_clock::now();
static std::deque<uint64_t> g_inflight;   // submission indices, oldest first
static std::atomic<uint64_t> g_completed{0};
static uint64_t g_last_index = 0;

void set_queue(int chunk, int depth) { OMT_LOCK; g_chunk_max = chunk; g_chunk = std::min(g_chunk, chunk); g_depth = depth; }
void set_target_backlog_ms(double ms) { g_target_backlog_ms = ms; }
std::tuple<int, int, int, double> queue_state() { return {g_chunk, g_chunk_max, g_depth, g_chunk_ms}; }

uint64_t last_submission() { return g_last_index; }
bool submission_done(uint64_t idx) {
  if (idx == 0 || g_completed >= idx) return true;
  process_events();
  return g_completed >= idx;
}
void wait_submission(uint64_t idx) {
  while (idx && g_completed < idx) {
    WGPUSubmissionIndex si = idx;
    wgpuDevicePoll(C.device, 1, &si);  // blocks in the driver until that submission finishes
    process_events();
  }
}

void submit_pending() {
  if (g_pending.empty()) return;
  auto& st = stats();
  if (st.skip) {
    for (auto& c : g_pending) if (c.group) wgpuBindGroupRelease(c.group);
    g_pending.clear();
    return;
  }
  WGPUCommandEncoderDescriptor ed = {};
  WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(C.device, &ed);
  WGPUComputePassEncoder cp = nullptr;
  int last = -1;
  for (auto& c : g_pending) {
    if (c.pipe < 0) {
      if (cp) { wgpuComputePassEncoderEnd(cp); wgpuComputePassEncoderRelease(cp); cp = nullptr; last = -1; }
      wgpuCommandEncoderCopyBufferToBuffer(enc, c.src, c.so, c.dst, c.dO, c.bytes);
      continue;
    }
    if (!cp) {
      WGPUComputePassDescriptor pd = {};
      cp = wgpuCommandEncoderBeginComputePass(enc, &pd);
    }
    if (c.pipe != last) {
      wgpuComputePassEncoderSetPipeline(cp, g_pipes[c.pipe].p);
      last = c.pipe;
    }
    wgpuComputePassEncoderSetBindGroup(cp, 0, c.group, 0, nullptr);
    wgpuComputePassEncoderDispatchWorkgroups(cp, c.x, c.y, c.z);
  }
  if (cp) { wgpuComputePassEncoderEnd(cp); wgpuComputePassEncoderRelease(cp); }
  WGPUCommandBufferDescriptor cd = {};
  WGPUCommandBuffer cb = wgpuCommandEncoderFinish(enc, &cd);
  for (auto& c : g_pending) if (c.group) wgpuBindGroupRelease(c.group);
  g_pending.clear();
  bool blocked = false;
  while ((int)g_inflight.size() >= g_depth) {  // wait for the oldest chunk only; newer ones keep the GPU busy
    auto w0 = std::chrono::steady_clock::now();
    wait_submission(g_inflight.front());
    g_inflight.pop_front();
    if (std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - w0).count() > 0.2) blocked = true;
  }
  while (!g_inflight.empty() && g_completed >= g_inflight.front()) g_inflight.pop_front();
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
  uint64_t idx = wgpuQueueSubmitForIndex(C.queue, 1, &cb);
  g_last_index = idx;
  WGPUQueueWorkDoneCallbackInfo wi = {};
  wi.mode = WGPUCallbackMode_AllowSpontaneous;
  wi.callback = [](WGPUQueueWorkDoneStatus, WGPUStringView, void* u1, void*) {
    uint64_t i = (uint64_t)(uintptr_t)u1;
    uint64_t cur = g_completed.load();
    while (cur < i && !g_completed.compare_exchange_weak(cur, i)) {}
  };
  wi.userdata1 = (void*)(uintptr_t)idx;
  wgpuQueueOnSubmittedWorkDone(C.queue, wi);
  g_inflight.push_back(idx);
  wgpuCommandEncoderRelease(enc);
  wgpuCommandBufferRelease(cb);
}

void sync_all() {
  OMT_LOCK;
  if (!C.ready) return;
  submit_pending();
  wait_submission(g_last_index);
  g_inflight.clear();
  check_errors();
}

void grid(int64_t total, int wg, uint32_t& gx, uint32_t& gy) {
  int64_t groups = std::max<int64_t>((total + wg - 1) / wg, 1);
  gx = (uint32_t)std::min<int64_t>(groups, 65535);
  gy = (uint32_t)((groups + gx - 1) / gx);
}

void record_copy(Buf* src, uint64_t so, Buf* dst, uint64_t dO, uint64_t bytes) {
  if (!bytes) return;
  TORCH_INTERNAL_ASSERT(so % 4 == 0 && dO % 4 == 0 && bytes % 4 == 0, "unaligned buffer copy");
  g_pending.push_back({-1, nullptr, 0, 0, 0, src->b, dst->b, so, dO, bytes});
  if ((int)g_pending.size() >= g_chunk) submit_pending();
}

static WGPUBindGroup bind_group(int p, const std::vector<Binding>& b) {
  std::vector<uint64_t> key;
  key.reserve(1 + 3 * b.size());
  key.push_back((uint64_t)p);
  for (auto& e : b) {
    key.push_back(e.buf->id);
    key.push_back(e.off);
    key.push_back(e.size);
  }
  if (WGPUBindGroup* g = g_groups.get(key)) return *g;
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
  WGPUBindGroup grp = wgpuDeviceCreateBindGroup(C.device, &gd);
  g_groups.put(key, grp);
  trim_groups();
  return grp;
}

void launch(const char* kernel, const std::vector<Binding>& b_in, uint32_t x, uint32_t y, uint32_t z) {
  TORCH_CHECK(!g_lost, "omarchy-mac-torch: the GPU device was lost (", g_lost_reason, ")");
  int p = pipe_id(kernel);
  auto& st = stats();
  st.kcount[kernel]++;
  // WebGPU rejects a dispatch that binds one buffer both read-only and read-write. When that happens (an op whose
  // input and output are views of the same storage) the read-only range is copied to a scratch buffer first.
  const Pipe& pp = g_pipes[p];
  const std::vector<Binding>* bp = &b_in;
  std::vector<Binding> fixed;
  std::vector<Buf*> scratch;
  for (int r : pp.ro) {
    if (r >= (int)b_in.size()) continue;
    for (int w : pp.rw) {
      if (w >= (int)b_in.size() || b_in[r].buf != b_in[w].buf) continue;
      if (fixed.empty()) { fixed = b_in; bp = &fixed; }
      uint64_t lo = fixed[r].off & ~255ull, hi = fixed[r].off + fixed[r].size;
      Buf* s = alloc_buf(hi - lo);
      record_copy(fixed[r].buf, lo, s, 0, (hi - lo + 3) / 4 * 4);
      fixed[r] = {s, fixed[r].off - lo, fixed[r].size};
      scratch.push_back(s);
      break;
    }
  }
  WGPUBindGroup grp = bind_group(p, *bp);
  for (Buf* s : scratch) free_buf(s);  // in-order queue: safe to recycle now
  wgpuBindGroupAddRef(grp);            // the pending list owns a reference until it is encoded
  g_pending.push_back({p, grp, x, y, z, nullptr, nullptr, 0, 0, 0});
  if (st.time_each) {
    submit_pending();
    sync_all();
    auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < 6; r++) {  // time = (6 runs) / 6
      wgpuBindGroupAddRef(grp);
      g_pending.push_back({p, grp, x, y, z, nullptr, nullptr, 0, 0, 0});
    }
    submit_pending();
    sync_all();
    st.ktime[kernel] += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / 6.0;
    return;
  }
  if ((int)g_pending.size() >= g_chunk) submit_pending();
}

// ------------------------------------------------------------------------------------------------
// transfers
// ------------------------------------------------------------------------------------------------

// The write is ordered after all previously submitted work (the WebGPU queue is in order), so recorded work only has
// to be submitted, not waited for: the CPU keeps running ahead of the GPU across uploads.
void upload_bytes(Buf* dst, uint64_t off, const void* src, size_t bytes) {
  if (!bytes) return;
  init_ctx();
  submit_pending();
  if (off % 4 == 0 && bytes % 4 == 0) {
    wgpuQueueWriteBuffer(C.queue, dst->b, off, src, bytes);
    return;
  }
  uint64_t lo = off & ~3ull, hi = (off + bytes + 3) & ~3ull;  // read-modify-write the partial words at either end
  std::vector<uint8_t> tmp(hi - lo);
  if (lo != off) download_bytes(dst, lo, tmp.data(), 4);
  if (hi != off + bytes) download_bytes(dst, hi - 4, tmp.data() + tmp.size() - 4, 4);
  std::memcpy(tmp.data() + (off - lo), src, bytes);
  wgpuQueueWriteBuffer(C.queue, dst->b, lo, tmp.data(), tmp.size());
}

void download_bytes(Buf* src, uint64_t off, void* dst, size_t bytes) {
  if (!bytes) return;
  init_ctx();
  submit_pending();
  static WGPUBuffer stage = nullptr;
  static size_t stage_bytes = 0;
  uint64_t lo = off & ~3ull, hi = (off + bytes + 3) & ~3ull;
  size_t need = hi - lo;
  if (!stage || stage_bytes < need) {
    if (stage) { wgpuBufferDestroy(stage); wgpuBufferRelease(stage); }
    WGPUBufferDescriptor d = {};
    d.usage = WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst;
    d.size = size_class(need);
    stage = wgpuDeviceCreateBuffer(C.device, &d);
    stage_bytes = d.size;
  }
  WGPUCommandEncoderDescriptor ed = {};
  WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(C.device, &ed);
  wgpuCommandEncoderCopyBufferToBuffer(enc, src->b, lo, stage, 0, need);
  WGPUCommandBufferDescriptor cd = {};
  WGPUCommandBuffer cb = wgpuCommandEncoderFinish(enc, &cd);
  g_last_index = wgpuQueueSubmitForIndex(C.queue, 1, &cb);
  std::atomic<bool> mapped{false};
  WGPUBufferMapCallbackInfo mi = {};
  mi.mode = WGPUCallbackMode_AllowSpontaneous;
  mi.callback = [](WGPUMapAsyncStatus, WGPUStringView, void* u1, void*) { *(std::atomic<bool>*)u1 = true; };
  mi.userdata1 = &mapped;
  wgpuBufferMapAsync(stage, WGPUMapMode_Read, 0, need, mi);
  WGPUSubmissionIndex si = g_last_index;
  wgpuDevicePoll(C.device, 1, &si);
  wait_until(mapped);
  uint64_t c = g_completed.load();  // everything up to this submission has finished
  while (c < g_last_index && !g_completed.compare_exchange_weak(c, g_last_index)) {}
  g_inflight.clear();
  const uint8_t* p = (const uint8_t*)wgpuBufferGetConstMappedRange(stage, 0, need);
  std::memcpy(dst, p + (off - lo), bytes);
  wgpuBufferUnmap(stage);
  wgpuCommandEncoderRelease(enc);
  wgpuCommandBufferRelease(cb);
  check_errors();
}

void empty_cache() {
  OMT_LOCK;
  sync_all();
  release_groups();  // bind groups may reference the buffers destroyed below
  std::lock_guard<std::mutex> g(g_pool_mu);
  for (auto& kv : g_free)
    for (Buf* b : kv.second) {
      g_reserved -= b->bytes;
      wgpuBufferDestroy(b->b);
      wgpuBufferRelease(b->b);
      delete b;
    }
  g_free.clear();
}

}  // namespace omt
