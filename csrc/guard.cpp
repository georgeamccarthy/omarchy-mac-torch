// The only C++ in omarchy-mac-torch: a device guard for PyTorch's PrivateUse1 device
// ("omarchy_mac_torch"). PyTorch asks it which device and stream are current (autograd,
// .to(), factories). There is one device and one in-order stream, so it is
// all no-ops; the tensors and kernels themselves live in Python.

#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <ATen/detail/PrivateUse1HooksInterface.h>
#include <torch/extension.h>

namespace {

struct GuardImpl final : c10::impl::DeviceGuardImplInterface {
  static constexpr c10::DeviceType kType = c10::DeviceType::PrivateUse1;

  c10::DeviceType type() const override { return kType; }
  c10::Device exchangeDevice(c10::Device) const override { return {kType, 0}; }
  c10::Device getDevice() const override { return {kType, 0}; }
  void setDevice(c10::Device) const override {}
  void uncheckedSetDevice(c10::Device) const noexcept override {}
  c10::Stream getStream(c10::Device d) const noexcept override {
    return c10::Stream(c10::Stream::DEFAULT, d);
  }
  c10::Stream getDefaultStream(c10::Device d) const override {
    return c10::Stream(c10::Stream::DEFAULT, d);
  }
  c10::Stream exchangeStream(c10::Stream s) const noexcept override { return s; }
  c10::DeviceIndex deviceCount() const noexcept override { return 1; }

  // Everything runs in order on one stream, so events are trivial.
  void record(void**, const c10::Stream&, const c10::DeviceIndex,
              const c10::EventFlag) const override {}
  void block(void*, const c10::Stream&) const override {}
  bool queryEvent(void*) const override { return true; }
  void destroyEvent(void*, const c10::DeviceIndex) const noexcept override {}
  bool queryStream(const c10::Stream&) const override { return true; }
  void synchronizeStream(const c10::Stream&) const override {}
  void synchronizeDevice(const c10::DeviceIndex) const override {}
};

C10_REGISTER_GUARD_IMPL(PrivateUse1, GuardImpl);

// Autograd's engine asks the accelerator hooks about the device.
struct Hooks final : at::PrivateUse1HooksInterface {
  bool isBuilt() const override { return true; }
  bool isAvailable() const override { return true; }
  bool hasPrimaryContext(c10::DeviceIndex) const override { return true; }
};

struct HooksRegistrar {
  HooksRegistrar() {
    static Hooks hooks;
    at::RegisterPrivateUse1HooksInterface(&hooks);
  }
} registrar;

}  // namespace

bool registered() { return true; }
