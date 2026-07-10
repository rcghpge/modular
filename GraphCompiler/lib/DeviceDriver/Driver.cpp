//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/DeviceDriver/Driver.h"
#include "AsyncRT/Runtime/AnyAsyncValueRef.h"
#include "GraphCompiler/DeviceDriver/Utils.h"
#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "Support/AssertStream.h"
#include "Support/MArchTarget/MArchTargetMinimal.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <cstdint>
#include <cstring>

using namespace M::GraphCompiler::Driver;
using namespace M;

// Evaluate EXPR. If the returned value is an error, return it.
// Otherwise, assign to a new variable named VAR.
#define ASSIGN_OR_RETURN(VAR, EXPR)                                            \
  auto VAR##OrErr = EXPR;                                                      \
  if (VAR##OrErr.isError())                                                    \
    return VAR##OrErr.takeError();                                             \
  auto VAR = VAR##OrErr.takeValue()

MODULAR_DRIVER_EXPORT
ErrorOrSuccess DeviceMemory::copyTo(DeviceMemory &dst) const {
  return this->buf->copyTo(*dst.buf);
}

MODULAR_DRIVER_EXPORT ErrorOr<DeviceMemory *>
DeviceMemory::createViewOn(Device &device) const {
  ASSIGN_OR_RETURN(newBuf, buf->createSubBuffer(0, buf->byteSize()));
  if (!device.isHost()) {
    ErrorOrSuccess err =
        newBuf->reassignOwnershipTo(*device.tmpGetDeviceContext());
    if (err.isError())
      return err.takeError();
  }
  return new DeviceMemory(M::copyRCRef(&device), newBuf, spec);
}

MODULAR_DRIVER_EXPORT ErrorOr<DeviceMemory *>
DeviceMemory::createView(const TensorSpec &spec, size_t byteOffset) const {
  if ((spec.getSizeInBytes() + byteOffset) > this->spec.getSizeInBytes())
    RETURN_ERROR("Can't create a view larger than the initial buffer");

  // If the requested view is offsetted or has a different size, create a
  // sub-buffer
  if (byteOffset || spec.getSizeInBytes() != this->spec.getSizeInBytes()) {
    ASSIGN_OR_RETURN(newBuf,
                     buf->createSubBuffer(byteOffset, spec.getSizeInBytes()));
    return new DeviceMemory(device, newBuf, spec);
  }

  return new DeviceMemory(device, buf, spec);
}

MODULAR_DRIVER_EXPORT ErrorOrSuccess DeviceMemory::memset(int value) {
  return buf->context().enqueueSetMemory(*buf, value, 1);
}

MODULAR_DRIVER_EXPORT void M_destroyDevice(Device *devPtr) {
  (void)M::takeRCRef(devPtr);
}

MODULAR_DRIVER_EXPORT const char *M_getDeviceDesc(const Device *devPtr) {
  return devPtr->getString().data();
}

MODULAR_DRIVER_EXPORT std::string Device::toString() const {
  std::string str;
  llvm::raw_string_ostream oss(str);
  oss << "Device(type=" << deviceSpecification.ref.label << ",";
  oss << "id=" << deviceSpecification.ref.id << ")";
  return str;
}

MODULAR_DRIVER_EXPORT ErrorOr<std::string> Device::getStats() {
  ASSIGN_OR_RETURN(info, ctx->getMemoryInfo());

  // Crafting JSON string
  std::string s;
  llvm::raw_string_ostream os(s);
  llvm::json::OStream j(os);

  j.object([&] {
    j.attribute("free_memory", info.free);
    j.attribute("total_memory", info.total);
    auto graphMemResult = ctx->device().getGraphMemoryInfo();
    if (!graphMemResult.isError()) {
      auto graphMem = graphMemResult.takeValue();
      j.attribute("graph_mem_reserved", graphMem.reserved);
      j.attribute("graph_mem_used", graphMem.used);
    }
  });

  os.flush();
  return s;
}

MODULAR_DRIVER_EXPORT ErrorOr<std::string> Device::getApi() {
  return ctx->device().api().str();
}

MODULAR_DRIVER_EXPORT ErrorOr<size_t> Device::getMaxSingleAllocationSize() {
  return ctx->device().maxSingleAllocationSize();
}

MODULAR_DRIVER_EXPORT ErrorOr<std::string> Device::getArchName() {
  return ctx->device().getArchName();
}

MODULAR_DRIVER_EXPORT ErrorOr<std::string> Device::getModelName() {
  return ctx->device().name();
}

MODULAR_DRIVER_EXPORT ErrorOrSuccess Device::isCompatible() const {
  return ctx->device().isCompatible();
}

MODULAR_DRIVER_EXPORT
ErrorOr<M::Driver::DeviceBufferRef>
Device::memmapFile(const std::filesystem::path &filename) {
  return ctx->memmapFile(filename);
}

MODULAR_DRIVER_EXPORT ErrorOr<DeviceMemory *>
Device::createDeviceMemory(const TensorSpec &spec, bool hostAlloc) {
  ASSIGN_OR_RETURN(buf, hostAlloc
                            ? ctx->createHostBuffer(spec.getSizeInBytes())
                            : ctx->enqueueCreateBuffer(spec.getSizeInBytes()));

  // Debug allocation poisoning: fills newly allocated memory with default
  // patterns to help detect use of uninitialized memory. Floats are filled
  // with the bit pattern of the largest finite value (e.g. FLT_MAX); this is
  // non-NaN/non-Inf so it doesn't conflict with the nan-check pass when EP
  // kernels leave allocation padding un-overwritten. Integers are filled
  // with 0xCD. Host and device fills use the same per-type pattern so the
  // Mojo stdlib uninit-read check has a single value to look for.
  // Enable via: MODULAR_DEBUG_DEVICE_ALLOCATOR=uninitialized-poison
  if (ctx->isDebugAllocatorEnabled("uninitialized-poison")) {
    DType dtype = spec.getEltType();
    size_t totalBytes = spec.getSizeInBytes();

    uint64_t bits;
    size_t patternBytes;
    const char *patternKind;
    if (const llvm::fltSemantics *sem = dtype.getFloatSemantics()) {
      if (llvm::APFloat::getSizeInBits(*sem) < 8) {
        // Sub-byte float types (e.g. float4_e2m1fn): the byte-granularity
        // fill below can't write a single 4-bit lane, and the Mojo poison
        // check excludes these dtypes too. Skip the fill so host and check
        // contracts stay in agreement.
        llvm::errs() << "DEBUG_ALLOC_POISON: skipping " << dtype
                     << " allocation (" << totalBytes
                     << " bytes; sub-byte float type)\n";
        return new DeviceMemory(M::copyRCRef(this), buf, spec);
      }
      // Skip the fill for fp8 formats. Two reasons:
      // - `float8_e3m4` and `float8_e8m0fnu` are scale-only / no-mantissa
      //   formats for which `max_finite` is not defined.
      // - The remaining fp8 formats have such a narrow dynamic range that
      //   the `max_finite` sentinel collides with the saturate-to-max
      //   value produced by legitimate quantization, generating pervasive
      //   false-positive uninit-read aborts.
      // The Mojo poison-check set is kept in agreement with this exclusion.
      if (llvm::is_contained(
              std::initializer_list<DType>{DType::f8e3m4, DType::f8e8m0fnu,
                                           DType::f8e4m3fn, DType::f8e4m3fnuz,
                                           DType::f8e5m2, DType::f8e5m2fnuz},
              dtype)) {
        llvm::errs() << "DEBUG_ALLOC_POISON: skipping " << dtype
                     << " allocation (" << totalBytes
                     << " bytes; not in Mojo poison-check set)\n";
        return new DeviceMemory(M::copyRCRef(this), buf, spec);
      }
      llvm::APFloat largest = llvm::APFloat::getLargest(*sem);
      bits = largest.bitcastToAPInt().getZExtValue();
      patternBytes = llvm::APFloat::getSizeInBits(*sem) / 8;
      patternKind = "largest-finite";
    } else {
      // Integer types. Every float dtype we support has LLVM APFloat
      // semantics (see `Support/include/Support/ML/FloatTypes.def`), so
      // a future float type without semantics is the only way to land
      // here as a float — at which point this code needs to grow a fill
      // strategy for it rather than silently fall through to 0xCD.
      assert(!dtype.isFloat() && "float dtype lacks LLVM APFloat semantics");
      bits = 0xCD;
      patternBytes = 1;
      patternKind = "byte-fill";
    }

    llvm::errs() << "DEBUG_ALLOC_POISON: Filling "
                 << (hostAlloc ? "host" : "device") << " " << dtype
                 << " allocation (" << totalBytes << " bytes) with "
                 << patternKind << " pattern ";
    (llvm::errs().changeColor(llvm::raw_ostream::RED)
     << "0x" << llvm::format_hex_no_prefix(bits, patternBytes * 2))
        .resetColor();
    llvm::errs() << "\n";

    if (hostAlloc) {
      uint8_t *dst = static_cast<uint8_t *>(buf->ptr());
      if (patternBytes == 1) {
        std::memset(dst, static_cast<uint8_t>(bits), totalBytes);
      } else {
        // Replicate the patternBytes-byte little-endian pattern across the
        // buffer. We avoid `std::fill_n` through a `uint{16,32,64}_t *`
        // because `createHostBuffer` does not guarantee `alignof(T)` for
        // T larger than a byte, and a misaligned typed store is UB. The
        // `bits` argument to `memcpy` is taken by address but read as a
        // contiguous byte sequence, so no alignment requirement applies.
        for (size_t off = 0; off + patternBytes <= totalBytes;
             off += patternBytes) {
          std::memcpy(dst + off, &bits, patternBytes);
        }
      }
    } else {
      ErrorOrSuccess err = ctx->enqueueSetMemory(*buf, bits, patternBytes);
      if (err.isError())
        return err.takeError();
      err = ctx->synchronize();
      if (err.isError())
        return err.takeError();
    }
  }

  return new DeviceMemory(M::copyRCRef(this), buf, spec);
}

MODULAR_DRIVER_EXPORT DeviceMemory *
Device::createDeviceMemory(void *data, const TensorSpec &spec,
                           llvm::unique_function<void()> destructor) {
  auto buf =
      ctx->createBufferOwning(data, spec.getSizeInBytes(), /*owning=*/true);
  // Set the destructor on the underlying DeviceBuffer
  buf->setDestructorFn(std::move(destructor));
  return new DeviceMemory(M::copyRCRef(this), buf, spec);
}

ErrorOr<llvm::SmallVector<AsyncRT::AnyAsyncValueRef>>
M::GraphCompiler::Driver::mapDeviceRefsToDevices(
    const ArrayRef<DeviceRef> deviceRefs, const DeviceMap &deviceMap,
    CPUDevice &cpuDevice) {

  // We need to explicitly synchronize the device here until MGP shares the same
  // DeviceContext as the driver. Otherwise there may be an async alloc/copy
  // pending on the driver when the first kernel tries to access the resource.
  for (auto &dev : deviceMap) {
    if (dev.second) {
      if (auto err = dev.second->synchronize())
        return err.takeError();
    }
  }

  llvm::SmallVector<AsyncRT::AnyAsyncValueRef> deviceContexts;
  for (const auto &devRef : deviceRefs) {
    // CPU device not explicitly set so deviceMap["cpu"] may be null
    if (devRef.label == kCPULabel) {
      ASSIGN_OR_RETURN(ctx, M::Driver::DeviceContext::create("cpu", 0));
      deviceContexts.push_back(
          AnyAsyncValueRef::createReady<M::Driver::DeviceContextRef>(cpuDevice,
                                                                     ctx));
    } else {
      if (auto it = deviceMap.find(devRef);
          it != deviceMap.end() && it->second) {
        const RCRef<Device> &dev = it->second;
        deviceContexts.push_back(
            AnyAsyncValueRef::createReady<M::Driver::DeviceContextRef>(
                cpuDevice, dev->tmpGetDeviceContext()));
      } else {
        ASSERT_STREAM(devRef.label == kGPULabel || devRef.label == kNPULabel,
                      "Only accelerator device context supported");
        // In this case we use the device ref id as the device id. THIS IS NOT
        // THE CASE WHEN THE DEVICE IS PASSED IN TO THE GRAPH.
        // We should eventually push the device selection up the stack into the
        // mt command and remove all of the code in this else branch.
        size_t numPhysicalDevices =
            M::Driver::Device::numberOfDevices(M::Driver::Device::gpuAPI);
        if (devRef.id >= numPhysicalDevices)
          return Error(devRef.toString() +
                       " id exceeds number of available accelerator devices.");
        ASSIGN_OR_RETURN(ctx, M::Driver::DeviceContext::create(
                                  M::Driver::Device::gpuAPI, devRef.id));
        deviceContexts.push_back(
            AnyAsyncValueRef::createReady<M::Driver::DeviceContextRef>(
                cpuDevice, ctx));
      }
    }
  }
  return deviceContexts;
}

static ErrorOr<Device *> createCPUDevice(ssize_t numaId) {
  ASSIGN_OR_RETURN(targetInfo, getHostTargetInfo());
  DeviceSpec spec;
  spec.ref.id = 0; // TODO: this was stored in CPUDevice but never used???
  spec.ref.label = kCPULabel;
  spec.target = std::move(targetInfo);
  ASSIGN_OR_RETURN(ctx, M::Driver::DeviceContext::create("cpu"));
  auto device = RCRef<Device>::create(spec, ctx);

  ErrorOrSuccess compatibleOr = device->isCompatible();
  if (compatibleOr.isError())
    return compatibleOr.takeError();

  return device.release();
}

MODULAR_DRIVER_EXPORT Device *M_createCPUDevice(ssize_t numaId,
                                                Status *status) {
  auto deviceOrError = createCPUDevice(numaId);
  if (deviceOrError.isError()) {
    status->errorOr = deviceOrError.takeError();
    return nullptr;
  }
  return deviceOrError.takeValue();
}

ErrorOr<std::unique_ptr<Device, decltype(&M_destroyDevice)>>
M::GraphCompiler::Driver::makeCPUDevice() {
  Status status;
  Device *device = M_createCPUDevice(0, &status);
  if (status.errorOr.isError())
    return status.errorOr.takeError();
  return std::unique_ptr<Device, decltype(&M_destroyDevice)>{device,
                                                             &M_destroyDevice};
}

static ErrorOr<Device *> createAccelerator(ssize_t gpuID,
                                           const char *label = kGPULabel) {
  size_t id = gpuID >= 0 ? gpuID : 0;

  ASSIGN_OR_RETURN(ctx, M::Driver::DeviceContext::getOrCreate(
                            M::Driver::Device::gpuAPI, id));
  DeviceSpec spec;
  spec.ref.label = label;
  spec.ref.id = gpuID;
  auto device = RCRef<Device>::create(spec, ctx);

  ErrorOrSuccess compatibleOr = device->isCompatible();
  if (compatibleOr.isError())
    return compatibleOr.takeError();

  return device.release();
}

MODULAR_DRIVER_EXPORT Device *
M_createAcceleratorDevice(ssize_t gpuID,
                          M::GraphCompiler::Driver::Status *status) {
  auto deviceOrError = createAccelerator(gpuID);
  if (deviceOrError.isError()) {
    status->errorOr = deviceOrError.takeError();
    return nullptr;
  }
  return deviceOrError.takeValue();
}

MODULAR_DRIVER_EXPORT Device *
M_createNPUDevice(ssize_t npuID, M::GraphCompiler::Driver::Status *status) {
  auto deviceOrError = createAccelerator(npuID, kNPULabel);
  if (deviceOrError.isError()) {
    status->errorOr = deviceOrError.takeError();
    return nullptr;
  }
  return deviceOrError.takeValue();
}
