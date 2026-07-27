//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Pure shim implementing the HAL DeviceContext.h replacement header: every
// method forwards 1:1 to a MojoHAL_* C-ABI export of the Mojo HAL
// DeviceContext.
//
//===----------------------------------------------------------------------===//

#include "MLRT/Driver/DeviceContext/HALDeviceContext.h"

#include "AsyncRT/Runtime/CPUDevice.h"
#include "MLRT/Driver/DeviceContext/BinaryID.h"
#include "Support/BinaryID.h"
#include "Support/Configuration.h"
#include "llvm/ADT/Twine.h"

#include <atomic>
#include <mutex>

extern "C" {
void MojoHAL_strfree(const char *s);
int32_t MojoHAL_Device_numberOfDevices(const char *api);
const char *MojoHAL_DeviceContext_create(MojoHALContextBox **out_ctx,
                                         const char *api, int32_t device_id);
void MojoHAL_DeviceContext_release(MojoHALContextBox *ctx);
const char *MojoHAL_DeviceContext_synchronize(MojoHALContextBox *ctx);
const char *MojoHAL_DeviceContext_createBuffer(MojoHALBufferBox **out_buf,
                                               void **out_ptr,
                                               MojoHALContextBox *ctx,
                                               uint64_t byteSize);
const char *MojoHAL_DeviceContext_createBufferOwning(MojoHALBufferBox **out_buf,
                                                     MojoHALContextBox *ctx,
                                                     void *ptr,
                                                     uint64_t byteSize,
                                                     bool owning);
void MojoHAL_DeviceBuffer_release(MojoHALBufferBox *buf);
const char *MojoHAL_DeviceBuffer_relinquishOwnership(MojoHALBufferBox *buf);
const char *MojoHAL_DeviceContext_createHostBuffer(MojoHALBufferBox **out_buf,
                                                   void **out_ptr,
                                                   MojoHALContextBox *ctx,
                                                   uint64_t byteSize);
void MojoHAL_HostBuffer_release(MojoHALBufferBox *buf);
const char *MojoHAL_DeviceContext_enqueueCopy(MojoHALContextBox *ctx, void *dst,
                                              const void *src,
                                              uint64_t byteSize);
const char *MojoHAL_DeviceContext_enqueueCopyToDevice(MojoHALContextBox *ctx,
                                                      void *dst,
                                                      const void *src,
                                                      uint64_t byteSize);
const char *MojoHAL_DeviceContext_enqueueCopyFromDevice(MojoHALContextBox *ctx,
                                                        void *dst,
                                                        const void *src,
                                                        uint64_t byteSize);
const char *MojoHAL_DeviceContext_enqueueSetMemory(MojoHALContextBox *ctx,
                                                   void *dst, uint64_t byteSize,
                                                   uint64_t value,
                                                   uint64_t valueSize);
const char *MojoHAL_DeviceContext_getArchName(char **out_name,
                                              MojoHALContextBox *ctx);
const char *MojoHAL_DeviceContext_getName(char **out_name,
                                          MojoHALContextBox *ctx);
const char *MojoHAL_DeviceContext_getApi(char **out_api,
                                         MojoHALContextBox *ctx);
int64_t MojoHAL_DeviceContext_id(MojoHALContextBox *ctx);
bool MojoHAL_DeviceContext_canAccess(MojoHALContextBox *ctx,
                                     MojoHALContextBox *peer);
bool MojoHAL_DeviceContext_isCompatible(MojoHALContextBox *ctx);
const char *MojoHAL_DeviceContext_getMemoryInfo(uint64_t *out_free,
                                                uint64_t *out_total,
                                                MojoHALContextBox *ctx);
const char *MojoHAL_DeviceContext_getAttribute(int32_t *out_value,
                                               MojoHALContextBox *ctx,
                                               int32_t attr);
const char *MojoHAL_DeviceBuffer_createSubBuffer(MojoHALBufferBox **out_buf,
                                                 MojoHALBufferBox *parent,
                                                 uint64_t byteOffset,
                                                 uint64_t byteSize);
const char *MojoHAL_HostBuffer_createSubBuffer(MojoHALBufferBox **out_buf,
                                               MojoHALBufferBox *parent,
                                               uint64_t byteOffset,
                                               uint64_t byteSize);
const char *MojoHAL_DeviceContext_reassignWait(MojoHALContextBox *dst_ctx,
                                               MojoHALContextBox *src_ctx);
const char *MojoHAL_DeviceContext_memmapFile(MojoHALBufferBox **out_buf,
                                             void **out_ptr, uint64_t *out_size,
                                             MojoHALContextBox *ctx,
                                             const char *path);
void MojoHAL_MmapBuffer_release(MojoHALBufferBox *buf);
const char *MojoHAL_DeviceContext_stream(MojoHALStreamBox **out_stream,
                                         MojoHALContextBox *ctx);
const char *MojoHAL_DeviceContext_createStream(MojoHALStreamBox **out_stream,
                                               MojoHALContextBox *ctx,
                                               int32_t priority);
const char *MojoHAL_DeviceContext_withStream(MojoHALContextBox **out_ctx,
                                             MojoHALContextBox *ctx,
                                             MojoHALStreamBox *stream);
void MojoHAL_DeviceStream_release(MojoHALStreamBox *stream);
const char *MojoHAL_DeviceStream_synchronize(MojoHALStreamBox *stream);
const char *MojoHAL_DeviceStream_waitForEvent(MojoHALStreamBox *stream,
                                              MojoHALEventBox *event);
const char *MojoHAL_DeviceStream_nativeHandle(void **out_handle,
                                              MojoHALStreamBox *stream);
const char *MojoHAL_DeviceContext_eventCreate(MojoHALEventBox **out_event,
                                              MojoHALContextBox *ctx,
                                              bool record);
void MojoHAL_DeviceEvent_release(MojoHALEventBox *event);
const char *MojoHAL_DeviceEvent_synchronize(MojoHALEventBox *event);
const char *MojoHAL_DeviceEvent_isReady(bool *out_ready,
                                        MojoHALEventBox *event);
const char *MojoHAL_DeviceContext_enqueueEventRecord(MojoHALContextBox *ctx,
                                                     MojoHALEventBox *event);
const char *MojoHAL_DeviceContext_enqueueWaitForEvent(MojoHALContextBox *ctx,
                                                      MojoHALEventBox *event);
const char *MojoHAL_DeviceContext_bindToCurrentThread(MojoHALContextBox *ctx);
const char *MojoHAL_DeviceContext_pushContext(MojoHALScopeBox **out_scope,
                                              MojoHALContextBox *ctx);
void MojoHAL_DeviceContextScope_pop(MojoHALScopeBox *scope);
bool MojoHAL_DeviceContext_getDLPackStream(MojoHALStreamBox **out_stream,
                                           MojoHALContextBox *ctx);
const char *MojoHAL_DeviceContext_allocateCompletionFlag(
    MojoHALCompletionFlagBox **out_flag, void **out_host_ptr,
    uint64_t *out_dev_addr, MojoHALContextBox *ctx);
void MojoHAL_CompletionFlag_release(MojoHALCompletionFlagBox *flag);
void MojoHAL_CompletionFlag_signal(void *host_ptr, uint64_t value);
uint64_t MojoHAL_CompletionFlag_load(void *host_ptr);
void MojoHAL_CompletionFlag_reset(void *host_ptr);
const char *MojoHAL_DeviceStream_waitOnHostValue(MojoHALStreamBox *stream,
                                                 uint64_t device_addr,
                                                 uint64_t value);
}

namespace M {
namespace Driver {

namespace {
/// Converts a MojoHAL_* error return (malloc'd message or null) into an Error.
/// Only call when `err` is non-null.
Error takeError(const char *err) {
  Error result{llvm::Twine(err)};
  MojoHAL_strfree(err);
  return result;
}
} // namespace

// ===-- Device ----------------------------------------------------------=== //

int Device::numberOfDevices(StringRef api) {
  return MojoHAL_Device_numberOfDevices(std::string(api).c_str());
}

namespace {
/// Maps a plugin API name onto the `Device::API` string constants. Plugin
/// APIs without a dedicated constant (e.g. the mock test plugin) report the
/// generic gpuAPI.
Device::API mapAPI(StringRef api) {
  if (api == Device::cpuAPI)
    return Device::cpuAPI;
  if (api == Device::cudaAPI)
    return Device::cudaAPI;
  if (api == Device::hipAPI)
    return Device::hipAPI;
  if (api == Device::metalAPI)
    return Device::metalAPI;
  return Device::gpuAPI;
}

} // namespace

ErrorOr<DeviceRef> Device::create(StringRef api, int id) {
  // Each Device owns its own boxed Mojo HAL DeviceContext.
  MojoHALContextBox *ctx = nullptr;
  if (const char *err =
          MojoHAL_DeviceContext_create(&ctx, std::string(api).c_str(), id))
    return takeError(err);
  char *name = nullptr;
  if (const char *err = MojoHAL_DeviceContext_getName(&name, ctx)) {
    MojoHAL_DeviceContext_release(ctx);
    return takeError(err);
  }
  std::string nameStr = name;
  MojoHAL_strfree(name);
  // Resolve the concrete device API ("cuda"/"hip"/"metal"/"cpu") from the
  // driver rather than the requested name, which may be the generic "gpu".
  // This is what `api()` reports and what DLPack/vendor dispatch keys on.
  char *apiName = nullptr;
  if (const char *err = MojoHAL_DeviceContext_getApi(&apiName, ctx)) {
    MojoHAL_DeviceContext_release(ctx);
    return takeError(err);
  }
  std::string apiStr = apiName;
  MojoHAL_strfree(apiName);
  return DeviceRef::take(new Device(
      ctx, mapAPI(apiStr), apiStr,
      static_cast<int>(MojoHAL_DeviceContext_id(ctx)), std::move(nameStr)));
}

ErrorOr<DeviceContextRef> Device::createContext() {
  MojoHALContextBox *ctx = nullptr;
  if (const char *err =
          MojoHAL_DeviceContext_create(&ctx, apiStr_.c_str(), id_))
    return takeError(err);
  return DeviceContextRef::take(new DeviceContext(ctx, DeviceRef::copy(this)));
}

Device::~Device() {
  // Metadata-only Devices (see adoptMojoBox) own no context.
  if (mojoCtx_)
    MojoHAL_DeviceContext_release(mojoCtx_);
}

const std::string &Device::getAcceleratorArchOrEmpty() {
  static std::string arch = [] {
    ErrorOr<DeviceRef> devOr = create(gpuAPI, 0);
    if (devOr.isError())
      return std::string();

    DeviceRef dev = devOr.takeValue();
    ErrorOr<const char *> nameOr = dev.getPointer()->getArchName();
    if (nameOr.isError())
      return std::string();
    llvm::StringRef name(nameOr.takeValue());

    // FIXME: Vendor mapping logic should move out of here.
    llvm::StringRef vendor;
    if (dev->api() == "cuda")
      vendor = "nvidia";
    else if (dev->api() == "hip")
      vendor = "amdgpu";
#if MODULAR_MACOS
    else if (dev->api() == "metal")
      vendor = "metal";
#endif
    if (vendor.empty())
      return std::string();

    return (llvm::Twine(vendor) + ":" + name).str();
  }();
  return arch;
}

ErrorOr<const char *> Device::getArchName() const {
  if (archName_.empty()) {
    char *name = nullptr;
    if (const char *err = MojoHAL_DeviceContext_getArchName(&name, mojoCtx_))
      return takeError(err);
    archName_ = name;
    MojoHAL_strfree(name);
  }
  return archName_.c_str();
}

ErrorOrSuccess Device::isCompatible() const {
  if (!MojoHAL_DeviceContext_isCompatible(mojoCtx_))
    return Error("device is not compatible with MAX");
  return {};
}

ErrorOr<Device::MemInfo> Device::getMemoryInfo() const {
  MemInfo info;
  uint64_t memFree = 0;
  uint64_t memTotal = 0;
  if (const char *err =
          MojoHAL_DeviceContext_getMemoryInfo(&memFree, &memTotal, mojoCtx_))
    return takeError(err);
  info.free = memFree;
  info.total = memTotal;
  return info;
}

ErrorOr<int> Device::getAttribute(Attribute attr) const {
  int32_t value = 0;
  if (const char *err = MojoHAL_DeviceContext_getAttribute(
          &value, mojoCtx_, static_cast<int32_t>(attr)))
    return takeError(err);
  return static_cast<int>(value);
}

ErrorOr<size_t> Device::maxSingleAllocationSize() const {
  auto info = getMemoryInfo();
  if (info.isError())
    return info.takeError();
  return info->total;
}

ErrorOr<bool> Device::canAccess(const Device &other) const {
  return MojoHAL_DeviceContext_canAccess(mojoCtx_, other.mojoCtx_);
}

// ===-- DeviceBuffer ----------------------------------------------------=== //

DeviceBuffer::~DeviceBuffer() {
  // After the custom destructor, relinquish ownership of the wrapped memory
  // before releasing the Mojo box so nothing is freed twice. Only Device boxes
  // (the createBufferOwning path, the sole setDestructorFn consumer) support
  // the ownership flip.
  if (destructorFn_ && kind_ == BoxKind::Device) {
    destructorFn_();
    if (const char *err = MojoHAL_DeviceBuffer_relinquishOwnership(mojoBox_)) {
      MojoHAL_strfree(err);
      return;
    }
    MojoHAL_DeviceBuffer_release(mojoBox_);
    return;
  }
  switch (kind_) {
  case BoxKind::Device:
    MojoHAL_DeviceBuffer_release(mojoBox_);
    return;
  case BoxKind::Host:
    MojoHAL_HostBuffer_release(mojoBox_);
    return;
  case BoxKind::Mmap:
    MojoHAL_MmapBuffer_release(mojoBox_);
    return;
  }
}

ErrorOr<DeviceBufferRef> DeviceBuffer::createSubBuffer(size_t byteOffset,
                                                       size_t byteSize) {
  // Device and Mmap boxes expose a Mojo DeviceBuffer view the export can
  // re-slice directly; Host boxes carry a Mojo HostBuffer, whose export
  // returns the same kind of DeviceBuffer view over the pinned host region.
  MojoHALBufferBox *buf = nullptr;
  const char *err = kind_ == BoxKind::Host
                        ? MojoHAL_HostBuffer_createSubBuffer(
                              &buf, mojoBox_, byteOffset, byteSize)
                        : MojoHAL_DeviceBuffer_createSubBuffer(
                              &buf, mojoBox_, byteOffset, byteSize);
  if (err)
    return takeError(err);
  // Sub-buffer boxes are plain Mojo DeviceBuffer views regardless of what
  // backs the parent; `parent_` keeps the parent box (and any mapping) alive.
  return DeviceBufferRef::take(new DeviceBuffer(
      buf, static_cast<char *>(ptr_) + byteOffset, byteSize, isHost_,
      BoxKind::Device, context_.copy(), DeviceBufferRef::copy(this)));
}

void *DeviceBuffer::releasePtr() {
  if (const char *err = MojoHAL_DeviceBuffer_relinquishOwnership(mojoBox_))
    MojoHAL_strfree(err); // Cannot fail in practice (plugin-side flag flip);
                          // the legacy signature is non-failing.
  auto tmp = ptr_;
  ptr_ = nullptr;
  return tmp;
}

ErrorOrSuccess DeviceBuffer::copyTo(DeviceBuffer &dst, size_t srcByteOffset,
                                    size_t srcByteSize) {
  // Mirror DeviceContext::enqueueCopyDeviceToDevice (the legacy impl):
  // dispatch based on whether src and dst are host memory.
  DeviceBuffer &src = *this;
  const Device::API srcApi = src.context().device().api();
  const Device::API dstApi = dst.context().device().api();
  const uint8_t *srcAt = static_cast<const uint8_t *>(src.ptr_) + srcByteOffset;

  if (src.isHost() || srcApi == Device::cpuAPI) {
    // Host -> device. If the source is owned by a CPU device context,
    // synchronize it first: events cannot be awaited across device types.
    if (srcApi == Device::cpuAPI && dstApi != Device::cpuAPI) {
      ErrorOrSuccess sync = src.context().synchronize();
      if (sync.isError())
        return sync;
    }
    return dst.context().enqueueCopyToDevice(dst.ptr_, srcAt, srcByteSize);
  }
  if (dst.isHost() || dstApi == Device::cpuAPI) {
    // Device -> host.
    if (dstApi == Device::cpuAPI && srcApi != Device::cpuAPI) {
      ErrorOrSuccess sync = dst.context().synchronize();
      if (sync.isError())
        return sync;
    }
    return src.context().enqueueCopyFromDevice(dst.ptr_, srcAt, srcByteSize);
  }
  if (srcApi == dstApi) {
    if (const char *err = MojoHAL_DeviceContext_enqueueCopy(
            dst.context_->mojoCtx_, dst.ptr_, srcAt, srcByteSize))
      return takeError(err);
    return {};
  }
  return Error(llvm::Twine("device API mismatch copying from \"") + srcApi +
               "\" to \"" + dstApi + "\"");
}

ErrorOrSuccess DeviceBuffer::synchronize(void *external_stream) {
  // The HAL backend has no external-stream support (`createExternalStream` is
  // unimplemented). Fall back to a full host-blocking synchronize of the
  // producer's context.
  (void)external_stream;
  return context_->synchronize();
}

ErrorOrSuccess DeviceBuffer::reassignOwnershipTo(DeviceContext &ctx) {
  if (const char *err =
          MojoHAL_DeviceContext_reassignWait(ctx.mojoCtx_, context_->mojoCtx_))
    return takeError(err);
  context_ = DeviceContextRef::copy(&ctx);
  return {};
}

// ===-- DeviceContext ---------------------------------------------------=== //

ErrorOr<DeviceContextRef> DeviceContext::create(StringRef api, int id) {
  MojoHALContextBox *ctx = nullptr;
  if (const char *err =
          MojoHAL_DeviceContext_create(&ctx, std::string(api).c_str(), id))
    return takeError(err);
  auto device = Device::create(api, id);
  if (device.isError()) {
    MojoHAL_DeviceContext_release(ctx);
    return device.takeError();
  }
  return DeviceContextRef::take(new DeviceContext(ctx, std::move(*device)));
}

ErrorOr<DeviceContextRef> DeviceContext::getOrCreate(StringRef api, int id) {
  // TODO: context caching belongs on the Mojo side of the exports; until
  // a MojoHAL_DeviceContext_getOrCreate export exists, each call creates a
  // fresh context over the same underlying device.
  return create(api, id);
}

ErrorOr<DeviceContextRef> DeviceContext::adoptMojoBox(void *mojoBox) {
  auto *box = static_cast<MojoHALContextBox *>(mojoBox);
  // The box is an owned copy of a Mojo DeviceContext (from the Mojo
  // `mojoHandle()` exchange); this is on the kernel-execution hot path. Build a
  // METADATA-ONLY Device by cheaply querying the existing box (api/id/name) —
  // do NOT call Device::create, which reloads the plugin (dlopen/dlsym) and
  // crashes when run mid-execution. The Device carries no box of its own
  // (mojoCtx_ == null) and is used only for cached `device().api()`/`id()`
  // (e.g. the copyTo direction dispatch); box-backed device queries are not
  // reachable on this path. The new DeviceContext takes ownership of the box
  // and releases it when its refcount drops to zero.
  char *apiName = nullptr;
  if (const char *err = MojoHAL_DeviceContext_getApi(&apiName, box)) {
    MojoHAL_DeviceContext_release(box);
    return takeError(err);
  }
  std::string apiStr = apiName;
  MojoHAL_strfree(apiName);

  char *name = nullptr;
  if (const char *err = MojoHAL_DeviceContext_getName(&name, box)) {
    MojoHAL_DeviceContext_release(box);
    return takeError(err);
  }
  std::string nameStr = name;
  MojoHAL_strfree(name);

  DeviceRef device = DeviceRef::take(new Device(
      /*mojoCtx=*/nullptr, mapAPI(apiStr), apiStr,
      static_cast<int>(MojoHAL_DeviceContext_id(box)), std::move(nameStr)));
  return DeviceContextRef::take(new DeviceContext(box, std::move(device)));
}

DeviceContext::DeviceContext(MojoHALContextBox *mojoCtx, DeviceRef device)
    : mojoCtx_(mojoCtx), streamPool_(std::make_shared<StreamPool>()),
      device_(std::move(device)) {
  // Capture this context's own stream as the pool's entry 0 so
  // selectStream(0) resolves to the base stream from any view. Best-effort:
  // if it fails the pool has no base and selectStream(0) falls back to this
  // context.
  if (ErrorOr<DeviceStreamRef> s = stream(); !s.isError())
    streamPool_->base = std::move(*s);
}

DeviceContext::~DeviceContext() { MojoHAL_DeviceContext_release(mojoCtx_); }

ErrorOr<DeviceBufferRef> DeviceContext::enqueueCreateBuffer(size_t byteSize) {
  MojoHALBufferBox *buf = nullptr;
  void *ptr = nullptr;
  if (const char *err =
          MojoHAL_DeviceContext_createBuffer(&buf, &ptr, mojoCtx_, byteSize))
    return takeError(err);
  return DeviceBufferRef::take(new DeviceBuffer(
      buf, ptr, byteSize, /*isHost=*/false, DeviceBuffer::BoxKind::Device,
      DeviceContextRef::copy(this)));
}

ErrorOr<DeviceBufferRef> DeviceContext::createHostBuffer(size_t byteSize) {
  // Like the legacy implementation: an empty host buffer is non-pinned
  // (isHost=false), so it can stand in for a GPU graph input (a pinned one is
  // rejected).
  if (byteSize == 0)
    return createBufferOwning(nullptr, 0, /*owning=*/true);
  MojoHALBufferBox *buf = nullptr;
  void *ptr = nullptr;
  if (const char *err = MojoHAL_DeviceContext_createHostBuffer(
          &buf, &ptr, mojoCtx_, byteSize))
    return takeError(err);
  return DeviceBufferRef::take(new DeviceBuffer(
      buf, ptr, byteSize, /*isHost=*/true, DeviceBuffer::BoxKind::Host,
      DeviceContextRef::copy(this)));
}

DeviceBufferRef DeviceContext::createBufferOwning(void *ptr, size_t byteSize,
                                                  bool owning) {
  MojoHALBufferBox *buf = nullptr;
  if (const char *err = MojoHAL_DeviceContext_createBufferOwning(
          &buf, mojoCtx_, ptr, byteSize, owning)) {
    // The replaced API returns a bare ref; surface failure as an empty ref.
    MojoHAL_strfree(err);
    return DeviceBufferRef();
  }
  return DeviceBufferRef::take(new DeviceBuffer(
      buf, ptr, byteSize, /*isHost=*/false, DeviceBuffer::BoxKind::Device,
      DeviceContextRef::copy(this)));
}

DeviceBufferRef
DeviceContext::createHostBufferOwning(void *ptr, size_t byteSize, bool owning) {
  // Like the legacy implementation, a raw wrap of the caller's host pointer:
  // the same Mojo box as createBufferOwning, surfaced with isHost set.
  MojoHALBufferBox *buf = nullptr;
  if (const char *err = MojoHAL_DeviceContext_createBufferOwning(
          &buf, mojoCtx_, ptr, byteSize, owning)) {
    // The replaced API returns a bare ref; surface failure as an empty ref.
    MojoHAL_strfree(err);
    return DeviceBufferRef();
  }
  return DeviceBufferRef::take(new DeviceBuffer(
      buf, ptr, byteSize, /*isHost=*/true, DeviceBuffer::BoxKind::Device,
      DeviceContextRef::copy(this)));
}

ErrorOr<DeviceBufferRef>
DeviceContext::memmapFile(const std::filesystem::path &filename) {
  MojoHALBufferBox *buf = nullptr;
  void *ptr = nullptr;
  uint64_t size = 0;
  if (const char *err = MojoHAL_DeviceContext_memmapFile(
          &buf, &ptr, &size, mojoCtx_, filename.c_str()))
    return takeError(err);
  return DeviceBufferRef::take(new DeviceBuffer(buf, ptr, size, /*isHost=*/true,
                                                DeviceBuffer::BoxKind::Mmap,
                                                DeviceContextRef::copy(this)));
}

ErrorOrSuccess DeviceContext::enqueueCopyToDevice(void *dst, const void *src,
                                                  size_t byteSize) {
  // Host-to-device: `dst` is a device address, `src` a host pointer. Use the
  // direction-specific entry (cuMemcpyHtoD) so pageable host memory works.
  if (const char *err = MojoHAL_DeviceContext_enqueueCopyToDevice(
          mojoCtx_, dst, src, byteSize))
    return takeError(err);
  return {};
}

ErrorOrSuccess DeviceContext::enqueueCopyFromDevice(void *dst, const void *src,
                                                    size_t byteSize) {
  // Device-to-host: `dst` is a host pointer, `src` a device address
  // (cuMemcpyDtoH).
  if (const char *err = MojoHAL_DeviceContext_enqueueCopyFromDevice(
          mojoCtx_, dst, src, byteSize))
    return takeError(err);
  return {};
}

ErrorOrSuccess DeviceContext::enqueueSetMemory(void *dst, size_t byteSize,
                                               uint64_t value,
                                               size_t valueSize) {
  if (const char *err = MojoHAL_DeviceContext_enqueueSetMemory(
          mojoCtx_, dst, byteSize, value, valueSize))
    return takeError(err);
  return {};
}

ErrorOrSuccess DeviceContext::synchronize(void *external_stream) {
  if (external_stream != nullptr)
    return Error("HAL DeviceContext::synchronize: external streams are not "
                 "supported");
  if (const char *err = MojoHAL_DeviceContext_synchronize(mojoCtx_))
    return takeError(err);
  return {};
}

ErrorOr<Device::MemInfo> DeviceContext::getMemoryInfo() const {
  Device::MemInfo info;
  uint64_t memFree = 0;
  uint64_t memTotal = 0;
  if (const char *err =
          MojoHAL_DeviceContext_getMemoryInfo(&memFree, &memTotal, mojoCtx_))
    return takeError(err);
  info.free = memFree;
  info.total = memTotal;
  return info;
}

ErrorOr<const char *> DeviceContext::getArchName() const {
  if (archName_.empty()) {
    char *name = nullptr;
    if (const char *err = MojoHAL_DeviceContext_getArchName(&name, mojoCtx_))
      return takeError(err);
    archName_ = name;
    MojoHAL_strfree(name);
  }
  return archName_.c_str();
}

ErrorOrSuccess DeviceContext::isCompatible() const {
  if (!MojoHAL_DeviceContext_isCompatible(mojoCtx_))
    return Error("device is not compatible with MAX");
  return {};
}

// ===-- DeviceContextScope ----------------------------------------------=== //

DeviceContextScope::DeviceContextScope(const DeviceContext &ctx)
    : mojoScope_(nullptr) {
  if (const char *err =
          MojoHAL_DeviceContext_pushContext(&mojoScope_, ctx.mojoCtx_))
    err_ = takeError(err);
}

DeviceContextScope::~DeviceContextScope() {
  if (mojoScope_)
    MojoHAL_DeviceContextScope_pop(mojoScope_);
}

// ===-- CompletionFlag --------------------------------------------------=== //

CompletionFlag::~CompletionFlag() { MojoHAL_CompletionFlag_release(mojoFlag_); }

void CompletionFlag::reset() { MojoHAL_CompletionFlag_reset(hostPtr_); }

void CompletionFlag::signal(uint64_t value) {
  MojoHAL_CompletionFlag_signal(hostPtr_, value);
}

uint64_t CompletionFlag::load() const {
  return MojoHAL_CompletionFlag_load(hostPtr_);
}

// ===-- DeviceStream ----------------------------------------------------=== //

DeviceStream::~DeviceStream() { MojoHAL_DeviceStream_release(mojoStream_); }

ErrorOrSuccess DeviceStream::synchronize() {
  if (const char *err = MojoHAL_DeviceStream_synchronize(mojoStream_))
    return takeError(err);
  return {};
}

ErrorOr<DeviceEventRef> DeviceStream::enqueueEvent() {
  // Only reachable through external streams, which createExternalStream
  // rejects on the HAL backend.
  return Error("stream-level event records are not supported with the HAL "
               "DeviceContext backend");
}

ErrorOrSuccess DeviceStream::enqueueWaitForEvent(DeviceEvent &event) {
  if (const char *err =
          MojoHAL_DeviceStream_waitForEvent(mojoStream_, event.mojoEvent_))
    return takeError(err);
  return {};
}

void *DeviceStream::ptr() const {
  void *handle = nullptr;
  if (const char *err =
          MojoHAL_DeviceStream_nativeHandle(&handle, mojoStream_)) {
    MojoHAL_strfree(err); // The replaced API cannot fail; report null.
    return nullptr;
  }
  return handle;
}

ErrorOrSuccess DeviceStream::enqueueWaitOnHostValue(const CompletionFlag &flag,
                                                    uint64_t value) const {
  if (const char *err = MojoHAL_DeviceStream_waitOnHostValue(
          mojoStream_, flag.devicePtr(), value))
    return takeError(err);
  return {};
}

// ===-- DeviceEvent -----------------------------------------------------=== //

DeviceEvent::~DeviceEvent() { MojoHAL_DeviceEvent_release(mojoEvent_); }

ErrorOrSuccess DeviceEvent::synchronize() {
  if (const char *err = MojoHAL_DeviceEvent_synchronize(mojoEvent_))
    return takeError(err);
  return {};
}

ErrorOr<bool> DeviceEvent::isReady() const {
  bool ready = false;
  if (const char *err = MojoHAL_DeviceEvent_isReady(&ready, mojoEvent_))
    return takeError(err);
  return ready;
}

ErrorOr<float> DeviceEvent::elapsedTime(const DeviceEvent &end) const {
  return Error(
      "event timing is not supported with the HAL DeviceContext backend");
}

// ===-- DeviceContext streams / events ----------------------------------=== //

ErrorOr<DeviceStreamRef> DeviceContext::stream() const {
  MojoHALStreamBox *stream = nullptr;
  if (const char *err = MojoHAL_DeviceContext_stream(&stream, mojoCtx_))
    return takeError(err);
  return DeviceStreamRef::take(new DeviceStream(stream, device_.copy()));
}

ErrorOr<DeviceStreamRef> DeviceContext::createStream(int priority) {
  MojoHALStreamBox *stream = nullptr;
  if (const char *err =
          MojoHAL_DeviceContext_createStream(&stream, mojoCtx_, priority))
    return takeError(err);
  DeviceStreamRef ref =
      DeviceStreamRef::take(new DeviceStream(stream, device_.copy()));
  {
    // Retain the stream in the pool so it is counted in numStreams() and
    // addressable via selectStream(), per the legacy contract.
    std::lock_guard<std::mutex> lock(streamPool_->mu);
    streamPool_->side.push_back(ref.copy());
  }
  return ref;
}

ErrorOr<DeviceStreamRef> DeviceContext::createExternalStream(void *) {
  return Error("external streams are not supported with the HAL "
               "DeviceContext backend");
}

unsigned DeviceContext::numStreams() const {
  std::lock_guard<std::mutex> lock(streamPool_->mu);
  return 1 + streamPool_->side.size();
}

ErrorOr<DeviceContextRef> DeviceContext::selectStream(unsigned streamId) {
  DeviceStreamRef target;
  {
    std::lock_guard<std::mutex> lock(streamPool_->mu);
    if (streamId == 0) {
      if (!streamPool_->base)
        return DeviceContextRef::copy(this);
      target = streamPool_->base.copy();
    } else if (streamId - 1 < streamPool_->side.size()) {
      target = streamPool_->side[streamId - 1].copy();
    } else {
      return Error("invalid stream id: " + llvm::Twine(streamId) +
                   " (context has " +
                   llvm::Twine(1 + streamPool_->side.size()) + " streams)");
    }
  }
  // A view is a copy of this context's Mojo box whose current stream is the
  // selected one; it shares the pool so further selection resolves the same
  // stream set.
  MojoHALContextBox *box = nullptr;
  if (const char *err =
          MojoHAL_DeviceContext_withStream(&box, mojoCtx_, target->mojoStream_))
    return takeError(err);
  return DeviceContextRef::take(
      new DeviceContext(box, device_.copy(), streamPool_));
}

bool DeviceContext::operator==(const DeviceContext &other) const {
  return this == &other;
}

ErrorOr<DeviceEventRef> DeviceContext::enqueueEvent() {
  MojoHALEventBox *event = nullptr;
  if (const char *err =
          MojoHAL_DeviceContext_eventCreate(&event, mojoCtx_, /*record=*/true))
    return takeError(err);
  return DeviceEventRef::take(
      new DeviceEvent(event, device_.copy(), DeviceContextRef::copy(this)));
}

ErrorOrSuccess DeviceContext::enqueueEvent(const DeviceEvent &event) {
  if (const char *err =
          MojoHAL_DeviceContext_enqueueEventRecord(mojoCtx_, event.mojoEvent_))
    return takeError(err);
  return {};
}

ErrorOr<DeviceEventRef> DeviceContext::eventCreate(bool enableTiming) {
  (void)enableTiming; // HAL events are always CPU-visible.
  MojoHALEventBox *event = nullptr;
  if (const char *err = MojoHAL_DeviceContext_eventCreate(&event, mojoCtx_,
                                                          /*record=*/false))
    return takeError(err);
  return DeviceEventRef::take(
      new DeviceEvent(event, device_.copy(), DeviceContextRef::copy(this)));
}

ErrorOrSuccess DeviceContext::enqueueWaitFor(DeviceEvent &event) {
  if (const char *err =
          MojoHAL_DeviceContext_enqueueWaitForEvent(mojoCtx_, event.mojoEvent_))
    return takeError(err);
  return {};
}

ErrorOrSuccess DeviceContext::enqueueWaitFor(DeviceContext &ctx) {
  // This context waits for all work currently enqueued on `ctx`.
  if (const char *err =
          MojoHAL_DeviceContext_reassignWait(mojoCtx_, ctx.mojoCtx_))
    return takeError(err);
  return {};
}

ErrorOrSuccess DeviceContext::bindToCurrentThread() const {
  if (const char *err = MojoHAL_DeviceContext_bindToCurrentThread(mojoCtx_))
    return takeError(err);
  return {};
}

std::optional<intptr_t> DeviceContext::getDLPackStream() const {
  MojoHALStreamBox *stream = nullptr;
  if (!MojoHAL_DeviceContext_getDLPackStream(&stream, mojoCtx_))
    return std::nullopt;
  return reinterpret_cast<intptr_t>(stream);
}

Error DeviceContext::formatError(const char *file, int line, const char *func,
                                 const llvm::Twine &msg) {
  return Error(llvm::Twine(func) + " at " + file + ":" + llvm::Twine(line) +
               ": " + msg);
}

ErrorOr<CompletionFlagRef> DeviceContext::allocateCompletionFlag() {
  MojoHALCompletionFlagBox *flag = nullptr;
  void *hostPtr = nullptr;
  uint64_t devAddr = 0;
  if (const char *err = MojoHAL_DeviceContext_allocateCompletionFlag(
          &flag, &hostPtr, &devAddr, mojoCtx_))
    return takeError(err);
  return CompletionFlagRef::take(new CompletionFlag(flag, hostPtr, devAddr));
}

// ===-- Device graphs (unsupported) --------------------------------------===
// //

// Capture always fails, so no DeviceGraph instance ever exists and the replay
// entry points below are unreachable; they are defined only to close the link.

static constexpr const char *kNoDeviceGraphs =
    "device graphs are not supported with the HAL DeviceContext backend";

ErrorOr<DeviceGraphCollection>
DeviceGraph::capture(M::AsyncRT::CPUDevice &cpuDevice,
                     llvm::ArrayRef<DeviceStreamRef> streams,
                     llvm::function_ref<ErrorOrSuccess()> execute) {
  return Error(kNoDeviceGraphs);
}

ErrorOrSuccess DeviceGraph::replay() const { return Error(kNoDeviceGraphs); }

ErrorOrSuccess DeviceGraphCollection::replay() const {
  return Error(kNoDeviceGraphs);
}

ErrorOrSuccess DeviceGraphCollection::debugVerifyReplay(
    llvm::function_ref<ErrorOrSuccess()> execute) const {
  return Error(kNoDeviceGraphs);
}

// ===-- Configuration -----------------------------------------------------===
// //

static std::mutex &getConfigMutex() {
  static std::mutex mu;
  return mu;
}

static ErrorOr<Config> &getConfig() {
  static ErrorOr<Config> configuration = Config::open();
  return configuration;
}

int64_t DeviceContext::configAsInt(StringRef key, int64_t defaultValue,
                                   bool *isDefault) {
  std::lock_guard<std::mutex> lock(getConfigMutex());
  ErrorOr<Config> &config = getConfig();
  if (config.isError()) {
    if (isDefault != nullptr) {
      *isDefault = true;
    }
    return defaultValue;
  }

  StringRef s = config->getValue((llvm::Twine("device-context.") + key).str());
  if (isDefault != nullptr) {
    *isDefault = s.empty();
  }
  long long result = defaultValue;
  if (s.getAsInteger(10, result)) {
    if (isDefault != nullptr) {
      *isDefault = true;
    }
    return defaultValue;
  }
  return result;
}

// ===-- Virtual device mode ----------------------------------------------===
// //
// Virtual device mode flags are the process-global AsyncRT runtime globals
// (AsyncRT/Runtime/Globals/VirtualDeviceGlobals.h), shared with the legacy
// backend; only the CPU codegen target below is Driver-owned.

static std::string &getVirtualCpuTargetImpl() {
  static std::string cpu;
  return cpu;
}
static std::mutex &getVirtualCpuTargetMutex() {
  static std::mutex mutex;
  return mutex;
}

MODULAR_CXX_EXPORT void setVirtualCpuTarget(StringRef cpu) {
  std::lock_guard<std::mutex> lock(getVirtualCpuTargetMutex());
  getVirtualCpuTargetImpl() = cpu.str();
}

MODULAR_CXX_EXPORT std::string getVirtualCpuTarget() {
  std::lock_guard<std::mutex> lock(getVirtualCpuTargetMutex());
  return getVirtualCpuTargetImpl();
}

// ===-- BinaryID ----------------------------------------------------------===
// //

std::string getMojoBindingsBinaryID() {
  // Binary ID of the shared library containing the device runtime, used for
  // MEF cache invalidation (BinaryID.h contract). Here that library is the
  // HAL shim's home rather than the AsyncRT Mojo bindings'.
  return M::getBinaryID();
}

} // namespace Driver
} // namespace M

// ===-- Legacy-named interop exports --------------------------------------===
// The stdlib's `_nvidia_cuda` / `_amdgpu_hip` vendor-interop modules call
// these AsyncRT_-named C exports (normally provided by the legacy Mojo
// bindings). Wiring the handle accessors to the HAL native-handle plumbing is
// future work; until then they report a clear error at the (vendor-library)
// call sites that use them.

namespace {
const char *halInteropUnavailable() {
  return strdup("CUDA/HIP interop handles are not available with the HAL "
                "DeviceContext yet");
}
} // namespace

extern "C" {
MODULAR_CXX_EXPORT void AsyncRT_DeviceContext_strfree(const char *s) {
  std::free(const_cast<char *>(s));
}
MODULAR_CXX_EXPORT const char *AsyncRT_DeviceContext_cuda_context(void *result,
                                                                  void *ctx) {
  return halInteropUnavailable();
}
MODULAR_CXX_EXPORT const char *
AsyncRT_DeviceContext_cuda_current_context(void *result) {
  return halInteropUnavailable();
}
MODULAR_CXX_EXPORT const char *AsyncRT_DeviceContext_hip_device(void *result,
                                                                void *ctx) {
  return halInteropUnavailable();
}
MODULAR_CXX_EXPORT const char *AsyncRT_DeviceFunction_cuda_module(void *result,
                                                                  void *func) {
  return halInteropUnavailable();
}
MODULAR_CXX_EXPORT const char *AsyncRT_DeviceFunction_hip_module(void *result,
                                                                 void *func) {
  return halInteropUnavailable();
}
MODULAR_CXX_EXPORT const char *AsyncRT_DeviceStream_cuda_stream(void *result,
                                                                void *stream) {
  return halInteropUnavailable();
}
MODULAR_CXX_EXPORT const char *AsyncRT_DeviceStream_hip_stream(void *result,
                                                               void *stream) {
  return halInteropUnavailable();
}
} // extern "C"

// ===-- AsyncValue runtime ABI ---------------------------------------------===
// //
// Backend-independent AsyncValue helpers Mojo's `std.runtime.async_value`
// calls via external_call.

using M::AsyncRT::AnyAsyncValueRef;
using M::AsyncRT::CPUDevice;
using M::Driver::DeviceBuffer;
using M::Driver::DeviceBufferRef;

// Wraps an already-owning `DeviceBuffer` handle (surrendered net-zero by the
// Mojo side via `DeviceBuffer.take_handle()`) in an
// `AsyncValue<DeviceBufferRef>`.
MODULAR_EXPORT M::AsyncRT::AsyncValue *
AsyncRT_AsyncValue_createFromDeviceBuffer(DeviceBuffer *handle) {
  CPUDevice &cpuDevice = *CPUDevice::getCurrentCPUDeviceOrNull();
  DeviceBufferRef buf = M::takeRCRef(handle);
  AnyAsyncValueRef storage =
      AnyAsyncValueRef::createReady<DeviceBufferRef>(cpuDevice, std::move(buf));
  return storage.releasePointer();
}
