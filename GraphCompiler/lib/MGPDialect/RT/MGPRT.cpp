//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Bindings between C++ primitive implementation and mojo. This is the successor
// to putting things in KGEN/lib/CompilerRT/AsyncRT.cpp
//
//===----------------------------------------------------------------------===//

#include "AsyncRT/ForwardDecls.h"
#include "AsyncRT/Runtime/AnyAsyncValueRef.h"
#include "AsyncRT/Runtime/AsyncValue.h"
#include "AsyncRT/Runtime/CPUDevice.h"
#include "AsyncRT/Support/UnknownLocationDecoder.h"
#include "GraphCompiler/Support/List.h"
#include "Init/Init.h"
#include "MLRT/Core/MojoValue.h"
#include "MLRT/Core/Tensor/StateContext.h"
#include "MLRT/Core/Tensor/TensorBufferRef.h"
#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "Support/Context.h"
#include "Support/ML/DebugPrint.h"
#include "Support/SymbolExport.h"

using namespace M;
using namespace M::MLRT;
using namespace M::AsyncRT;

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// An opaque wrapper around a pointer to a T.
template <typename T>
struct AsyncRTWrapper {
  void *ptr;
};

template <typename T>
T &unwrap(AsyncRTWrapper<T> ref) {
  return *reinterpret_cast<T *>(ref.ptr);
}
template <typename T>
AsyncRTWrapper<T> wrap(T *ptr) {
  return AsyncRTWrapper<T>{ptr};
}

template <typename T, typename... Args>
void fulfillReady(AnyAsyncValueRef &value, Args &&...args) {
  if (value.getPointer() && value.getPointer()->isIndirect()) {
    value.copy().emplaceIndirect<T>(std::forward<Args>(args)...);
  } else {
    CPUDevice &cpuDevice = *CPUDevice::getCurrentCPUDeviceOrNull();
    value = AnyAsyncValueRef::createReady<T>(cpuDevice,
                                             std::forward<Args>(args)...);
  }
}

//===----------------------------------------------------------------------===//
// Functions visible from Mojo
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Testing hooks
//===----------------------------------------------------------------------===//

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void MGP_RT_TEST() {
  llvm::outs() << "MGP_RT_TEST\n";
}

//===----------------------------------------------------------------------===//
// Actual functions called from Mojo
//===----------------------------------------------------------------------===//

/// This is the C++ implementation of mgp.int.cached.
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
MGP_RT_SetCachedInt(size_t bufferSlot, StateContext *stateCtx, size_t value) {
  stateCtx->emplaceStateSlot<IntStateSlot>(bufferSlot, value);
}

/// This is the C++ implementation of mgp.int.get_cached.
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT size_t
MGP_RT_GetCachedInt(size_t bufferSlot, StateContext *stateCtx) {
  auto &slot = stateCtx->getStateSlotPayload<IntStateSlot>(bufferSlot);
  return slot.getValue();
}

/// This is the new C++ implementation of the mgp.buffer.get_cached primitive.
/// The old one is not prefixed with TMP_.
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void *
MGP_RT_GetCachedBuffer(size_t bufferSlot, StateContext *stateCtx, size_t *size,
                       void **data) {
  auto &stateSlot = stateCtx->getStateSlotPayload<TensorStateSlot>(bufferSlot);
  ErrorOr<TensorBufferRef &> errOr = stateSlot.getBufferRef();

  /// TODO: Support errors in MGPRT
  if (errOr) {
  }

  TensorBufferRef &ref = errOr.get();
  *size = ref.getSize();
  *data = ref.getBuffer();

  return ref.getMemStorageHandleRef();
}

/// This is the C++ implementation of mgp.device_graph.cache. The Mojo
/// `DeviceGraphCache` is created on first use by calling back into Mojo:
/// `createFn` allocates and initializes it in one step, and `destructorFn`
/// destroys it and frees that allocation.
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void *
MGP_RT_GetOrCreateDeviceGraphCache(StateContext *stateCtx, void *(*createFn)(),
                                   void (*destructorFn)(void *)) {
  MojoValue &cache = stateCtx->getDeviceGraphCache(createFn, destructorFn);
  return cache.getData();
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
MGP_RT_CreateAsync_ssizet(ssize_t data,
                          AsyncRTWrapper<AnyAsyncValueRef> async) {
  AnyAsyncValueRef &value = unwrap(async);
  fulfillReady<ssize_t>(value, data);
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
MGP_RT_CreateAsync_int64t(int64_t data,
                          AsyncRTWrapper<AnyAsyncValueRef> async) {
  AnyAsyncValueRef &value = unwrap(async);
  fulfillReady<int64_t>(value, data);
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
MGP_RT_CreateAsync_chain(AsyncRTWrapper<AnyAsyncValueRef> async) {
  AnyAsyncValueRef &value = unwrap(async);
  fulfillReady<Chain>(value);
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
MGP_RT_CreateAsync_bool(bool data, AsyncRTWrapper<AnyAsyncValueRef> async) {
  AnyAsyncValueRef &value = unwrap(async);
  fulfillReady<bool>(value, data);
}

/// TAKES an already-owning DeviceBuffer handle the producer surrendered instead
/// of minting a fresh owner with createBufferOwning. Used by mgp.buffer.alloc:
/// the Mojo side hands
/// over the live DeviceBuffer from enqueue_create_buffer via
/// DeviceBuffer.take_handle() (deinit self, which suppresses __del__/release),
/// and this entrypoint adopts that single reference net-zero.
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
MGP_RT_CreateAsyncDeviceBufferRefByTakingHandle(
    Driver::DeviceBuffer *handle, void *data, size_t size,
    AsyncRTWrapper<AnyAsyncValueRef> async) {
  CPUDevice &cpuDevice = *CPUDevice::getCurrentCPUDeviceOrNull();
  AnyAsyncValueRef &value = unwrap(async);

  // TAKE the reference the Mojo owner just surrendered via take_handle(). The
  // pairing is load-bearing: take_handle is `deinit self` (no release) and
  // takeRCRef adopts the handle WITHOUT addRef, so the single live reference is
  // transferred net-zero -- no addRef, no fresh DeviceBuffer
  // (createBufferOwning would double-own the same allocation -> double free),
  // no releasePtr. Using copyRCRef here would leak +1, since the Mojo side no
  // longer releases.
#if MODULAR_HAL_DEVICE_CONTEXT
  // Under HAL the consumed handle is the heap-boxed Mojo DeviceBuffer;
  // wrap it in a fresh shim DeviceBuffer that owns the box.
  Driver::DeviceBufferRef buf = Driver::DeviceBuffer::adoptMojoBox(
      reinterpret_cast<MojoHALBufferBox *>(handle), data, size);
#else
  Driver::DeviceBufferRef buf = M::takeRCRef(handle);
#endif

  // Wrap the DeviceBuffer in an AsyncValue.
  auto asyncDeviceBufferRef =
      AnyAsyncValueRef::createReady<Driver::DeviceBufferRef>(cpuDevice,
                                                             std::move(buf));

  // Finally, wrap the pointer and the AsyncValue wrapper in a TensorBufferRef.
  auto tensorBufferRef = ::M::MLRT::TensorBufferRef::create(
      data, size, std::move(asyncDeviceBufferRef));

  fulfillReady<TensorBufferRef>(value, std::move(tensorBufferRef));
}

/// TAKES an already-owning DeviceGraph handle the producer surrendered (via the
/// Mojo `DeviceGraph.take_handle()`) and wraps it in an
/// AsyncValue<DeviceGraphRef>, writing it into the async slot. The device-graph
/// analogue of MGP_RT_CreateAsyncDeviceBufferRefByTakingHandle: it packs the
/// graph produced by `mgp.device_graph.create` so that
/// `mgp.device_graph.execute` can consume it as a first-class `DeviceGraphRef`
/// (rather than an opaque Mojo value). The single live reference is adopted
/// net-zero, mirroring the take_handle/takeRCRef pairing used for device
/// buffers.
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
MGP_RT_CreateAsyncDeviceGraphRefByTakingHandle(
    Driver::DeviceGraph *handle, AsyncRTWrapper<AnyAsyncValueRef> async) {
  AnyAsyncValueRef &value = unwrap(async);
  Driver::DeviceGraphRef graph = M::takeRCRef(handle);
  fulfillReady<Driver::DeviceGraphRef>(value, std::move(graph));
}

/// Builds a TensorBufferRef from a data view (ptr + size) and an owning storage
/// handle (an AsyncValue* the Mojo side surrendered net-zero), writing it into
/// the async slot. Generalizes MGP_RT_CreateAsyncDeviceBufferRefByTakingHandle:
/// the storage may wrap a DeviceBufferRef, or (once producers are unified) be
/// non-tracked or borrowed.
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
MGP_RT_CreateAsyncBufferRefFromStorage(AsyncValue *storage, void *data,
                                       size_t size,
                                       AsyncRTWrapper<AnyAsyncValueRef> async) {
  AnyAsyncValueRef &value = unwrap(async);
  // Adopt the surrendered reference net-zero, then move it into the
  // TensorBufferRef's storage.
  AnyAsyncValueRef storageRef = AnyAsyncValueRef::take(storage);
  auto tensorBufferRef =
      TensorBufferRef::create(data, size, std::move(storageRef));

  fulfillReady<TensorBufferRef>(value, std::move(tensorBufferRef));
}

/// Builds a Tensor (TensorBufferRef + spec) from a data view (ptr + size +
/// shape/dtype) and an owning storage handle (an AsyncValue* the Mojo side
/// surrendered net-zero), writing it into the async slot. The tensor
/// counterpart of MGP_RT_CreateAsyncBufferRefFromStorage.
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
MGP_RT_CreateAsyncTensorRefFromStorage(AsyncValue *storage, void *data,
                                       size_t size, size_t rank, size_t *dims,
                                       int8_t type,
                                       AsyncRTWrapper<AnyAsyncValueRef> async) {
  AnyAsyncValueRef &value = unwrap(async);
  // Adopt the surrendered reference net-zero, then move it into the tensor's
  // TensorBufferRef storage.
  AnyAsyncValueRef storageRef = AnyAsyncValueRef::take(storage);
  TensorBufferRef buf =
      TensorBufferRef::create(data, size, std::move(storageRef));
  TensorSpec spec(ArrayRef<size_t>(dims, rank), DType(type));

  fulfillReady<Tensor>(value, std::move(buf), std::move(spec));
}

/// Builds a *new* AsyncValue holding a TensorBufferRef from a data view (ptr +
/// size) and an owning storage handle (surrendered net-zero), returning the
/// owning AsyncValue* (refcount 1) for a Mojo AnyAsyncValueRef to adopt. Unlike
/// MGP_RT_CreateAsyncBufferRefFromStorage, which fills a runtime-provided async
/// slot, this allocates a fresh AsyncValue -- used to register a graph output.
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT AsyncValue *
MGP_RT_CreateBufferRefAsyncValue(AsyncValue *storage, void *data, size_t size) {
  CPUDevice &cpuDevice = *CPUDevice::getCurrentCPUDeviceOrNull();
  AnyAsyncValueRef storageRef = AnyAsyncValueRef::take(storage);
  auto tensorBufferRef =
      TensorBufferRef::create(data, size, std::move(storageRef));
  AnyAsyncValueRef value = AnyAsyncValueRef::createReady<TensorBufferRef>(
      cpuDevice, std::move(tensorBufferRef));
  return value.releasePointer();
}

/// Builds a *new* AsyncValue holding a Tensor (TensorBufferRef + spec) from a
/// data view and an owning storage handle, returning the owning AsyncValue*.
/// The tensor counterpart of MGP_RT_CreateBufferRefAsyncValue.
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT AsyncValue *
MGP_RT_CreateTensorRefAsyncValue(AsyncValue *storage, void *data, size_t size,
                                 size_t rank, size_t *dims, int8_t type) {
  CPUDevice &cpuDevice = *CPUDevice::getCurrentCPUDeviceOrNull();
  AnyAsyncValueRef storageRef = AnyAsyncValueRef::take(storage);
  TensorBufferRef buf =
      TensorBufferRef::create(data, size, std::move(storageRef));
  TensorSpec spec(ArrayRef<size_t>(dims, rank), DType(type));
  AnyAsyncValueRef value = AnyAsyncValueRef::createReady<Tensor>(
      cpuDevice, std::move(buf), std::move(spec));
  return value.releasePointer();
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
MGP_RT_CreateAsyncTensorShape(ssize_t *data, ssize_t rank,
                              AsyncRTWrapper<AnyAsyncValueRef> async) {
  AnyAsyncValueRef &value = unwrap(async);

  ArrayRef<ssize_t> dims(data, data + rank);
  fulfillReady<TensorShape>(value, dims);
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
MGP_RT_CreateOwnedAsyncMojoValue(void *data, void (*destructorFn)(void *),
                                 AsyncRTWrapper<AnyAsyncValueRef> async) {
  AnyAsyncValueRef &value = unwrap(async);
  fulfillReady<MojoValue>(value, data, destructorFn);
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void *
MGP_RT_MojoValueAllocateBuffer(size_t size, size_t align) {
  return MojoValue::allocateBuffer(size, align);
}

//===----------------------------------------------------------------------===//
// Unpacking functions for reading async values
//===----------------------------------------------------------------------===//

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void *
MGP_RT_GetValueFromAsync(AsyncRTWrapper<AnyAsyncValueRef> async) {
  AnyAsyncValueRef &value = unwrap(async);
  if (value.getPointer() && value.getPointer()->isIndirect())
    return value.getPointer()->getUnderlyingPtr();
  else
    return value.getPointerToData();
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void *
MGP_RT_UnpackStateContext(AsyncRTWrapper<AnyAsyncValueRef> async) {
  AnyAsyncValueRef &value = unwrap(async);
  assert(value.isReady() &&
         "MGP_RT_UnpackStateContext: async value not yet ready");
  assert(value.isType<MLRT::StateContext>() &&
         "MGP_RT_UnpackStateContext: async value does not hold a "
         "StateContext — argument index mismatch or wrong async value");
  auto &stateContextRef = value.get<StateContext>();
  return &stateContextRef;
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void *
MGP_RT_UnpackDeviceContext(AsyncRTWrapper<AnyAsyncValueRef> async) {
  AnyAsyncValueRef &value = unwrap(async);
  assert(value.isReady() &&
         "MGP_RT_UnpackDeviceContext: async value not yet ready");
  assert(value.isType<Driver::DeviceContextRef>() &&
         "MGP_RT_UnpackDeviceContext: async value does not hold a "
         "DeviceContextRef — argument index mismatch or wrong async value");
  auto deviceContextRef = value.get<Driver::DeviceContextRef>();
  auto *ptr =
      const_cast<Driver::DeviceContext *>(deviceContextRef.getPointer());
  assert(ptr != nullptr &&
         "MGP_RT_UnpackDeviceContext: DeviceContextRef holds a null pointer");
#if MODULAR_HAL_DEVICE_CONTEXT
  // Under the HAL backend the Mojo `DeviceContext` is the source of truth and
  // the C++ object merely wraps it. The kernel ABI reconstructs a Mojo
  // `DeviceContext` from this handle (DeviceContext._from_unsafe_cpp_handle),
  // so hand out the boxed Mojo `DeviceContext` rather than the C++ wrapper.
  return ptr->mojoHandle();
#else
  return ptr;
#endif
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void *
MGP_RT_GetDataFromBuffer(AsyncRTWrapper<AnyAsyncValueRef> async,
                         size_t *sizeOut) {
  AnyAsyncValueRef &value = unwrap(async);
  auto &buffer = value.get<TensorBufferRef>();
  *sizeOut = buffer.getSize();
  return buffer.getBuffer();
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void *
MGP_RT_GetShapeAndDataFromTensor(size_t *shape,
                                 AsyncRTWrapper<AnyAsyncValueRef> async) {
  AnyAsyncValueRef &value = unwrap(async);
  assert(value.isReady());
  auto &tensor = value.get<Tensor>();
  const TensorSpec &spec = tensor.getSpec();
  for (size_t i = 0; i < spec.getRank(); ++i)
    shape[i] = spec[i];
  return tensor.getMutableBuffer();
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
MGP_RT_GetTensorShapeFromAsync(ssize_t *data, ssize_t rank,
                               AsyncRTWrapper<AnyAsyncValueRef> async) {
  AnyAsyncValueRef &value = unwrap(async);
  assert(value.isReady());
  auto &shape = value.get<TensorShape>();
  for (int i = 0; i < rank; ++i)
    data[i] = shape[i];
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
MGP_RT_DestructAsyncRefs(size_t size, void **storageRefPtr) {
  for (size_t i = 0; i < size; ++i) {
    AnyAsyncValueRef *refAddr =
        reinterpret_cast<AnyAsyncValueRef *>(storageRefPtr[i]);
    // The refAddr could be nullptr here since compiled model tries to destruct
    // all async values upon failure and some of them might not have been
    // initialized.
    if (refAddr && refAddr->getPointer())
      refAddr->reset();
  }
}

//===----------------------------------------------------------------------===//
// debug print function
//===----------------------------------------------------------------------===//

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
MGP_RT_DebugTensorPrint(const char *label, size_t labelLen, uint8_t dtype,
                        size_t *shapeArray, size_t shapeLen, void *bufAddr,
                        size_t bufLen) {
  auto buffer = TensorBufferRef::createWithNonTrackedMemory(bufAddr, bufLen,
                                                            std::nullopt);
  TensorSpec spec(ArrayRef(shapeArray, shapeLen), DType(dtype));

  StringRef labelRef(label, labelLen);
  if (labelRef.empty()) {
    // Metadata only.
    llvm::outs() << "Tensor(" << spec << ", " << buffer << ")\n";
    llvm::outs().flush();
  } else {
    // Buffer contents, using tensor spec for shape. This is only possible for
    // CPU buffers. The printTensor function will crash for GPU buffers, so this
    // primitive assumes that the buffer is a CPU buffer. I don't see a good way
    // to assert on this condition though.
    ContextRef context = Init::getContext();

    DebugTensorPrintOptions defaultOptions;
    DebugTensorPrintOptions *options = &defaultOptions;
    if (context)
      if (DebugTensorPrintOptions *opts =
              context->get<DebugTensorPrintOptions>())
        options = opts;
    auto errOr = options->printTensor(buffer.getBuffer(), spec, labelRef);
    if (errOr.isError())
      llvm_unreachable(errOr.getError());
  }
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
MGP_RT_AsyncRT_CreateAsync_Error(AsyncRTWrapper<AnyAsyncValueRef> async,
                                 const char *messagePtr, size_t messageLen) {
  StringRef errorMsg(messagePtr, messageLen);
  CPUDevice &cpuDevice = *CPUDevice::getCurrentCPUDeviceOrNull();
  AnyAsyncValueRef &value = unwrap(async);
  EncodedDiagnostic diagnostic{Twine(errorMsg),
                               UnknownLocationDecoder::getEncodedLocation()};
  if (value.getPointer() && value.getPointer()->isIndirect())
    value.copy().setToError(std::move(diagnostic));
  else
    value = value.createError(cpuDevice, std::move(diagnostic));
}
