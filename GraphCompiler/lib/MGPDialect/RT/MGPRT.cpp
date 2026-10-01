//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Bindings between C++ primitive implementation and mojo. This is the successor
// to putting things in Mojo/lib/CompilerRT/AsyncRT.cpp
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

template <typename T, typename... Args>
AsyncValue *createReadyValue(Args &&...args) {
  CPUDevice &cpuDevice = *CPUDevice::getCurrentCPUDeviceOrNull();
  AnyAsyncValueRef value =
      AnyAsyncValueRef::createReady<T>(cpuDevice, std::forward<Args>(args)...);
  return value.releasePointer();
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
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT AsyncValue *
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

  // Return an owning reference to the buffer's backing storage so the Mojo
  // caller can keep the memory alive without punning AnyAsyncValueRef layout.
  return ref.getMemStorageHandle().copy().releasePointer();
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

//===----------------------------------------------------------------------===//
// Value-returning async creation helpers (used by the EmitsMojo backend when
// the producer emits a value and the terminator assigns it to the output slot).
//===----------------------------------------------------------------------===//

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT AsyncValue *
MGP_RT_CreateAsync_ssizet(ssize_t data) {
  return createReadyValue<ssize_t>(data);
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT AsyncValue *
MGP_RT_CreateAsync_int64t(int64_t data) {
  return createReadyValue<int64_t>(data);
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT AsyncValue *
MGP_RT_CreateAsync_bool(bool data) {
  return createReadyValue<bool>(data);
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT AsyncValue *
MGP_RT_CreateAsync_chain() {
  return createReadyValue<Chain>();
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT AsyncValue *
MGP_RT_CreateOwnedAsyncMojoValue(void *data, void (*destructorFn)(void *)) {
  return createReadyValue<MojoValue>(data, destructorFn);
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT AsyncValue *
MGP_RT_CreateAsyncDeviceGraphRefByTakingHandle(Driver::DeviceGraph *handle) {
  Driver::DeviceGraphRef graph = M::takeRCRef(handle);
  return createReadyValue<Driver::DeviceGraphRef>(std::move(graph));
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT AsyncValue *
MGP_RT_AsyncRT_CreateAsync_Error(const char *messagePtr, size_t messageLen) {
  StringRef errorMsg(messagePtr, messageLen);
  CPUDevice &cpuDevice = *CPUDevice::getCurrentCPUDeviceOrNull();
  EncodedDiagnostic diagnostic{Twine(errorMsg),
                               UnknownLocationDecoder::getEncodedLocation()};
  AnyAsyncValueRef value =
      AnyAsyncValueRef::createError(cpuDevice, std::move(diagnostic));
  return value.releasePointer();
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
MGP_RT_AssignAsyncValue(AsyncRTWrapper<AnyAsyncValueRef> slot,
                        AsyncValue *src) {
  AnyAsyncValueRef &slotRef = unwrap(slot);
  AnyAsyncValueRef srcRef = AnyAsyncValueRef::take(src);

  if (slotRef.getPointer() && slotRef.getPointer()->isIndirect()) {
    slotRef.copy().resolveIndirect(std::move(srcRef));
  } else {
    slotRef = std::move(srcRef);
  }
}

/// Builds a *new* AsyncValue holding a TensorBufferRef from a data view (ptr +
/// size) and an owning storage handle (surrendered net-zero), returning the
/// owning AsyncValue* (refcount 1) for a Mojo AnyAsyncValueRef to adopt.
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

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT AsyncValue *
MGP_RT_CreateAsyncTensorShape(ssize_t *data, size_t rank) {
  ArrayRef<ssize_t> dims(data, data + rank);
  return createReadyValue<TensorShape>(dims);
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void *
MGP_RT_MojoValueAllocateBuffer(size_t size, size_t align) {
  return MojoValue::allocateBuffer(size, align);
}

//===----------------------------------------------------------------------===//
// Unpacking functions for reading async values
//===----------------------------------------------------------------------===//

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void *
MGP_RT_GetValueFromAsync(AsyncValue *async) {
  return async->getUnderlyingPtr();
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void *
MGP_RT_UnpackStateContext(AsyncValue *async) {
  assert(async && "MGP_RT_UnpackStateContext: null async value");
  assert(async->isReady() &&
         "MGP_RT_UnpackStateContext: async value not yet ready");
  assert(async->isType<MLRT::StateContext>() &&
         "MGP_RT_UnpackStateContext: async value does not hold a "
         "StateContext — argument index mismatch or wrong async value");
  auto &stateContextRef = async->get<StateContext>();
  return &stateContextRef;
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void *
MGP_RT_UnpackDeviceContext(AsyncValue *async) {
  assert(async && "MGP_RT_UnpackDeviceContext: null async value");
  assert(async->isReady() &&
         "MGP_RT_UnpackDeviceContext: async value not yet ready");
  assert(async->isType<Driver::DeviceContextRef>() &&
         "MGP_RT_UnpackDeviceContext: async value does not hold a "
         "DeviceContextRef — argument index mismatch or wrong async value");
  auto deviceContextRef = async->get<Driver::DeviceContextRef>();
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
MGP_RT_GetDataFromBuffer(AsyncValue *async, size_t *sizeOut) {
  assert(async && "MGP_RT_GetDataFromBuffer: null async value");
  auto &buffer = async->get<TensorBufferRef>();
  *sizeOut = buffer.getSize();
  return buffer.getBuffer();
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void *
MGP_RT_GetShapeAndDataFromTensor(size_t *shape, AsyncValue *async) {
  assert(async && "MGP_RT_GetShapeAndDataFromTensor: null async value");
  assert(async->isReady());
  auto &tensor = async->get<Tensor>();
  const TensorSpec &spec = tensor.getSpec();
  for (size_t i = 0; i < spec.getRank(); ++i)
    shape[i] = spec[i];
  return tensor.getMutableBuffer();
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
MGP_RT_GetTensorShapeFromAsync(ssize_t *data, ssize_t rank, AsyncValue *async) {
  assert(async && "MGP_RT_GetTensorShapeFromAsync: null async value");
  assert(async->isReady());
  auto &shape = async->get<TensorShape>();
  for (int i = 0; i < rank; ++i)
    data[i] = shape[i];
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
