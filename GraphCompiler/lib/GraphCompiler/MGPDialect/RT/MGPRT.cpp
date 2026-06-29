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

#include "GraphCompiler/Support/List.h"
#include "Init/Init.h"
#include "MLRT/AsyncRT/ForwardDecls.h"
#include "MLRT/AsyncRT/Runtime/AnyAsyncValueRef.h"
#include "MLRT/AsyncRT/Runtime/AsyncValue.h"
#include "MLRT/AsyncRT/Runtime/CPUDevice.h"
#include "MLRT/AsyncRT/Support/UnknownLocationDecoder.h"
#include "MLRT/Core/MojoValue.h"
#include "MLRT/Core/Tensor/StateContext.h"
#include "MLRT/Core/Tensor/TensorBufferRef.h"
#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "Support/Context.h"
#include "Support/ML/DebugPrint.h"
#include "Support/SymbolExport.h"

using namespace M;
using namespace M::MLRT;

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

enum BorroweeType : size_t {
  kHandle = 0,
  kBuffer = 1,
  kTensor = 2,
};

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

/// This is the C++ implementation of mogg.async.unpack.
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void *
MGP_RT_AsyncUnpack(AsyncRTWrapper<AnyAsyncValueRef> async) {
  AnyAsyncValueRef &value = unwrap(async);
  if (value.getPointer() && value.getPointer()->isIndirect())
    return value.getPointer()->getUnderlyingPtr();
  else
    return value.getPointerToData();
}

/// This is the old C++ implementation of mgp.buffer.get_cached. The new one
/// starts with TMP_.
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void *
MGP_RT_GetCachedBuffer(size_t bufferSlot, StateContext &stateCtx, size_t *size,
                       void *storageRefPtr) {
  assert(storageRefPtr != nullptr);

  auto &stateSlot = stateCtx.getStateSlot(bufferSlot);
  ErrorOr<TensorBufferRef> errOr = stateSlot.getBuffer(/*bufferIndex=*/0);
  if (errOr)
    return nullptr;

  TensorBufferRef ref = errOr.takeValue();
  // Emplace construction using the storage_ref of the TensorBufferRef.
  new (storageRefPtr) AnyAsyncValueRef(ref.getMemStorageHandle());
  *size = ref.getSize();
  return ref.getBuffer();
}

/// This is the C++ implementation of mgp.buffer.remove_cached.
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
MGP_RT_RemoveCachedBuffer(size_t bufferSlot, StateContext &stateCtx) {
  auto &stateSlot = stateCtx.getStateSlot(bufferSlot);
  stateSlot.clear();
}

/// This is the C++ implementation of mgp.int.cached.
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
MGP_RT_SetCachedInt(size_t bufferSlot, StateContext *stateCtx, size_t value) {
  stateCtx->emplaceStateSlot<IntStateSlot>(bufferSlot,
                                           /*numBuffers=*/0, value);
}

/// This is the C++ implementation of mgp.int.get_cached.
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT size_t
MGP_RT_GetCachedInt(size_t bufferSlot, StateContext *stateCtx) {
  auto &slot = stateCtx->getStateSlot(bufferSlot);
  return slot.get<IntStateSlot>().getValue();
}

/// Returns the size of a C++ list.
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT int64_t
MGP_RT_ListSize(AsyncRTWrapper<GML::List> asyncList) {
  return unwrap(asyncList).getElements().size();
}

/// Gets the data and shape of a given ptr.
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
MGP_RT_ListPopulate(AsyncRTWrapper<GML::List> asyncList, void **dataPtr,
                    int64_t *shapePtr) {
  GML::List &list = unwrap(asyncList);

  int64_t tensorId = 0;
  for (GML::Variant &ref : list.getElements()) {
    if (ref.getValue().isType<Tensor>()) {
      auto &runtimeTensor = ref.getValue().get<Tensor>();
      const TensorShape &shape = runtimeTensor.getShape();
      dataPtr[tensorId] = const_cast<void *>(runtimeTensor.getBuffer());

      for (size_t i = 0; i < shape.getRank(); ++i)
        shapePtr[tensorId * shape.getRank() + i] = shape[i];

      // Promote scalar tensors to 1D.
      if (shape.getRank() == 0)
        shapePtr[tensorId] = 1;
      ++tensorId;
    }
  }
}

/// This is the new C++ implementation of the mgp.buffer.get_cached primitive.
/// The old one is not prefixed with TMP_.
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void *
TMP_MGP_RT_GetCachedBuffer(size_t bufferSlot, StateContext *stateCtx,
                           size_t *size, void **data) {
  auto &stateSlot = stateCtx->getStateSlot(bufferSlot);
  ErrorOr<TensorBufferRef &> errOr = stateSlot.getBufferRef(/*bufferIndex=*/0);

  /// TODO: Support errors in MGPRT
  if (errOr) {
  }

  TensorBufferRef &ref = errOr.get();
  *size = ref.getSize();
  *data = ref.getBuffer();

  return ref.getMemStorageHandleRef();
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
TMP_MGP_RT_RemoveCachedBuffer(size_t bufferSlot, StateContext *stateCtx) {
  auto &stateSlot = stateCtx->getStateSlot(bufferSlot);
  stateSlot.clear();
}

/// This is the implementation of the mgp.buffer.borrow primitive.
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
MGP_RT_BufferBorrow(AsyncRTWrapper<AnyAsyncValueRef> borrower,
                    TensorBufferRef *borrowee) {
  AnyAsyncValueRef &borrowerValue = unwrap(borrower);
  TensorBufferRef buf = ::M::MLRT::TensorBufferRef::create(
      borrowee->getBuffer(), borrowee->getSize(),
      borrowee->getMemStorageHandle());
  if (borrowerValue.getPointer() && borrowerValue.getPointer()->isIndirect()) {
    borrowerValue.copy().emplaceIndirect<TensorBufferRef>(std::move(buf));
  } else {
    CPUDevice &cpuDevice = *CPUDevice::getCurrentCPUDeviceOrNull();
    borrowerValue =
        borrowerValue.createReady<TensorBufferRef>(cpuDevice, std::move(buf));
  }
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
MGP_RT_TensorBorrowV2(AsyncRTWrapper<AnyAsyncValueRef> borrower, void *data,
                      size_t byteCount, size_t rank, size_t *dims, int8_t type,
                      AnyAsyncValueRef *mem) {
  AnyAsyncValueRef &borrowerValue = unwrap(borrower);

  TensorBufferRef buf =
      mem ? ::M::MLRT::TensorBufferRef::create(data, byteCount, mem->copy())
          : ::TensorBufferRef::createWithNonTrackedMemory(data, byteCount);

  // Pack buffer and spec into tensor.
  TensorSpec spec(ArrayRef<size_t>(dims, rank), DType(type));
  if (borrowerValue.getPointer() && borrowerValue.getPointer()->isIndirect())
    borrowerValue.copy().emplaceIndirect<Tensor>(std::move(buf),
                                                 std::move(spec));
  else {
    CPUDevice &rt = *CPUDevice::getCurrentCPUDeviceOrNull();
    borrowerValue =
        borrowerValue.createReady<Tensor>(rt, std::move(buf), std::move(spec));
  }
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
MGP_RT_BufferBorrowV2(AsyncRTWrapper<AnyAsyncValueRef> borrower, void *data,
                      size_t size, AnyAsyncValueRef *mem) {
  AnyAsyncValueRef &borrowerValue = unwrap(borrower);
  TensorBufferRef buf =
      mem ? ::M::MLRT::TensorBufferRef::create(data, size, mem->copy())
          : ::TensorBufferRef::createWithNonTrackedMemory(data, size);
  if (borrowerValue.getPointer() && borrowerValue.getPointer()->isIndirect()) {
    borrowerValue.copy().emplaceIndirect<TensorBufferRef>(std::move(buf));
  } else {
    CPUDevice &cpuDevice = *CPUDevice::getCurrentCPUDeviceOrNull();
    borrowerValue =
        borrowerValue.createReady<TensorBufferRef>(cpuDevice, std::move(buf));
  }
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
MGP_RT_BufferBorrowForTensorRef(AsyncRTWrapper<AnyAsyncValueRef> borrower,
                                TensorBufferRef *borrowee,
                                AnyAsyncValueRef *mem) {
  return MGP_RT_BufferBorrowV2(borrower, borrowee->getBuffer(),
                               borrowee->getSize(), mem);
}

/// Assigns to the borrower a tensor buffer that isn't owned by the cpuDevice
/// (used for constants).
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
MGP_RT_BufferUntracked(AsyncRTWrapper<AnyAsyncValueRef> borrower, void *data,
                       size_t size) {
  AnyAsyncValueRef &borrowerValue = unwrap(borrower);
  TensorBufferRef buf =
      ::M::MLRT::TensorBufferRef::createWithNonTrackedMemory(data, size);
  if (borrowerValue.getPointer() && borrowerValue.getPointer()->isIndirect()) {
    borrowerValue.copy().emplaceIndirect<TensorBufferRef>(std::move(buf));
  } else {
    CPUDevice &cpuDevice = *CPUDevice::getCurrentCPUDeviceOrNull();
    borrowerValue =
        borrowerValue.createReady<TensorBufferRef>(cpuDevice, std::move(buf));
  }
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void *
MGP_RT_GetContextPayloadPtr(size_t index, StateContext *ctx) {
  return ctx->getStateSlot(index).getUnderlyingPointer();
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
MGP_RT_CreateAsync_ssizet(ssize_t data,
                          AsyncRTWrapper<AnyAsyncValueRef> async) {
  CPUDevice &cpuDevice = *CPUDevice::getCurrentCPUDeviceOrNull();
  AnyAsyncValueRef &value = unwrap(async);
  if (value.getPointer() && value.getPointer()->isIndirect()) {
    value.copy().emplaceIndirect<size_t>(data);
  } else {
    assert(!value.isReady());
    value = value.createReady<size_t>(cpuDevice, data);
  }
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
MGP_RT_CreateAsync_int64t(int64_t data,
                          AsyncRTWrapper<AnyAsyncValueRef> async) {
  CPUDevice &cpuDevice = *CPUDevice::getCurrentCPUDeviceOrNull();
  AnyAsyncValueRef &value = unwrap(async);
  if (value.getPointer() && value.getPointer()->isIndirect()) {
    value.copy().emplaceIndirect<int64_t>(data);
  } else {
    assert(!value.isReady());
    value = value.createReady<int64_t>(cpuDevice, data);
  }
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
MGP_RT_CreateAsync_chain(AsyncRTWrapper<AnyAsyncValueRef> async) {
  CPUDevice &cpuDevice = *CPUDevice::getCurrentCPUDeviceOrNull();
  AnyAsyncValueRef &value = unwrap(async);
  if (value.getPointer() && value.getPointer()->isIndirect()) {
    value.copy().emplaceIndirect<Chain>();
  } else {
    assert(!value.isReady());
    value = value.createReady<Chain>(cpuDevice);
  }
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
MGP_RT_CreateAsync_bool(bool data, AsyncRTWrapper<AnyAsyncValueRef> async) {
  CPUDevice &cpuDevice = *CPUDevice::getCurrentCPUDeviceOrNull();
  AnyAsyncValueRef &value = unwrap(async);
  if (value.getPointer() && value.getPointer()->isIndirect()) {
    value.copy().emplaceIndirect<bool>(data);
  } else {
    assert(!value.isReady());
    value = value.createReady<bool>(cpuDevice, data);
  }
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
  Driver::DeviceBufferRef buf = M::takeRCRef(handle);

  // Wrap the DeviceBuffer in an AsyncValue.
  auto asyncDeviceBufferRef =
      AnyAsyncValueRef::createReady<Driver::DeviceBufferRef>(cpuDevice,
                                                             std::move(buf));

  // Finally, wrap the pointer and the AsyncValue wrapper in a TensorBufferRef.
  auto tensorBufferRef = ::M::MLRT::TensorBufferRef::create(
      data, size, std::move(asyncDeviceBufferRef));

  if (value.getPointer() && value.getPointer()->isIndirect()) {
    value.copy().emplaceIndirect<TensorBufferRef>(std::move(tensorBufferRef));
  } else {
    assert(!value.isReady());
    value = value.createReady<TensorBufferRef>(cpuDevice,
                                               std::move(tensorBufferRef));
  }
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
MGP_RT_CreateAsyncNonTrackedBufferRef(void *data, size_t size,
                                      AsyncRTWrapper<AnyAsyncValueRef> async) {
  CPUDevice &cpuDevice = *CPUDevice::getCurrentCPUDeviceOrNull();
  AnyAsyncValueRef &value = unwrap(async);
  if (value.getPointer() && value.getPointer()->isIndirect()) {
    value.copy().emplaceIndirect<TensorBufferRef>(
        TensorBufferRef::createWithNonTrackedMemory(
            data, size, /*alignment=*/std::nullopt));
  } else {
    assert(!value.isReady());
    value = value.createReady<TensorBufferRef>(
        cpuDevice, TensorBufferRef::createWithNonTrackedMemory(
                       data, size, /*alignment=*/std::nullopt));
  }
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
MGP_RT_CreateAsyncNonTrackedTensor(void *data, size_t byteCount, size_t rank,
                                   size_t *dims, int8_t type,
                                   AsyncRTWrapper<AnyAsyncValueRef> async) {

  CPUDevice &cpuDevice = *CPUDevice::getCurrentCPUDeviceOrNull();
  AnyAsyncValueRef &value = unwrap(async);

  // Pack buffer and spec into tensor.
  TensorBufferRef buf = ::M::MLRT::TensorBufferRef::createWithNonTrackedMemory(
      data, byteCount, /*alignment=*/std::nullopt);
  TensorSpec spec(ArrayRef<size_t>(dims, rank), DType(type));
  if (value.getPointer() && value.getPointer()->isIndirect())
    value.copy().emplaceIndirect<Tensor>(std::move(buf), std::move(spec));
  else
    value =
        value.createReady<Tensor>(cpuDevice, std::move(buf), std::move(spec));
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
MGP_RT_CreateAsyncBufferWithBorrow(
    void *data, size_t size, AsyncRTWrapper<AnyAsyncValueRef> toBorrowFrom,
    size_t borroweeType, AsyncRTWrapper<AnyAsyncValueRef> async) {
  CPUDevice &cpuDevice = *CPUDevice::getCurrentCPUDeviceOrNull();
  AnyAsyncValueRef &outVal = unwrap(async);
  AnyAsyncValueRef &handleOrTensor = unwrap(toBorrowFrom);
  AnyAsyncValueRef handle;
  if (borroweeType == kHandle) {
    handle = std::move(handleOrTensor);
  } else if (borroweeType == kBuffer) {
    // Use the lifetime of the other tensor by sharing the same storage handle.
    handle = handleOrTensor.get<TensorBufferRef>().getMemStorageHandle();
  } else {
    assert(borroweeType == kTensor);
    handle = handleOrTensor.get<Tensor>().getBufferRef().getMemStorageHandle();
  }
  TensorBufferRef buf =
      ::M::MLRT::TensorBufferRef::create(data, size, std::move(handle));

  // Emplace into the async value.
  if (outVal.getPointer() && outVal.getPointer()->isIndirect()) {
    outVal.copy().emplaceIndirect<TensorBufferRef>(std::move(buf));
  } else {
    outVal = outVal.createReady<TensorBufferRef>(cpuDevice, std::move(buf));
  }
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
MGP_RT_CreateAsyncTensorShape(ssize_t *data, ssize_t rank,
                              AsyncRTWrapper<AnyAsyncValueRef> async) {
  CPUDevice &cpuDevice = *CPUDevice::getCurrentCPUDeviceOrNull();
  AnyAsyncValueRef &value = unwrap(async);
  llvm::SmallVector<ssize_t> dims;
  for (int i = 0; i < rank; ++i)
    dims.push_back(data[i]);

  if (value.getPointer() && value.getPointer()->isIndirect()) {
    value.copy().emplaceIndirect<TensorShape>(dims);
  } else {
    assert(!value.isReady() &&
           "Value needs to not be ready so we can construct it.");
    value = AnyAsyncValueRef::createReady<TensorShape>(cpuDevice, dims);
  }
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
MGP_RT_CreateAsyncTensorWithBorrow(
    void *data, size_t byteCount, size_t rank, size_t *dims, int8_t type,
    AsyncRTWrapper<AnyAsyncValueRef> toBorrowFrom, size_t borroweeType,
    AsyncRTWrapper<AnyAsyncValueRef> async) {
  CPUDevice &rt = *CPUDevice::getCurrentCPUDeviceOrNull();
  AnyAsyncValueRef &outVal = unwrap(async);

  // Create a borrowed buffer ref.
  AnyAsyncValueRef &handleOrTensor = unwrap(toBorrowFrom);
  AnyAsyncValueRef handle;
  if (borroweeType == kHandle) {
    handle = std::move(handleOrTensor);
  } else if (borroweeType == kBuffer) {
    // Use the lifetime of the other tensor by sharing the same storage handle.
    handle = handleOrTensor.get<TensorBufferRef>().getMemStorageHandle();
  } else {
    assert(borroweeType == kTensor);
    handle = handleOrTensor.get<Tensor>().getBufferRef().getMemStorageHandle();
  }
  TensorBufferRef buf =
      ::M::MLRT::TensorBufferRef::create(data, byteCount, std::move(handle));

  // Pack buffer and spec into tensor.
  TensorSpec spec(ArrayRef<size_t>(dims, rank), DType(type));
  if (outVal.getPointer() && outVal.getPointer()->isIndirect())
    outVal.copy().emplaceIndirect<Tensor>(std::move(buf), std::move(spec));
  else
    outVal = outVal.createReady<Tensor>(rt, std::move(buf), std::move(spec));
}
static void
createAsyncMojoValue(void *data, void (*destructorFn)(void *),
                     AsyncRTWrapper<AnyAsyncValueRef> async,
                     MojoValue::Tag tag = MojoValue::Tag::kDefault) {
  CPUDevice &cpuDevice = *CPUDevice::getCurrentCPUDeviceOrNull();
  AnyAsyncValueRef &value = unwrap(async);
  if (value.getPointer() && value.getPointer()->isIndirect()) {
    value.copy().emplaceIndirect<MojoValue>(data, destructorFn, tag);
  } else {
    assert(!value.isReady());
    value = AnyAsyncValueRef::createReady<MojoValue>(cpuDevice, data,
                                                     destructorFn, tag);
  }
}
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
MGP_RT_CreateOwnedAsyncMojoValue(void *data, void (*destructorFn)(void *),
                                 AsyncRTWrapper<AnyAsyncValueRef> async) {
  createAsyncMojoValue(data, destructorFn, async);
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void *
MGP_RT_MojoValueAllocateBuffer(size_t size, size_t align) {
  return MojoValue::allocateBuffer(size, align);
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
MGP_RT_CreateOwnedAsyncPythonMojoValue(void *data, void (*destructorFn)(void *),
                                       AsyncRTWrapper<AnyAsyncValueRef> async) {
  createAsyncMojoValue(data, destructorFn, async, MojoValue::Tag::kPython);
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
  return ptr;
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

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void *
MGP_RT_GetContextAndSizeFromAsync(size_t *size,
                                  AsyncRTWrapper<AnyAsyncValueRef> async) {
  AnyAsyncValueRef &value = unwrap(async);
  assert(value.isReady());
  auto &ctx = value.get<StateContext>();
  *size = ctx.getNumStateSlots();
  return reinterpret_cast<void *>(&ctx);
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
MGP_RT_DestructAsyncRefs(size_t size, void **storageRefPtr, bool directRef) {
  for (size_t i = 0; i < size; ++i) {
    AnyAsyncValueRef *refAddr =
        directRef ? reinterpret_cast<AnyAsyncValueRef *>(&storageRefPtr[i])
                  : reinterpret_cast<AnyAsyncValueRef *>(storageRefPtr[i]);
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

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
MGP_RT_GetBufferMemStorageHandle(AsyncRTWrapper<AnyAsyncValueRef> buffer,
                                 size_t type,
                                 AnyAsyncValueRef **memStorageHandle) {
  AnyAsyncValueRef &unwrappedBuffer = unwrap(buffer);
  if (type == kBuffer) {
    *memStorageHandle =
        unwrappedBuffer.get<TensorBufferRef>().getMemStorageHandleRef();
  } else if (type == kTensor) {
    *memStorageHandle =
        unwrappedBuffer.get<Tensor>().getBufferRef().getMemStorageHandleRef();
  } else {
    llvm_unreachable("Unexpected buffer type");
  }
}
