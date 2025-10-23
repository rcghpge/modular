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

#include "AsyncRT/Runtime/Runtime.h"
#include "GenericML/Support/List.h"
#include "Runtime/Tensor/StateContext.h"
#include "Support/SymbolExport.h"

using namespace M;

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/**
The following assertion is necessary as long as we are passing StateContext from
Mojo to C++. Even if we are passing references, certain MGP primitives in Mojo
currently expect the StateContext to be passed as a const reference. This is
dangerous because the Mojo code could try to dereference it.

Instead, we should use the aliases that are defined in Mojo. Everything is an
OpaquePointer which is guaranteed to be safe and not to be dereferenced by the
Mojo.

TLDR: Mojo and C++ should communicate with only native types like int, bool and
void*.
 */
static_assert(
    sizeof(StateContext) == 16,
    "Expected StateContext to be 16 bytes wide to match the Mojo equivalent");

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
      tensorId++;
    }
  }
}

/// This is the new C++ implementation of the mgp.buffer.get_cached primitive.
/// The old one is not prefixed with TMP_.
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void *
TMP_MGP_RT_GetCachedBuffer(size_t bufferSlot,
                           AsyncRTWrapper<AnyAsyncValueRef> stateCtxAsync) {
  void *stateCtxPtr = MGP_RT_AsyncUnpack(stateCtxAsync);
  StateContext &stateCtx = *reinterpret_cast<StateContext *>(stateCtxPtr);

  auto &stateSlot = stateCtx.getStateSlot(bufferSlot);
  ErrorOr<TensorBufferRef &> errOr = stateSlot.getBufferRef(/*bufferIndex=*/0);
  /// TODO: Support errors in MGPRT
  if (errOr) {
  }

  return &errOr.get();
}

/// This is the implementation of the mgp.buffer.borrow primitive.
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
MGP_RT_BufferBorrow(AsyncRTWrapper<AnyAsyncValueRef> borrower,
                    TensorBufferRef *borrowee) {
  AnyAsyncValueRef &borrowerValue = unwrap(borrower);
  TensorBufferRef buf =
      ::M::TensorBufferRef::create(borrowee->getBuffer(), borrowee->getSize(),
                                   borrowee->getMemStorageHandle());
  if (borrowerValue.getPointer() && borrowerValue.getPointer()->isIndirect()) {
    borrowerValue.copy().emplaceIndirect<TensorBufferRef>(std::move(buf));
  } else {
    Runtime &runtime = *Runtime::getCurrentRuntimeOrNull();
    borrowerValue =
        borrowerValue.createReady<TensorBufferRef>(runtime, std::move(buf));
  }
}
