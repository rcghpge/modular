//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Bindings between C++ primitive implementation and mojo.
//
//===----------------------------------------------------------------------===//

#include "GenericML/Support/List.h"
#include "Runtime/Tensor/StateContext.h"
#include "Support/SymbolExport.h"

using namespace M;
using namespace M::AsyncRT;

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

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void MGP_RT_TEST() {
  llvm::outs() << "MGP_RT_TEST\n";
}

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
