//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Bindings between C++ primitive implementation and mojo.
//
//===----------------------------------------------------------------------===//

#include "AsyncRT/Runtime/AsyncValueRef.h"
#include "Runtime/Tensor/StateContext.h"
#include "Support/SymbolExport.h"
#include <iostream>

using namespace M;
using namespace M::AsyncRT;

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
