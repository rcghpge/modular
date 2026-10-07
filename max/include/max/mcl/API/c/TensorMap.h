//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_API_TENSOR_MAP_H
#define GENERICML_API_TENSOR_MAP_H

#include "AsyncRT/ForwardDecls.h"
#include "AsyncRT/Runtime/CPUDevice.h"
#include "MLRT/Core/Tensor/Tensor.h"

namespace M {

/// A `TensorMap` owns the a map from tensor names to async tensor types along
/// with a cpuDevice used for executing a model.
struct TensorMap {

  TensorMap(AsyncRT::CPUDevice &cpuDevice) : cpuDevice{cpuDevice} {}

  TensorMap(llvm::StringMap<AnyAsyncValueRef> tensors,
            AsyncRT::CPUDevice &cpuDevice)
      : tensors{std::move(tensors)}, cpuDevice{cpuDevice} {}

  /// Map of tensor names to  async tensor values useful for representing inputs
  /// or outputs in `M_executeModelSync`.
  llvm::StringMap<AsyncRT::AnyAsyncValueRef> tensors;

  /// The underlying cpuDevice used for execution
  AsyncRT::CPUDevice &cpuDevice;
};

} // namespace M

#endif // GENERICML_API_TENSOR_MAP_H
