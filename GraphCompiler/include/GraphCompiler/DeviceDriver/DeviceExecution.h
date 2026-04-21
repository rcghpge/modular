//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_DEVICEDRIVER_DEVICEEXECUTION_H
#define GENERICML_DEVICEDRIVER_DEVICEEXECUTION_H

#include "GraphCompiler/DeviceDriver/Driver.h"
#include "MLRT/Core/Tensor/Tensor.h"
#include "Support/Context.h"
#include "Support/ML/TensorSpec.h"
#include "Support/SymbolExport.h"

namespace M::GraphCompiler::Driver {

// Construct a DeviceMemory block to hold the provided tensor.
MODULAR_VISIBILITY_EXPORT
ErrorOr<Driver::DeviceMemory *>
constructDeviceMemoryFromTensor(M::ContextRef context,
                                GraphCompiler::Driver::Device *device,
                                M::MLRT::Tensor tensor);

/// Transforms a DeviceTensor representation into an AnyAsyncValueRef executable
/// returned by an ExecutableModel.
MODULAR_VISIBILITY_EXPORT AnyAsyncValueRef
prepareInput(Runtime &runtime, DeviceMemory *inputMemory, TensorSpec inputSpec);

} // namespace M::GraphCompiler::Driver

#endif
