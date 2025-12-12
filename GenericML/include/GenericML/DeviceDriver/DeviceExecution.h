//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_DEVICEDRIVER_DEVICEEXECUTION
#define GENERICML_DEVICEDRIVER_DEVICEEXECUTION

#include "GenericML/DeviceDriver/Driver.h"
#include "Runtime/Tensor/Tensor.h"
#include "Support/Context.h"
#include "Support/ML/TensorSpec.h"
#include "Support/SymbolExport.h"

namespace M::Driver {

// Construct a DeviceMemory block to hold the provided tensor.
MODULAR_VISIBILITY_EXPORT
ErrorOr<Driver::DeviceMemory *>
constructDeviceMemoryFromTensor(M::ContextRef context, Driver::Device *device,
                                M::Tensor tensor);

/// Transforms a DeviceTensor representation into an AnyAsyncValueRef executable
/// returned by an ExecutableModel.
MODULAR_VISIBILITY_EXPORT AnyAsyncValueRef
prepareInput(Runtime &runtime, DeviceMemory *inputMemory, TensorSpec inputSpec);

} // namespace M::Driver

#endif
