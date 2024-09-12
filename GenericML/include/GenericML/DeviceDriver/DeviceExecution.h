//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_DEVICEDRIVER_DEVICEEXECUTION
#define GENERICML_DEVICEDRIVER_DEVICEEXECUTION

#include "GenericML/DeviceDriver/Driver.h"
#include "GenericML/Support/Model.h"
#include "Runtime/MojoValue.h"
#include "Support/ADT/SmartVariant.h"
#include "Support/Context.h"
#include "Support/ML/TensorSpec.h"
#include "Support/SymbolExport.h"

namespace M::Driver {

/// Container holding the result of device execution.
/// Each element is either a tensor or a Mojo value.
using ExecResultType =
    SmallVector<SmartVariant<Driver::DeviceMemory *, M::MojoValue *>>;

// This is the bitwise equivalent C++ representation of _CMojoValue in
// mojo side. We destructure M::MojoValue into this struct and then pass to
// mojo and mojo can interpret it as _CMojoValue.
extern "C" struct DriverMojoValue {
  void *data;
  void (*destructor)(void *);
};

// Construct a DeviceMemory block to hold the provided tensor.
MODULAR_VISIBILITY_EXPORT
ErrorOr<Driver::DeviceMemory *>
constructDeviceMemoryFromTensor(M::ContextRef context, Driver::Device *device,
                                M::Tensor tensor);

// Fill output. These are defined and passed as a callback by mojo.
void addTensorToOutputList(void *mojoExecutableGraph, void *outputList,
                           DeviceMemory *memory, TensorSpec *spec);

// Add a value of some mojo type to output list. We don't know
// what it is. We transparently pass it here. This is defined and
// passed as a callback by mojo.
void addValueToOutputList(void *mojoExecutableGraph, void *outputList,
                          DriverMojoValue value);

// Transforms a DeviceTensor representation into a SmallVector executable
// by an ExecutableModel.
MODULAR_VISIBILITY_EXPORT SmallVector<AnyAsyncValueRef>
prepareInputs(Runtime &runtime, DeviceMemory **inputMemories,
              TensorSpec *inputSpecs, size_t inputCount,
              DriverMojoValue *mojoValues = nullptr,
              size_t mojoValuesCount = 0);

using GraphExecutionResultsTy =
    SmallVector<std::variant<M::Tensor, Driver::DriverMojoValue>>;

// Process results array and convert to max driver types.
MODULAR_VISIBILITY_EXPORT
GraphExecutionResultsTy
processResults(llvm::MutableArrayRef<AnyAsyncValueRef> results,
               ExecutableModel *model, ContextRef maxContext,
               RCRef<Device> device);

// Processes results and convert them into tensors.
MODULAR_VISIBILITY_EXPORT SmallVector<M::Tensor>
processTensorResults(llvm::MutableArrayRef<AnyAsyncValueRef> results,
                     ExecutableModel *model, ContextRef maxContext,
                     RCRef<Device> device);

// Writes results to mojo callback function
MODULAR_VISIBILITY_EXPORT
ErrorOrSuccess passResultsToMojo(
    MutableArrayRef<std::variant<M::Tensor, DriverMojoValue>> results,
    decltype(addTensorToOutputList) addTensorToOutputListFn,
    decltype(addValueToOutputList) addValueToOutputListFn, void *mojoContext,
    void *outputList, ContextRef maxContext, RCRef<Driver::Device> device);

} // namespace M::Driver

#endif
