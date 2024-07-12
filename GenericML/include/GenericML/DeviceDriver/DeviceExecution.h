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
#include "Support/Context.h"
#include "Support/ML/TensorSpec.h"
#include "Support/SymbolExport.h"

namespace M::Driver {
// This is the bitwise equivalent C++ representation of _CMojoValue in
// mojo side. We destructure M::MojoValue into this struct and then pass to
// mojo and mojo can interpret it as _CMojoValue.
extern "C" struct DriverMojoValue {
  void *data;
  M::MojoValue::DestructorFn destructor;
};

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

// Writes results array back to mojo list.
MODULAR_VISIBILITY_EXPORT size_t processResults(
    llvm::MutableArrayRef<AnyAsyncValueRef> results, ExecutableModel *model,
    decltype(addTensorToOutputList) addTensorToOutputListFn,
    decltype(addValueToOutputList) addValueToOutputListFn,
    const ContextRef &maxContext, const RCRef<Device> &device,
    void *mojoContext, void *outputList);
} // namespace M::Driver

#endif
