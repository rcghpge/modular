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
struct DriverMojoValue {
  void *data;
  M::MojoValue::DestructorFn destructor;
};

// Fill output. This is defined and passed as a callback by mojo.
void addToOutputList(void *mojoExecutableGraph, void *outputList,
                     DeviceMemory *memory, TensorSpec *spec);

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
    decltype(addToOutputList) addToOutputListFn, const ContextRef &maxContext,
    const RCRef<Device> &device, void *mojoContext, void *outputList);
} // namespace M::Driver

#endif
