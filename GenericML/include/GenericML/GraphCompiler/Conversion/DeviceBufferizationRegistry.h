//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Provides a registry so device-specific dialects can plug into the
// MOToPrimitives lowering pass.
//
// Cf GenericML/Support/HostDeviceSpecs.h, which provides
// a similar mechanism for collecting device specs at runtime.
//
// Note there are three phases at play:
//  1. Given a device reference label, how should builders be constructed
//     and registered.
//  2. Given a device specification, what builder should be used when lowering.
//  3. Given a builder, what device specific types and ops should be used
//     in the lowered model.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_CONVERSION_BUFFERBUILDERREGISTRY_H
#define GENERICML_GRAPHCOMPILER_CONVERSION_BUFFERBUILDERREGISTRY_H

#include "GenericML/GraphCompiler/Interfaces/LoweringState.h"
#include "Support/MDialect/MAttrs.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/StringMap.h"

namespace M::MO {

/// Type of functions which, given an #M.device_spec, registers a
/// MGP::BufferBuilder appropriate for the device specification within
/// loweringState. As a side effect, the function may also emit code into
/// the 'init' block for the given loweringState, for example to establish
/// !mgp.context entries appropriate for the device. These functions are
/// called once the lowered 'init' block has been established but before any
/// MO ops are lowered.
using MakeBufferBuilderFn = llvm::unique_function<LogicalResult(
    DeviceSpecAttr deviceSpec, MGP::MOGraphLoweringState &loweringState) const>;

/// Holds all the registered bufferization helpers.
class DeviceBufferizationRegistry {
public:
  DeviceBufferizationRegistry() = default;
  ~DeviceBufferizationRegistry() = default;

  /// Adds makeBufferBuilderFn into the registry for devices with the given
  /// label. At most one builder can be registered per label.
  void add(StringRef label, MakeBufferBuilderFn &&makeBufferBuilderFn);

  /// Calls the registered buffer builder appropriate for given args.
  /// A builder for the device spec's label must have been registered.
  LogicalResult call(DeviceSpecAttr deviceSpec,
                     MGP::MOGraphLoweringState &loweringState) const;

private:
  /// Protects map.
  mutable std::mutex mu;
  llvm::StringMap<MakeBufferBuilderFn> map;
};

/// Registers makeBufferBuilderFn to be called by the MOToPrimitives lowering
/// pass for devices with the given label.
void registerMakeBufferBuilder(StringRef label,
                               MakeBufferBuilderFn &&makeBufferBuilderFn);

/// Call the registered buffer builder appropriate for given args.
LogicalResult makeBufferBuilder(DeviceSpecAttr deviceSpec,
                                MGP::MOGraphLoweringState &loweringState);

} // namespace M::MO

#endif // GENERICML_GRAPHCOMPILER_CONVERSION_BUFFERBUILDERREGISTRY_H
