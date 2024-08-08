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

//===----------------------------------------------------------------------===//
// DeviceSpecificDialect
//===----------------------------------------------------------------------===//

/// Describes how create type and buffer builders for a particular device used
/// during lowering.
struct DeviceSpecificDialect {
  virtual ~DeviceSpecificDialect() = default;

  // Registers a device-specific dialect with the given registry. This function
  // is called when the MOToPrimitives pass is created.
  virtual void registerDialect(DialectRegistry &registry) = 0;

  /// Given an #M.device_spec, registers a MGP::TypeBuilder appropriate for
  /// the device specification within loweringState. This function is called
  /// during lowering before any MO types are lowered.
  virtual LogicalResult
  makeTypeBuilder(DeviceRefAttr device,
                  MGP::MOGraphLoweringState &loweringState) = 0;

  /// Given an #M.device_spec, registers a MGP::BufferBuilder appropriate for
  /// the device specification within loweringState. As a side effect, the
  /// function may also emit code into the 'init' block for the given
  /// loweringState, for example to establish !mgp.context entries appropriate
  /// for the device. This function is called once the lowered 'init' block
  /// has been established but before any MO ops are lowered.
  virtual CompileDiagnosticOrSuccess
  makeBufferBuilder(DeviceRefAttr hostDevice, DeviceRefAttr targetDevice,
                    MGP::MOGraphLoweringState &loweringState) = 0;

  virtual void vtableAnchor();
};

//===----------------------------------------------------------------------===//
// MGPDeviceSpecificDialect
//===----------------------------------------------------------------------===//

/// Generic implementation of DeviceSpecificDialect which lowers to the
/// device-generic MGP types and ops.
struct MGPDeviceSpecificDialect : public DeviceSpecificDialect {
  ~MGPDeviceSpecificDialect() override = default;

  void registerDialect(DialectRegistry &registry) override;

  LogicalResult
  makeTypeBuilder(DeviceRefAttr device,
                  MGP::MOGraphLoweringState &loweringState) override;

  CompileDiagnosticOrSuccess
  makeBufferBuilder(DeviceRefAttr hostDevice, DeviceRefAttr targetDevice,
                    MGP::MOGraphLoweringState &loweringState) override;
};

//===----------------------------------------------------------------------===//
// DeviceSpecificDialectRegistry
//===----------------------------------------------------------------------===//

/// Holds all the registered DeviceSpecificDialects.
class DeviceBufferizationRegistry {
public:
  DeviceBufferizationRegistry();

  ~DeviceBufferizationRegistry() = default;

  /// Adds DeviceSpecificDialect to registry for given label. Each label can
  /// be registered at most once.
  void add(StringRef label,
           std::unique_ptr<DeviceSpecificDialect> deviceSpecificDialect);

  /// Calls all registerDialect functions.
  void callAllRegisterDialects(DialectRegistry &registry);

  /// Calls the makeTypeBuilder function appropriate for given args.
  LogicalResult
  callMakeTypeBuilder(DeviceRefAttr device,
                      MGP::MOGraphLoweringState &loweringState) const;

  /// Calls the makeBufferBuilder function appropriate for given args.
  CompileDiagnosticOrSuccess
  callMakeBufferBuilder(DeviceRefAttr hostDevice, DeviceRefAttr targetDevice,
                        MGP::MOGraphLoweringState &loweringState) const;

private:
  /// Protects map.
  mutable std::mutex mu;
  llvm::StringMap<std::unique_ptr<DeviceSpecificDialect>> map;
};

/// Registers the device-specific dialect helpers to be called by the
/// MOToPrimitives lowering pass for devices with the given label.
void registerDeviceSpecificDialect(
    StringRef label,
    std::unique_ptr<DeviceSpecificDialect> deviceSpecificDialect);

/// Adds all device-specific dialects to registry.
void registerAllDeviceDialects(DialectRegistry &registry);

/// Call the registered make type builder function appropriate for given args.
LogicalResult makeTypeBuilder(DeviceRefAttr device,
                              MGP::MOGraphLoweringState &loweringState);

/// Call the registered make buffer builder function appropriate for given args.
CompileDiagnosticOrSuccess
makeBufferBuilder(DeviceRefAttr hostDevice, DeviceRefAttr targetDevice,
                  MGP::MOGraphLoweringState &loweringState);

} // namespace M::MO

#endif // GENERICML_GRAPHCOMPILER_CONVERSION_BUFFERBUILDERREGISTRY_H
