//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/Conversion/DeviceBufferizationRegistry.h"
#include "GenericML/Support/Assert.h"

using namespace M;
using namespace M::MO;

//===----------------------------------------------------------------------===//
// DeviceSpecificDialect
//===----------------------------------------------------------------------===//

void DeviceSpecificDialect::vtableAnchor() {}

//===----------------------------------------------------------------------===//
// DeviceSpecificDialectRegistry
//===----------------------------------------------------------------------===//

void DeviceBufferizationRegistry::add(
    StringRef label,
    std::unique_ptr<DeviceSpecificDialect> deviceSpecificDialect) {
  std::lock_guard<std::mutex> guard(mu);
  bool inserted = map.insert({label, std::move(deviceSpecificDialect)}).second;
  GENERICML_ASSERT(inserted)
      << "builder for label " << label << " has already been registered";
}

void DeviceBufferizationRegistry::callAllRegisterDialects(
    DialectRegistry &registry) {
  std::lock_guard<std::mutex> guard(mu);
  for (const auto &pair : map)
    pair.second->registerDialect(registry);
}

LogicalResult DeviceBufferizationRegistry::callMakeTypeBuilder(
    DeviceSpecAttr deviceSpec, MGP::MOGraphLoweringState &loweringState) const {
  std::lock_guard<std::mutex> guard(mu);
  auto itr = map.find(deviceSpec.getRef().getLabel());
  if (itr == map.end()) {
    return loweringState.getGraph().emitOpError("no type builder for label '")
           << deviceSpec.getRef().getLabel() << "' has been registered.";
  }
  return itr->second->makeTypeBuilder(deviceSpec, loweringState);
}

LogicalResult DeviceBufferizationRegistry::callMakeBufferBuilder(
    DeviceRefAttr hostDevice, DeviceSpecAttr deviceSpec,
    MGP::MOGraphLoweringState &loweringState) const {
  std::lock_guard<std::mutex> guard(mu);
  auto itr = map.find(deviceSpec.getRef().getLabel());
  if (itr == map.end()) {
    return loweringState.getGraph().emitOpError("no buffer builder for label '")
           << deviceSpec.getRef().getLabel() << "' has been registered.";
  }
  return itr->second->makeBufferBuilder(hostDevice, deviceSpec, loweringState);
}

//  TODO: May need to move into dedicated 'globals' shared library.
static DeviceBufferizationRegistry &getAddDevicesRegistrySingleton(
    const std::function<DeviceBufferizationRegistry *()> &ctor) {
  static DeviceBufferizationRegistry *reg = ctor();
  return *reg;
}

static DeviceBufferizationRegistry &getRegistry() {
  return getAddDevicesRegistrySingleton(
      [] { return new DeviceBufferizationRegistry(); });
}

void MO::registerDeviceSpecificDialect(
    StringRef label,
    std::unique_ptr<DeviceSpecificDialect> deviceSpecificDialect) {
  return getRegistry().add(label, std::move(deviceSpecificDialect));
}

void MO::registerAllDeviceDialects(DialectRegistry &registry) {
  getRegistry().callAllRegisterDialects(registry);
}

LogicalResult MO::makeTypeBuilder(DeviceSpecAttr deviceSpec,
                                  MGP::MOGraphLoweringState &loweringState) {
  return getRegistry().callMakeTypeBuilder(deviceSpec, loweringState);
}

LogicalResult MO::makeBufferBuilder(DeviceRefAttr hostDevice,
                                    DeviceSpecAttr deviceSpec,
                                    MGP::MOGraphLoweringState &loweringState) {
  return getRegistry().callMakeBufferBuilder(hostDevice, deviceSpec,
                                             loweringState);
}
