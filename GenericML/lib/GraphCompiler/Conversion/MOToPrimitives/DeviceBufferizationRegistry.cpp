//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/Conversion/DeviceBufferizationRegistry.h"
#include "Support/AssertStream.h"

using namespace M;
using namespace M::MO;

//===----------------------------------------------------------------------===//
// DeviceSpecificDialect
//===----------------------------------------------------------------------===//

void DeviceSpecificDialect::vtableAnchor() {}

//===----------------------------------------------------------------------===//
// MGPDeviceSpecificDialect
//===----------------------------------------------------------------------===//

void MGPDeviceSpecificDialect::registerDialect(DialectRegistry &registry) {}

LogicalResult MGPDeviceSpecificDialect::makeTypeBuilder(
    DeviceSpecAttr deviceSpec, MGP::MOGraphLoweringState &loweringState) {
  loweringState.registerTypeBuilder(
      deviceSpec.getRef(),
      std::make_unique<MGP::MGPTypeBuilder>(deviceSpec.getRef()));
  return success();
}

CompileDiagnosticOrSuccess MGPDeviceSpecificDialect::makeBufferBuilder(
    DeviceRefAttr hostDevice, DeviceSpecAttr deviceSpec,
    MGP::MOGraphLoweringState &loweringState) {
  MGP::MGPBufferBuilder *hostBufferBuilder;
  auto bufferBuilder =
      std::make_unique<MGP::MGPBufferBuilder>(deviceSpec.getRef());
  if (hostDevice == deviceSpec.getRef()) {
    hostBufferBuilder = bufferBuilder.get();
  } else {
    hostBufferBuilder = static_cast<MGP::MGPBufferBuilder *>(
        loweringState.getBufferBuilder(hostDevice));
    GENERICML_ASSERT(hostBufferBuilder)
        << "no buffer builder has been created for the host device '"
        << hostDevice << "'";
  }
  CompileDiagnosticOr<TypedValue<MEF::ChainType>> outChainOr =
      bufferBuilder->setupForModel(
          loweringState.getModelOp(), hostBufferBuilder,
          loweringState.getInitOpBuilder(), loweringState.getGraph()->getLoc(),
          cast<TypedValue<MEF::ChainType>>(loweringState.getInitChain()));
  if (outChainOr.isError())
    return outChainOr.takeError();
  loweringState.registerBufferBuilder(deviceSpec.getRef(),
                                      std::move(bufferBuilder));
  loweringState.setInitChain(outChainOr.takeValue());
  return success();
}

//===----------------------------------------------------------------------===//
// DeviceSpecificDialectRegistry
//===----------------------------------------------------------------------===//

DeviceBufferizationRegistry::DeviceBufferizationRegistry() {
  add(kCPULabel, std::make_unique<MGPDeviceSpecificDialect>());
  add(kCUDALabel, std::make_unique<MGPDeviceSpecificDialect>());
}

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

CompileDiagnosticOrSuccess DeviceBufferizationRegistry::callMakeBufferBuilder(
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

CompileDiagnosticOrSuccess
MO::makeBufferBuilder(DeviceRefAttr hostDevice, DeviceSpecAttr deviceSpec,
                      MGP::MOGraphLoweringState &loweringState) {
  return getRegistry().callMakeBufferBuilder(hostDevice, deviceSpec,
                                             loweringState);
}
