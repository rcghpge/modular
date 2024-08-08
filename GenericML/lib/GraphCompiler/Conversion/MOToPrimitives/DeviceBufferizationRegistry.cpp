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
    DeviceRefAttr device, MGP::MOGraphLoweringState &loweringState) {
  loweringState.registerTypeBuilder(
      device, std::make_unique<MGP::MGPTypeBuilder>(device));
  return success();
}

CompileDiagnosticOrSuccess MGPDeviceSpecificDialect::makeBufferBuilder(
    DeviceRefAttr hostDevice, DeviceRefAttr targetDevice,
    MGP::MOGraphLoweringState &loweringState) {
  MGP::MGPBufferBuilder *hostBufferBuilder;
  auto bufferBuilder = std::make_unique<MGP::MGPBufferBuilder>(targetDevice);
  if (hostDevice == targetDevice) {
    hostBufferBuilder = bufferBuilder.get();
  } else {
    hostBufferBuilder = static_cast<MGP::MGPBufferBuilder *>(
        loweringState.getBufferBuilder(hostDevice));
    ASSERT_STREAM(hostBufferBuilder,
                  << "no buffer builder has been created for the host device '"
                  << hostDevice << "'");
  }
  auto inChain = cast<TypedValue<MEF::ChainType>>(loweringState.getInitChain());
  // DANGER: init chain is the result of device context creation which is a
  // result of runtime creation. If this is a gpu device, it does not actually
  // depend on the device context.
  auto deviceContextCreate =
      loweringState.getInitChain().getDefiningOp<MGP::DeviceContextCreateOp>();
  if (deviceContextCreate)
    inChain = deviceContextCreate.getOperand(0)
                  .getDefiningOp<MGP::RuntimeCreateOp>()
                  .getOutChain();
  auto chainAndDevice = bufferBuilder->setupForModel(
      loweringState.getModelOp(), hostBufferBuilder,
      loweringState.getInitOpBuilder(), loweringState.getGraph()->getLoc(),
      inChain, loweringState.getInitDevice(bufferBuilder->device));
  loweringState.setInitChain(chainAndDevice.first);
  loweringState.setInitDevice(bufferBuilder->device, chainAndDevice.second);
  loweringState.registerBufferBuilder(targetDevice, std::move(bufferBuilder));
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
  [[maybe_unused]] bool inserted =
      map.insert({label, std::move(deviceSpecificDialect)}).second;
  ASSERT_STREAM(inserted, << "builder for label " << label
                          << " has already been registered");
}

void DeviceBufferizationRegistry::callAllRegisterDialects(
    DialectRegistry &registry) {
  std::lock_guard<std::mutex> guard(mu);
  for (const auto &pair : map)
    pair.second->registerDialect(registry);
}

LogicalResult DeviceBufferizationRegistry::callMakeTypeBuilder(
    DeviceRefAttr device, MGP::MOGraphLoweringState &loweringState) const {
  std::lock_guard<std::mutex> guard(mu);
  auto itr = map.find(device.getLabel());
  if (itr == map.end()) {
    return loweringState.getGraph().emitOpError("no type builder for label '")
           << device.getLabel() << "' has been registered.";
  }
  return itr->second->makeTypeBuilder(device, loweringState);
}

CompileDiagnosticOrSuccess DeviceBufferizationRegistry::callMakeBufferBuilder(
    DeviceRefAttr hostDevice, DeviceRefAttr targetDevice,
    MGP::MOGraphLoweringState &loweringState) const {
  std::lock_guard<std::mutex> guard(mu);
  auto itr = map.find(targetDevice.getLabel());
  if (itr == map.end()) {
    return loweringState.getGraph().emitOpError("no buffer builder for label '")
           << targetDevice.getLabel() << "' has been registered.";
  }
  return itr->second->makeBufferBuilder(hostDevice, targetDevice,
                                        loweringState);
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

LogicalResult MO::makeTypeBuilder(DeviceRefAttr device,
                                  MGP::MOGraphLoweringState &loweringState) {
  return getRegistry().callMakeTypeBuilder(device, loweringState);
}

CompileDiagnosticOrSuccess
MO::makeBufferBuilder(DeviceRefAttr hostDevice, DeviceRefAttr targetDevice,
                      MGP::MOGraphLoweringState &loweringState) {
  return getRegistry().callMakeBufferBuilder(hostDevice, targetDevice,
                                             loweringState);
}
