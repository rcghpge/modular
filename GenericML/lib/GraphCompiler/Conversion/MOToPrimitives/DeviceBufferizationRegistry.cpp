//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/Conversion/DeviceBufferizationRegistry.h"

using namespace M;
using namespace M::MO;

void DeviceBufferizationRegistry::add(
    StringRef label, MakeBufferBuilderFn &&makeBufferBuilderFn) {
  std::lock_guard<std::mutex> guard(mu);
  bool inserted = map.insert({label, std::move(makeBufferBuilderFn)}).second;
  assert(inserted && "builder for label has already been registered");
}

LogicalResult DeviceBufferizationRegistry::call(
    DeviceSpecAttr deviceSpec, MGP::MOGraphLoweringState &loweringState) const {
  std::lock_guard<std::mutex> guard(mu);
  auto itr = map.find(deviceSpec.getRef().getLabel());
  if (itr == map.end()) {
    return loweringState.getGraph().emitOpError("no builder for label '")
           << deviceSpec.getRef().getLabel() << "' has been registered.";
  }
  return itr->second(deviceSpec, loweringState);
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

void MO::registerMakeBufferBuilder(StringRef label,
                                   MakeBufferBuilderFn &&makeBufferBuilderFn) {
  return getRegistry().add(label, std::move(makeBufferBuilderFn));
}

LogicalResult MO::makeBufferBuilder(DeviceSpecAttr deviceSpec,
                                    MGP::MOGraphLoweringState &loweringState) {
  return getRegistry().call(deviceSpec, loweringState);
}
