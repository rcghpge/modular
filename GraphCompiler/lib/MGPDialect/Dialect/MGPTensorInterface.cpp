//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MGPDialect/Dialect/MGPTensorInterface.h"

#include "GraphCompiler/MGPDialect/Dialect/MGPOps.h"
#include "GraphCompiler/MGPDialect/Dialect/MGPTypes.h"
#include "Support/AssertStream.h"
#include "mlir/IR/Builders.h"

namespace M::MGP {

//===----------------------------------------------------------------------===//
// TensorBufferDialectInterface
//===----------------------------------------------------------------------===//

DenseSet<TensorBufferDialectInterface *>
TensorBufferDialectInterface::getAllInterfaces(ModelOp modelOp) {
  DenseSet<TensorBufferDialectInterface *> interfaces;
  modelOp.walk([&interfaces](Operation *op) {
    if (auto interface =
            dyn_cast_if_present<TensorBufferDialectInterface>(op->getDialect()))
      interfaces.insert(interface);
  });
  return interfaces;
}

BufferBuilderContext
TensorBufferDialectInterface::getAllBufferBuilders(ModelOp modelOp) {
  DenseSet<TensorBufferDialectInterface *> interfaces =
      getAllInterfaces(modelOp);
  BufferBuilderContext builderContext;
  for (TensorBufferDialectInterface *interface : interfaces)
    builderContext.hostDevice = interface->addBuilders(modelOp, builderContext);
  return builderContext;
}

MGPBufferBuilder *TensorBufferDialectInterface::getBuilderForOpExecution(
    const BufferBuilderContext &builderContext, Operation *op) {
  if (auto interface =
          dyn_cast_if_present<TensorBufferDialectInterface>(op->getDialect())) {
    DeviceRefAttr device = interface->getExecutionDevice(builderContext, op);
    return &builderContext.getBufferBuilder(device);
  }

  return &builderContext.getHostBufferBuilder();
}

MGPBufferBuilder *TensorBufferDialectInterface::getBuilderForTypeStorage(
    const BufferBuilderContext &builderContext, Type type) {
  if (auto interface =
          dyn_cast<TensorBufferDialectInterface>(&type.getDialect())) {
    DeviceRefAttr device = interface->getStorageDevice(builderContext, type);
    return &builderContext.getBufferBuilder(device);
  }

  return &builderContext.getHostBufferBuilder();
}

void TensorBufferDialectInterface::vtableAnchor() {}

//===----------------------------------------------------------------------===//
// MGPTensorBufferDialectInterface
//===----------------------------------------------------------------------===//

DeviceRefAttr MGPTensorBufferDialectInterface::addBuilders(
    ModelOp modelOp, BufferBuilderContext &builderContext) {
  DeviceRefAttr hostDevice = DeviceRefAttr::getDefault(modelOp->getContext());

  // Recover the 'host' buffer builder.
  auto hostBufferBuilder = std::make_unique<MGPBufferBuilder>(hostDevice);
  MGPBufferBuilder *hostBufferBuilderPtr = hostBufferBuilder.get();
  hostBufferBuilder->recoverFromModel(modelOp, hostBufferBuilderPtr);
  builderContext.registerTypeBuilder(hostDevice);
  builderContext.registerBufferBuilder(hostDevice,
                                       std::move(hostBufferBuilder));

  // Add the buffer builders for other supported devices.
  SmallVector<DeviceRefAttr> otherDevices;
  for (auto device : modelOp.getDevices()) {
    if (device != hostDevice)
      otherDevices.push_back(dyn_cast<DeviceRefAttr>(device));
  }

  // Recover all the other buffer builders.
  for (auto device : otherDevices) {
    auto bufferBuilder = std::make_unique<MGPBufferBuilder>(device);
    bufferBuilder->recoverFromModel(modelOp, hostBufferBuilderPtr);
    builderContext.registerTypeBuilder(device);
    builderContext.registerBufferBuilder(device, std::move(bufferBuilder));
  }

  return hostDevice;
}

DeviceRefAttr MGPTensorBufferDialectInterface::getExecutionDevice(
    const BufferBuilderContext &builderContext, Operation *op) {
  if (auto hasDeviceOp = dyn_cast<MGPHasDeviceOpInterface>(op))
    return hasDeviceOp.getDevice();
  return builderContext.hostDevice;
}

DeviceRefAttr MGPTensorBufferDialectInterface::getStorageDevice(
    const BufferBuilderContext &builderContext, Type type) {
  if (auto hasDeviceType = dyn_cast<MGPHasDeviceTypeInterface>(type))
    return hasDeviceType.getDevice();
  return builderContext.hostDevice;
}

} // namespace M::MGP
