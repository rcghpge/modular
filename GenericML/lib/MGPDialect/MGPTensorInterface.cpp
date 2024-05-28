//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/MGPDialect/MGPTensorInterface.h"

#include "GenericML/MGPDialect/MGPOps.h"
#include "GenericML/MGPDialect/MGPTypes.h"
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

TypeBuilder *TensorBufferDialectInterface::getTypeBuilder(
    const BufferBuilderContext &builderContext, Type type) {
  if (auto interface =
          dyn_cast<TensorBufferDialectInterface>(&type.getDialect())) {
    DeviceRefAttr device = interface->getStorageDevice(builderContext, type);
    return builderContext.getTypeBuilder(device);
  }

  return builderContext.getHostTypeBuilder();
}

BufferBuilder *TensorBufferDialectInterface::getBuilderForOpExecution(
    const BufferBuilderContext &builderContext, Operation *op) {
  if (auto interface =
          dyn_cast_if_present<TensorBufferDialectInterface>(op->getDialect())) {
    DeviceRefAttr device = interface->getExecutionDevice(builderContext, op);
    return builderContext.getBufferBuilder(device);
  }

  return builderContext.getHostBufferBuilder();
}

BufferBuilder *TensorBufferDialectInterface::getBuilderForTypeStorage(
    const BufferBuilderContext &builderContext, Type type) {
  if (auto interface =
          dyn_cast<TensorBufferDialectInterface>(&type.getDialect())) {
    DeviceRefAttr device = interface->getStorageDevice(builderContext, type);
    return builderContext.getBufferBuilder(device);
  }

  return builderContext.getHostBufferBuilder();
}

void TensorBufferDialectInterface::vtableAnchor() {}

//===----------------------------------------------------------------------===//
// MGPTensorBufferDialectInterface
//===----------------------------------------------------------------------===//

DeviceRefAttr MGPTensorBufferDialectInterface::addBuilders(
    ModelOp modelOp, BufferBuilderContext &builderContext) {
  SmallVector<DeviceRefAttr> allDevices;
  DeviceRefAttr hostDevice;
  // Recover all the devices used by the model, and the distinguished 'host'
  // device, from the unique mgp.check_runtime op.
  modelOp.walk([&](CheckRuntimeOp checkRuntimeOp) {
    hostDevice = checkRuntimeOp.getBDevices().getHost();
    for (auto spec : checkRuntimeOp.getBDevices().getDevices())
      allDevices.emplace_back(spec.getRef());
    return WalkResult::interrupt();
  });
  if (!hostDevice) {
    // If the model has no such op, fill in a default. (Helps for unit tests).
    hostDevice = DeviceRefAttr::getDefault(modelOp->getContext());
    allDevices.emplace_back(hostDevice);
  }

  // Recover the 'host' buffer builder.
  auto hostBufferBuilder = std::make_unique<MGPBufferBuilder>(hostDevice);
  MGPBufferBuilder *hostBufferBuilderPtr = hostBufferBuilder.get();
  hostBufferBuilder->recoverFromModel(modelOp, hostBufferBuilderPtr);
  builderContext.registerTypeBuilder(
      hostDevice, std::make_unique<MGPTypeBuilder>(hostDevice));
  builderContext.registerBufferBuilder(hostDevice,
                                       std::move(hostBufferBuilder));

  // Recover all the other buffer builders.
  for (auto device : allDevices) {
    if (device == hostDevice)
      continue;
    auto bufferBuilder = std::make_unique<MGPBufferBuilder>(device);
    bufferBuilder->recoverFromModel(modelOp, hostBufferBuilderPtr);
    builderContext.registerTypeBuilder(
        device, std::make_unique<MGPTypeBuilder>(device));
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
