//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/MODialect/MODevices.h"

#include "GenericML/Support/HostDeviceSpecs.h"
#include "Support/AssertStream.h"

using namespace M;
using namespace MO;

//===----------------------------------------------------------------------===//
// Utils
//===----------------------------------------------------------------------===//

DeviceRefAttr MO::getDeviceOrDefault(MOHasDeviceInterface op) {
  // Check the op annotation.
  DeviceRefAttr device = op.getDeviceRef();
  if (device)
    return device;

  // Fallback to the 'host' device.
  return getHostDeviceAny(op);
}

DeviceRefAttr MO::getDeviceOrDefaultAny(Operation *op) {
  if (auto hasDeviceOp = dyn_cast<MOHasDeviceInterface>(op))
    return getDeviceOrDefault(hasDeviceOp);

  // Fallback to the 'host' device.
  return getHostDeviceAny(op);
}

DeviceRefAttr MO::getDeviceOrDefault(GraphOp graphOp, TensorType type) {
  // Check the type annotation.
  DeviceRefAttr device = type.getDeviceRef();
  if (device)
    return device;

  // Fallback to the 'host' device.
  return getHostDevice(graphOp);
}

DeviceRefAttr MO::getDeviceOrDefaultAny(GraphOp graphOp, Type type) {
  if (auto tensorType = dyn_cast<TensorType>(type))
    return getDeviceOrDefault(graphOp, tensorType);

  // Fallback to the 'host' device.
  return getHostDevice(graphOp);
}

DeviceRefAttr MO::getHostDevice(GraphOp graphOp) {
  // Check the device specs annotation.
  DeviceSpecCollectionAttr deviceSpecs = graphOp.getDeviceSpecs();
  if (deviceSpecs)
    return deviceSpecs.getHost();

  // Fall back to 'cpu'.
  return DeviceRefAttr::getDefault(graphOp.getContext());
}

DeviceRefAttr MO::getHostDeviceAny(Operation *op) {
  auto graphOp = op->getParentOfType<GraphOp>();
  ASSERT_STREAM(graphOp, << "operation is not contained within a MO::GraphOp");
  return getHostDevice(graphOp);
}

bool MO::isOnHost(Operation *op) {
  return getDeviceOrDefaultAny(op) == getHostDeviceAny(op);
}

bool MO::isOnHost(GraphOp graphOp, Type type) {
  return getDeviceOrDefaultAny(graphOp, type) == getHostDevice(graphOp);
}

ErrorOr<DeviceSpecCollectionAttr> MO::getDeviceSpecsOrDefault(GraphOp graphOp) {
  // Check the device specs annotation.
  DeviceSpecCollectionAttr deviceSpecs = graphOp.getDeviceSpecs();
  if (deviceSpecs)
    return deviceSpecs;

  // Fallback to a description of the host CPU.
  // CAUTION: Unlike AddDevicesPass, no additional devices will be included.
  // This is to support unit tests which don't want to deal with devices.
  ErrorOr<std::unique_ptr<DeviceSpecCollection>> runtimeDeviceSpecsOr =
      GML::getHostCPUDeviceSpecs();
  if (runtimeDeviceSpecsOr)
    return runtimeDeviceSpecsOr.takeError();
  return fromRuntimeDeviceSpecs(graphOp.getContext(), **runtimeDeviceSpecsOr);
}

//===----------------------------------------------------------------------===//
// DeviceAnnots
//===----------------------------------------------------------------------===//

DeviceAnnots DeviceAnnots::forOp(Operation *op) {
  DeviceAnnots metadata;
  if (auto hasDevice = dyn_cast<MOHasDeviceInterface>(op))
    metadata.device = hasDevice.getDeviceRef();
  return metadata;
}

bool DeviceAnnots::add(const DeviceAnnots &that) {
  if (!isConsistentWith(that))
    return false;
  if (!device)
    device = that.device;
  return true;
}

bool DeviceAnnots::addOp(Operation *op) {
  if (!op)
    return true;
  return add(DeviceAnnots::forOp(op));
}

Operation *DeviceAnnots::stampAny(Operation *op) const {
  if (auto hasDevice = dyn_cast<MOHasDeviceInterface>(op)) {
    if (device && !hasDevice.getDeviceRef())
      hasDevice.setDeviceRef(device);
  }
  // If the op does not support device annotations then we'll just ignore
  // the annotation and trust the AssingDevicesPass will check everything is
  // consistent.
  return op;
}

bool DeviceAnnots::isConsistentWith(DeviceRefAttr thatDevice) const {
  return !device || !thatDevice || device == thatDevice;
}

bool DeviceAnnots::isConsistentWith(const DeviceAnnots &that) const {
  return isConsistentWith(that.device);
}
