//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/MODialect/MODevices.h"

#include "GenericML/Support/HostDeviceSpecs.h"

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
  GraphOp graphOp = op->getParentOfType<GraphOp>();
  assert(graphOp && "missing containing GraphOp");
  return getHostDevice(graphOp);
}

DeviceRefAttr MO::getDeviceOrDefaultAny(Operation *op) {
  if (auto hasDeviceOp = dyn_cast<MOHasDeviceInterface>(op))
    return getDeviceOrDefault(hasDeviceOp);

  GraphOp graphOp = op->getParentOfType<GraphOp>();
  assert(graphOp && "missing containing GraphOp");
  return getHostDevice(graphOp);
}

bool MO::isViaHostAny(Operation *op) {
  if (auto hasDeviceOp = dyn_cast<MOHasDeviceInterface>(op))
    return hasDeviceOp.isViaHost();

  return false;
}

DeviceRefAttr MO::getDeviceOrDefault(GraphOp graphOp, TensorType type) {
  // Check the type annotation.
  DeviceRefAttr device = type.getDeviceRef();
  if (device)
    return device;

  // Fallback to the 'host' device.
  return getHostDevice(graphOp);
}

DeviceRefAttr MO::getHostDevice(GraphOp graphOp) {
  // Check the device specs annotation.
  DeviceSpecCollectionAttr deviceSpecs = graphOp.getDeviceSpecs();
  if (deviceSpecs)
    return deviceSpecs.getHost();

  // Fall back to 'cpu'.
  return DeviceRefAttr::get(graphOp.getContext(), kCPULabel, 0);
}

bool MO::isOnHost(Operation *op) {
  return getDeviceOrDefaultAny(op) ==
         getHostDevice(op->getParentOfType<GraphOp>());
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
  if (auto hasDevice = dyn_cast<MOHasDeviceInterface>(op)) {
    metadata.device = hasDevice.getDeviceRef();
    metadata.isViaCpu = hasDevice.isViaHost();
    assert((metadata.device || !metadata.isViaCpu) &&
           "operation should only have an is_via_host attribute if it also has "
           "a device attribute");
  }
  return metadata;
}

bool DeviceAnnots::add(const DeviceAnnots &that) {
  if (!isConsistentWith(that))
    return false;
  if (!device) {
    device = that.device;
    isViaCpu = that.isViaCpu;
  }
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
    if (isViaCpu)
      hasDevice.setIsViaHost();
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
  if (!isConsistentWith(that.device))
    return false;
  if (device && that.device && isViaCpu != that.isViaCpu)
    return false;
  return true;
}
