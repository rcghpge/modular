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

static DeviceRefAttr operator||(DeviceRefAttr lhs, DeviceRefAttr rhs) {
  return lhs ? lhs : rhs;
}

DeviceRefAttr MO::getDeviceOrDefault(MOHasDeviceInterface op) {
  return op.getExecutionDevice() || getHostDevice(op.getContext());
}

DeviceRefAttr MO::getDeviceOrDefaultAny(Operation *op) {
  if (auto hasDeviceOp = dyn_cast<MOHasDeviceInterface>(op))
    return getDeviceOrDefault(hasDeviceOp);

  // Fallback to the 'host' device.
  return getHostDevice(op->getContext());
}

DeviceRefAttr MO::getDeviceOrDefault(MOHasDeviceTypeInterface type) {
  return type.getDeviceRef() || getHostDevice(type.getContext());
}

DeviceRefAttr MO::getDeviceOrDefaultAny(Type type) {
  if (auto hasDeviceType = dyn_cast<MOHasDeviceTypeInterface>(type))
    return getDeviceOrDefault(hasDeviceType);

  // Fallback to the 'host' device.
  return getHostDevice(type.getContext());
}

SmallVector<DeviceRefAttr> MO::getUniqueDevicesFromKernel(Operation *op) {
  SmallVector<DeviceRefAttr> devices;
  llvm::DenseSet<DeviceRefAttr> seenDevices;

  // Helper to add device if not seen before
  auto addDeviceIfNew = [&](TensorType tensorType) {
    if (auto deviceRef = tensorType.getDeviceRef())
      if (seenDevices.insert(deviceRef).second)
        devices.push_back(deviceRef);
  };

  // Check operands
  for (Value operand : op->getOperands())
    if (auto tensorType = dyn_cast<TensorType>(operand.getType()))
      addDeviceIfNew(tensorType);

  // Check results
  for (Value result : op->getResults())
    if (auto tensorType = dyn_cast<TensorType>(result.getType()))
      addDeviceIfNew(tensorType);

  return devices;
}

bool MO::isOnHost(Operation *op) {
  if (auto hasDeviceOp = dyn_cast<MOHasDeviceInterface>(op)) {
    auto devices = hasDeviceOp.getExecutionDevices();
    auto hostDevice = getHostDevice(op);
    return std::all_of(
        devices.begin(), devices.end(),
        [&](DeviceRefAttr device) { return device == hostDevice; });
  }
  return true;
}

bool MO::isOnHost(Type type) {
  return getDeviceOrDefaultAny(type) == getHostDevice(type.getContext());
}

//===----------------------------------------------------------------------===//
// DeviceAnnots
//===----------------------------------------------------------------------===//

static DeviceRefAttr getDeviceRef(MOHasDeviceInterface op) {
  return getDeviceOrDefault(op);
}

DeviceAnnots DeviceAnnots::forOp(Operation *op) {
  DeviceAnnots metadata;
  auto hasDevice = dyn_cast<MOHasDeviceInterface>(op);
  if (hasDevice && !op->hasTrait<MOMultiDevice>())
    metadata.device = getDeviceRef(hasDevice);
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
    if (device)
      hasDevice.getOperation()->setAttr(MO::kOperationDeviceRefAttrName,
                                        device);
  }
  // If the op does not support device annotations then we'll just ignore
  // the annotation and trust the AssignDevicesPass will check everything is
  // consistent.
  return op;
}

bool DeviceAnnots::isConsistentWith(DeviceRefAttr thatDevice) const {
  return !device || !thatDevice || device == thatDevice;
}

bool DeviceAnnots::isConsistentWith(const DeviceAnnots &that) const {
  return isConsistentWith(that.device);
}
