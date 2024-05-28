//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/MGPDialect/MGPDialect.h"
#include "GenericML/MGPDialect/MGPTypes.h"
#include "Support/AssertStream.h"
#include "Support/Compiler/MLIRDType.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/ML/DType.h"
#include "Support/ML/SizeUtils.h"
#include "Support/ML/TensorShape.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace M;
using namespace MGP;

//===----------------------------------------------------------------------===//
// DeviceChainType
//===----------------------------------------------------------------------===//

DeviceChainType DeviceChainType::get(DeviceRefAttr device) {
  return DeviceChainType::get(device.getContext(), device);
}

//===----------------------------------------------------------------------===//
// DeviceBufferType
//===----------------------------------------------------------------------===//

/// Verifies the buffer size and align constraints are valid.
LogicalResult
DeviceBufferType::verify(function_ref<InFlightDiagnostic()> emitError,
                         DeviceRefAttr device,
                         std::optional<uint64_t> optSize) {
  if (!isValidOptSize(optSize))
    return emitError() << "buffer size must be strictly positive.";
  return success();
}

DeviceBufferType DeviceBufferType::get(DeviceRefAttr device,
                                       std::optional<uint64_t> optSize) {
  return DeviceBufferType::get(device.getContext(), device, optSize);
}

DeviceBufferType
DeviceBufferType::getChecked(function_ref<InFlightDiagnostic()> emitError,
                             DeviceRefAttr device,
                             std::optional<uint64_t> optSize) {
  return DeviceBufferType::get(device, optSize);
}

DeviceBufferType DeviceBufferType::get(DeviceRefAttr device, uint64_t rawSize) {
  return DeviceBufferType::get(device.getContext(), device,
                               rawSizeToOptSize(rawSize));
}

DeviceBufferType
DeviceBufferType::getChecked(function_ref<InFlightDiagnostic()> emitError,
                             DeviceRefAttr device, uint64_t rawSize) {
  return DeviceBufferType::get(device, rawSize);
}

DeviceBufferType DeviceBufferType::get(DeviceRefAttr device,
                                       IntegerAttr rawSizeOrNull) {
  std::optional<uint64_t> optSize;
  if (rawSizeOrNull)
    optSize = rawSizeToOptSize(rawSizeOrNull.getUInt());
  return DeviceBufferType::get(device.getContext(), device, optSize);
}

DeviceBufferType
DeviceBufferType::getChecked(function_ref<InFlightDiagnostic()> emitError,
                             DeviceRefAttr device, IntegerAttr rawSizeOrNull) {
  return DeviceBufferType::get(device, rawSizeOrNull);
}

DeviceBufferType DeviceBufferType::get(DeviceRefAttr device,
                                       TensorSpecType tensorSpecType) {
  return DeviceBufferType::get(device.getContext(), device,
                               tensorSpecType.getSizeInBytes());
}

DeviceBufferType
DeviceBufferType::getChecked(function_ref<InFlightDiagnostic()> emitError,
                             DeviceRefAttr device,
                             TensorSpecType tensorSpecType) {
  return get(device, tensorSpecType);
}

DeviceBufferType DeviceBufferType::get(DeviceTensorType tensorType) {
  return DeviceBufferType::get(tensorType.getContext(), tensorType.getDevice(),
                               tensorType.getTensorSpec().getSizeInBytes());
}

DeviceBufferType
DeviceBufferType::getChecked(function_ref<InFlightDiagnostic()> emitError,
                             DeviceTensorType tensorType) {
  return get(tensorType);
}

IntegerAttr DeviceBufferType::getRawSize() const {
  Type ui64 = IntegerType::get(getContext(), 64, IntegerType::Unsigned);
  APInt value(ui64.getIntOrFloatBitWidth(), optSizeToRawSize(getOptSize()),
              ui64.isSignedInteger());
  return IntegerAttr::get(ui64, value);
}

ShapedType DeviceBufferType::cloneWith(std::optional<ArrayRef<int64_t>> shape,
                                       Type elementType) const {
  ASSERT_STREAM(!shape) << "cannot change shape in DeviceBufferType::cloneWith";
  ASSERT_STREAM(elementType.isUnsignedInteger(8))
      << "invalid dtype for DeviceBufferType::cloneWith";
  return DeviceBufferType::get(getContext(), getDevice(), getOptSize());
}

Type DeviceBufferType::getElementType() const {
  return IntegerType::get(getContext(), /*width=*/8, IntegerType::Unsigned);
}

bool DeviceBufferType::hasRank() const { return true; }

// DeviceBufferType::getShape() implemented in MGPTypes.cpp since must see
// ODS implementation.

//===----------------------------------------------------------------------===//
// DeviceTensorType
//===----------------------------------------------------------------------===//

ShapedType DeviceTensorType::cloneWith(std::optional<ArrayRef<int64_t>> shape,
                                       Type elementType) const {
  auto tensorSpec = llvm::dyn_cast<TensorSpecType>(
      getTensorSpec().cloneWith(shape, elementType));
  return CpuTensorType::get(getContext(), tensorSpec);
}

bool DeviceTensorType::hasRank() const { return getTensorSpec().hasRank(); }

ArrayRef<int64_t> DeviceTensorType::getShape() const {
  return getTensorSpec().getShape();
}

Type DeviceTensorType::getElementType() const {
  return getTensorSpec().getElementType();
}

DeviceTensorType DeviceTensorType::get(DeviceRefAttr device,
                                       TensorSpecType tensorSpecType) {
  return DeviceTensorType::get(device.getContext(), device, tensorSpecType);
}

TensorSpecType DeviceTensorType::getTensorSpec(bool isShapeOnly) {
  if (isShapeOnly)
    return TensorSpecType::get(getContext(), getTensorSpec().getOptDims(),
                               std::optional<DType>());
  return getTensorSpec();
}
