//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MGPDialect/Dialect/MGPDialect.h"
#include "GraphCompiler/MGPDialect/Dialect/MGPTypes.h"
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
// DeviceBufferType
//===----------------------------------------------------------------------===//

/// Verifies the buffer constraints are valid.
LogicalResult
DeviceBufferType::verify(function_ref<InFlightDiagnostic()> emitError,
                         DeviceRefAttr device) {
  return success();
}

DeviceBufferType DeviceBufferType::get(DeviceRefAttr device) {
  return DeviceBufferType::get(device.getContext(), device);
}

DeviceBufferType
DeviceBufferType::getChecked(function_ref<InFlightDiagnostic()> emitError,
                             DeviceRefAttr device) {
  return DeviceBufferType::get(device);
}

DeviceBufferType DeviceBufferType::get(DeviceTensorType tensorType) {
  return DeviceBufferType::get(tensorType.getContext(), tensorType.getDevice());
}

DeviceBufferType
DeviceBufferType::getChecked(function_ref<InFlightDiagnostic()> emitError,
                             DeviceTensorType tensorType) {
  return get(tensorType);
}

ShapedType DeviceBufferType::cloneWith(std::optional<ArrayRef<int64_t>> shape,
                                       Type elementType) const {
  ASSERT_STREAM(
      !shape, << "cannot change shape in DeviceBufferType::cloneWith");
  ASSERT_STREAM(elementType.isUnsignedInteger(8),
                << "invalid dtype for DeviceBufferType::cloneWith");
  return DeviceBufferType::get(getContext(), getDevice());
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
  return DeviceTensorType::get(getDevice(), tensorSpec);
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
