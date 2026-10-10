//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares types for the MO dialect.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MODIALECT_MOTYPES_H
#define GENERICML_GRAPHCOMPILER_MODIALECT_MOTYPES_H

#include "GraphCompiler/MODialect/MOEnums.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/MDialect/MAttrs.h"
#include "Support/ML/DType.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Types.h"

namespace M::MOSH {
class ChainType;
class ShapeAttr;
class ShapeType;
} // namespace M::MOSH

namespace M::MO {
class IOKindAttr;
class LayoutAttr;
class MOHasDeviceTypeInterface;

/// Kernel-operand category of a type, from `MOKernelOperandTypeInterface`.
/// `Buffer` is a mutable tensor.
enum class KernelOperandKind { Tensor, Buffer, Scalar, Opaque };
} // namespace M::MO

namespace mlir {
/// For automated parsing of M::DType.
template <>
struct FieldParser<M::DType> {
  static FailureOr<M::DType> parse(AsmParser &parser);
};
} // namespace mlir

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "GraphCompiler/MODialect/MOTypeInterfaces.h.inc"
#include "GraphCompiler/MODialect/MOTypes.h.inc"

namespace M::MO {
using ScalarValue = M::TypedValue<M::MO::ScalarType>;
using ShapeAttr = MOSH::ShapeAttr;
using ShapeType = MOSH::ShapeType;
using TensorValue = M::TypedValue<M::MO::TensorType>;
using ListValue = M::TypedValue<M::MO::ListType>;
using BufferValue = M::TypedValue<M::MO::BufferType>;
using ChainValue = M::TypedValue<M::MO::ChainType>;

ErrorOr<DType> fromMLIRType(Type mlirType);
ErrorOr<Type> toMLIRType(MLIRContext *context, DType dType);

/// Return the dtype of `ty` (possibly a composite type like TensorType), or
/// DType::invalid. Used to extract dtypes in tablegen.
DType getDTypeOrInvalid(Type ty);

TypedAttr getShape(Type ty);

/// Return a TensorType as is, or the TensorType equivalent of a BufferType;
/// nullptr otherwise.
TensorType dynCastFromTensorLike(Type ty);

/// Like `dynCastFromTensorLike`, but asserts `ty` is a TensorType or
/// BufferType.
TensorType castFromTensorLike(Type ty);

/// Returns strides from the tensor's static shape. If `unknown`, all strides
/// are dynamic; otherwise at least the innermost stride is 1.
llvm::SmallVector<int64_t> getStridesForTensor(MO::TensorType tensor,
                                               bool unknown);

/// Print device refs as `mo.bundle`'s compact `label:first[-last]` form when
/// homogeneous and contiguous, otherwise as comma-separated `label:id`.
///
/// Examples:
///   {gpu:0}             -> `gpu:0`
///   {gpu:0, gpu:1}      -> `gpu:0-1`
///   {gpu:0, cpu:0}      -> `gpu:0, cpu:0`
///   {gpu:0, gpu:2}      -> `gpu:0, gpu:2`
void printDeviceList(mlir::AsmPrinter &printer,
                     llvm::ArrayRef<DeviceRefAttr> devices);

/// Parse a list of device refs printed by `printDeviceList`.
mlir::ParseResult
parseDeviceList(mlir::AsmParser &parser,
                llvm::SmallVectorImpl<DeviceRefAttr> &devices);

} // namespace M::MO

#endif // GENERICML_GRAPHCOMPILER_MODIALECT_MOTYPES_H
