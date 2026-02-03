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

#include "GenericML/GraphCompiler/MODialect/MOEnums.h"
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
class LayoutAttr;
class MOHasDeviceTypeInterface;
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
#include "GenericML/GraphCompiler/MODialect/MOTypeInterfaces.h.inc"
#include "GenericML/GraphCompiler/MODialect/MOTypes.h.inc"

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

/// Return the dtype of `ty`.
/// `ty` might be a composite type (eg TensorType).
/// Returns DType::invalid if no valid dtype is found.
/// Helper used to extract dtype in tablegen.
DType getDTypeOrInvalid(Type ty);

TypedAttr getShape(Type ty);

/// Helper function for working with mo.buffer types. If given a TensorType will
/// pass it through and if given a BufferType will create the equivalent
/// TensorType and return that. If not given one of these two types returns
/// nullptr.
TensorType dynCastFromTensorLike(Type ty);

/// Helper function for working with mo.buffer types. If given a TensorType will
/// pass it through and if given a BufferType will create the equivalent
/// TensorType and return that. Asserts that ty is one of BufferType or
/// TensorType.
TensorType castFromTensorLike(Type ty);

/// Returns the calculated strides based on the static shape of the tensor.
/// If `unknown` is true, then the strides for all dimensions will be dynamic,
/// otherwise the result will have at least the innermost dimension have a
/// stride of 1.
llvm::SmallVector<int64_t> getStridesForTensor(MO::TensorType tensor,
                                               bool unknown);

} // namespace M::MO

#endif // GENERICML_GRAPHCOMPILER_MODIALECT_MOTYPES_H
