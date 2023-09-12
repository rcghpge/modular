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
#include "GenericML/GraphCompiler/MODialect/MOTypes.h.inc"

namespace M::MO {
using ChainType = MOSH::ChainType;
using ShapeAttr = MOSH::ShapeAttr;
using ShapeType = MOSH::ShapeType;
using TensorValue = M::TypedValue<M::MO::TensorType>;

ErrorOr<DType> fromMLIRType(Type mlirType);
ErrorOr<Type> toMLIRType(MLIRContext *context, DType dType);
} // namespace M::MO

#endif // GENERICML_GRAPHCOMPILER_MODIALECT_MOTYPES_H
