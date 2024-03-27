//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares the operation classes for the RMO dialect.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_RMODIALECT_RMOOPS_H
#define GENERICML_GRAPHCOMPILER_RMODIALECT_RMOOPS_H

#include "GenericML/GraphCompiler/MODialect/MOAttrs.h"
#include "GenericML/GraphCompiler/MODialect/MOOps.h"
#include "GenericML/GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "GenericML/GraphCompiler/RMODialect/RMOInterfaces.h"
#include "KGEN/KGENDialect/KGENTypes.h"
#include "mlir/IR/OpImplementation.h"

using M::KGEN::ParamDeclAttr;
using M::MOSH::ShapeAttr;

namespace M::RMO {

/// Given two tensor types which are meant to represent the same shape
/// return a new type which returns a more "refined type" using information
/// from both existing types
MO::TensorType refineShape(MO::TensorType ty1, MO::TensorType ty2);
MO::TensorType refineShapeWithDType(MO::TensorType ty1, MO::TensorType ty2,
                                    DType dtype);

/// Rebinds all values
SmallVector<Value> rebindUnknownShapeInValues(ValueRange inputValues,
                                              MO::GraphOp graphOp,
                                              OpBuilder &builder);
/// inferMoTypesImpl for all scatter-like ops
void inferMOTypesScatterImpl(
    MLIRContext *context, MO::GraphOp graphOp, TensorValue input,
    TensorValue updates, TensorValue indices, TensorValue axis,
    std::optional<ValueRange> rmoResults, SmallVectorImpl<Type> &moOperandTypes,
    SmallVectorImpl<Type> &moResultTypes,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moInputParamDecls,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moOutputParamDecls);

/// inferMoTypesImpl for all elementwise-unary-like ops
void inferMOTypesElemwiseUnaryImpl(
    MLIRContext *context, MO::GraphOp graphOp, TensorValue input,
    std::optional<ValueRange> rmoResults, SmallVectorImpl<Type> &moOperandTypes,
    SmallVectorImpl<Type> &moResultTypes,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moInputParamDecls,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moOutputParamDecls);

} // namespace M::RMO
//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "GenericML/GraphCompiler/RMODialect/RMO.h.inc"

#endif // GENERICML_GRAPHCOMPILER_RMODIALECT_RMOOPS_H
