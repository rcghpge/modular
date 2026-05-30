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

#include "GraphCompiler/GraphCompiler/MODialect/MOAttrs.h"
#include "GraphCompiler/GraphCompiler/MODialect/MOOps.h"
#include "GraphCompiler/GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "GraphCompiler/GraphCompiler/RMODialect/RMOInterfaces.h"
#include "KGEN/KGENDialect/KGENTypes.h"
#include "mlir/IR/OpImplementation.h"

using M::KGEN::ParamDeclAttr;
using M::MO::parseTransferResult;
using M::MO::printTransferResult;
using M::MOSH::ShapeAttr;

namespace M::RMO {

/// Given two tensor types which are meant to represent the same shape
/// return a new type which returns a more "refined type" using information
/// from both existing types
MO::TensorType refineShape(MO::TensorType ty1, MO::TensorType ty2);
MO::TensorType refineShapeWithDType(MO::TensorType ty1, MO::TensorType ty2,
                                    DType dtype);

/// inferMoTypesImpl for scatter-reduction ops whose axis is a scalar tensor
/// operand (`scatter.add/max/min/mul`).
void inferMOTypesScatterImpl(
    MLIRContext *context, MO::GraphOp graphOp, TensorValue input,
    TensorValue updates, TensorValue indices, TensorValue axis,
    std::optional<ValueRange> rmoResults, SmallVectorImpl<Type> &moOperandTypes,
    SmallVectorImpl<Type> &moResultTypes,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moInputParamDecls,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moOutputParamDecls);

/// inferMoTypesImpl for `scatter`, whose axis is a compile-time `index`
/// attribute (so it contributes no MO operand type).
void inferMOTypesScatterImpl(
    MLIRContext *context, MO::GraphOp graphOp, TensorValue input,
    TensorValue updates, TensorValue indices,
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

/// inferMoTypesImpl for all unary-like ops where output has an explicit type.
void inferMOTypesUnaryExplicitOutputTypeImpl(
    MLIRContext *context, MO::GraphOp graphOp, TensorValue input,
    std::optional<ValueRange> rmoResults, SmallVectorImpl<Type> &moOperandTypes,
    SmallVectorImpl<Type> &moResultTypes,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moInputParamDecls,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moOutputParamDecls);

/// inferMoTypesImpl for TopK-like ops
void inferMOTypesTopKImpl(
    MLIRContext *context, MO::GraphOp graphOp, TensorValue input, TensorValue k,
    TensorValue axis, TensorValue sorted, std::optional<ValueRange> rmoResults,
    SmallVectorImpl<Type> &moOperandTypes, SmallVectorImpl<Type> &moResultTypes,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moInputParamDecls,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moOutputParamDecls);

/// inferMoTypesImpl for all scatternd-like ops
void inferMOTypesScatterNdImpl(
    MLIRContext *context, MO::GraphOp graphOp, TensorValue input,
    TensorValue updates, TensorValue indices,
    std::optional<ValueRange> rmoResults, SmallVectorImpl<Type> &moOperandTypes,
    SmallVectorImpl<Type> &moResultTypes,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moInputParamDecls,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moOutputParamDecls);

/// Given two dtypes, returns the promotion dtype or an error.
/// The rules are:
/// 1. Category is an ordered hierarchy: `bool < unsigned int < signed int <
/// float`
/// 2. Promotion of values `x` and `y` picks:
///    (max(category(x), category(y)), max(bitwidth(x), bitwidth(y))
/// 3. Promotion raises an error if either input dtype might contain a value
///    unrepresentable by the chosen promotion type.
/// 4. Promotion raises an error if two dtypes have the same category and
/// bitwidth
///    but different representations.
ErrorOr<DType> promoteDtype(DType x, DType y);

/// Return the device attached to the type if has a device.
std::optional<M::DeviceRefAttr> getDeviceIfAvailable(Type t);
std::optional<M::DeviceRefAttr> getDeviceIfAvailable(Value v);

/// As `getDeviceIfAvailable` but assume device exists, otherwise get exception.
M::DeviceRefAttr getDevice(Type t);
M::DeviceRefAttr getDevice(Value v);

} // namespace M::RMO
//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "GraphCompiler/GraphCompiler/RMODialect/RMO.h.inc"

#endif // GENERICML_GRAPHCOMPILER_RMODIALECT_RMOOPS_H
