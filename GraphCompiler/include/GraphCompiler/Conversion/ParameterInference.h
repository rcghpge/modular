//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Parameter inference: derives a kernel's parameter dictionary from an MO op's
// attributes. Used by the MO->MAP lowering.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_CONVERSION_PARAMETERINFERENCE_H
#define GENERICML_GRAPHCOMPILER_CONVERSION_PARAMETERINFERENCE_H

#include "GraphCompiler/MOGGDialect/Analysis/KernelDeclAdaptor.h"
#include "GraphCompiler/MOGGDialect/Support/MojoBuilder.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"

#include <functional>

namespace M {

/// Converts a kernel parameter to a spelling emit accepts.
mlir::Attribute normalizeKernelParameter(mlir::Attribute value);

/// Infers `op`'s kernel parameters, normalized for emit.
mlir::DictionaryAttr inferParameters(mlir::Operation *op,
                                     mlir::OpBuilder &builder);

/// Appends the parameters that ride on `op`'s SSA operands instead of its
/// attributes, as i64 arrays.
void appendStaticOperandParameters(
    mlir::Operation *op, mlir::OpBuilder &builder,
    llvm::SmallVectorImpl<mlir::NamedAttribute> &params);

} // namespace M

namespace M::MOGG {

struct InferredParameters {
  llvm::SmallVector<MojoValue> structParameters;
  llvm::SmallVector<MojoValue> functionParameters;
};

/// Appends the kernel parameters (`P0`, `P1`, ...) implied by `moOp`'s
/// `!mo.opaque` operand and result types, resolved through the kernel's
/// `mogg.value_params` / `mogg.value_result_params` annotations. A kernel like
/// `MakeSimdPair.execute[P0, P1]() -> SIMDPair[P0, P1]` names them nowhere
/// else.
void inferOpaqueParameters(
    llvm::SmallVectorImpl<mlir::NamedAttribute> &forwardedParameters,
    mlir::Operation *moOp, const KernelDeclFunctionAdaptor &function,
    mlir::ArrayAttr inputParams);

/// Fill and return the list of inferred parameters for a struct kernel
/// function (execute, shape, elementwise, or build_graph). Uses the stored
/// struct-level and fn-level ParamDeclAttrs from the KernelDeclAdaptor.
/// Non-const because `FnTypeGeneratorType`'s pog list metadata accessor is.
ErrorOr<InferredParameters> inferKernelParameters(
    KernelDeclAdaptor &adaptor, llvm::StringRef funcLabel,
    std::function<ErrorOr<MojoValue>(StringRef)> &&inferCallback);

/// Fill and return the list of inferred parameters for a standalone
/// primitive function (mogg.primitive.decl). No struct-level params.
ErrorOr<InferredParameters> inferKernelParameters(
    PrimitiveDeclOp primDecl,
    std::function<ErrorOr<MojoValue>(StringRef)> &&inferCallback);

/// Fill and return the list of inferred parameters for a standalone shape
/// function (mogg.shape_func.decl registered via @register_shape_function).
/// No struct-level params.
ErrorOr<InferredParameters> inferKernelParameters(
    ShapeFunctionDeclOp shapeDecl,
    std::function<ErrorOr<MojoValue>(StringRef)> &&inferCallback);

} // namespace M::MOGG

#endif // GENERICML_GRAPHCOMPILER_CONVERSION_PARAMETERINFERENCE_H
