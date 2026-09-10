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
/// attributes, as i64 arrays. Shared with the legacy MO->MOGG lowering so both
/// paths agree on the names and the encoding.
void appendStaticOperandParameters(
    mlir::Operation *op, mlir::OpBuilder &builder,
    llvm::SmallVectorImpl<mlir::NamedAttribute> &params);

} // namespace M

namespace M::MOGG {

struct InferredParameters {
  llvm::SmallVector<MojoValue> structParameters;
  llvm::SmallVector<MojoValue> functionParameters;
};

/// Fill and return the list of inferred parameters for a struct kernel
/// function (execute, shape, elementwise, or update_view). Uses the stored
/// struct-level and fn-level ParamDeclAttrs from the KernelDeclAdaptor.
/// Non-const because reading the function signature's pog list metadata
/// is non-const-callable on `FnTypeGeneratorType`.
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
