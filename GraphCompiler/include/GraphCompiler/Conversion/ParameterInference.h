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

/// Infers the kernel parameter dictionary for `op` from its attributes: custom
/// ops forward their explicit `parameters`; for every other op, type attributes
/// become dtype constants and the remaining typed attributes pass through
/// (device references are dropped).
mlir::DictionaryAttr inferParameters(mlir::Operation *op,
                                     mlir::OpBuilder &builder);

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
