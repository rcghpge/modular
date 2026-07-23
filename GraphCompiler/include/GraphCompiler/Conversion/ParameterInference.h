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

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"

namespace M {

/// Infers the kernel parameter dictionary for `op` from its attributes: custom
/// ops forward their explicit `parameters`; for every other op, type attributes
/// become dtype constants and the remaining typed attributes pass through
/// (device references are dropped).
mlir::DictionaryAttr inferParameters(mlir::Operation *op,
                                     mlir::OpBuilder &builder);

} // namespace M

#endif // GENERICML_GRAPHCOMPILER_CONVERSION_PARAMETERINFERENCE_H
