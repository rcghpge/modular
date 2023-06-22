//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_MGPTRANSFORMS_FIXUPISOLATEDFROMABOVE_H
#define GENERICML_MGPTRANSFORMS_FIXUPISOLATEDFROMABOVE_H

#include "GenericML/MGPDialect/MGPOps.h"
#include "mlir/IR/Builders.h"

namespace M::MGP {

/// Returns a copy of inModelOp with the 'init' and 'execute' blocks rewritten
/// to ensure all control flow ops are correctly IsolatedFromAbove. Any free
/// values in sub-blocks are captured by new sub-block arguments, and the
/// overall control flow ops are rewritten to include (and, for loops, return)
/// those arguments.
ModelOp fixupIsolatedFromAbove(mlir::OpBuilder &builder, ModelOp inModelOp);

} // namespace M::MGP

#endif // GENERICML_MGPTRANSFORMS_FIXUPISOLATEDFROMABOVE_H
