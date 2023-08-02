//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_MGPTRANSFORMS_FIXUPISOLATEDFROMABOVE_H
#define GENERICML_MGPTRANSFORMS_FIXUPISOLATEDFROMABOVE_H

#include "GenericML/MGPDialect/MGPOps.h"
#include "mlir/IR/Builders.h"

#include "./IndexSet.h"

namespace M::MGP {

/// Returns a copy of inModelOp with the 'init' and 'execute' blocks rewritten
/// to ensure all control flow ops are correctly IsolatedFromAbove. Any free
/// values in sub-blocks are captured by new sub-block arguments, and the
/// overall control flow ops are rewritten to include (and, for loops, return)
/// those arguments. The returned model is built after inModelOp, the caller
/// is responsible for erasing inModelOp.
///
/// TODO(#15830): Can mildly generalize so can be also used for MO control
/// flow ops.
ModelOp fixupIsolatedFromAbove(mlir::OpBuilder &builder, ModelOp inModelOp);

} // namespace M::MGP

#endif // GENERICML_MGPTRANSFORMS_FIXUPISOLATEDFROMABOVE_H
