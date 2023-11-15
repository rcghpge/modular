//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Compile-time utilities for working with MGP buffers operands and operations.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_MGPDIALECT_MGPUTILS_H
#define GENERICML_MGPDIALECT_MGPUTILS_H

#include "GenericML/MGPDialect/MGPTypes.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/Operation.h"

namespace M::MGP {

/// Returns the !mgp.context in use in block, which can be found from:
///  - a block argument, such as the second argument to the 'execute' block,
///    or a control flow sub-block.
///  - the result of a mgp.context.create in the block, such as would be
///    found in the 'init' block.
///  - the parent block, if in a control flow sub-block which has not
///    carried the !mgp.context downwards. Though using the returned value
///    is illegal in the sub-block, the fixupIsolatedFromAbove can be used
///    to repair the model.
TypedValue<MGP::ContextType> getContextForBlock(Block *block);

//===----------------------------------------------------------------------===//
// CAUTION: The following methods are deprecated in favor of the
//          MGPInOutSemanticsOpInterface.
//===----------------------------------------------------------------------===//

// NOTE: We unfortunately  have two conventions for output buffer argument
// positions: at the front (eg mgp, kgen) and at the back (mop). Sigh.

/// Add attribute to op to record it has numOutputs output buffers at
/// the front (as per mgp and kgen convention).
void setNumFrontOutputBuffers(Operation *op, size_t numFrontOutputs);

/// Add attribute to op to record it has numOutputs output buffers at
/// the back (as per mop convention).
void setNumBackOutputBuffers(Operation *op, size_t numBackOutputs);

/// Removes any attribute for output buffers, both front and back.
void removeNumOutputBuffers(Operation *op);

/// Returns true if opOperand will be read by the operation.
bool isBufferReadingOpOperand(OpOperand &opOperand);

/// Returns true if opOperand will be written by the operation.
bool isBufferWritingOpOperand(OpOperand &opOperand);

} // namespace M::MGP

#endif // GENERICML_MGPDIALECT_MGPUTILS_H
