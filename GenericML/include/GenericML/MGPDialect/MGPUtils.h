//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_MGPDIALECT_MGPUTILS_H
#define GENERICML_MGPDIALECT_MGPUTILS_H

#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/Operation.h"

namespace M::MGP {

/// Add attribute to op to record it has numOutputs output buffers.
void setNumOutputBuffers(Operation *op, size_t numOutputs);

/// Returns the number of output buffers for op. Will default to zero
/// if the op has not atttribute.
size_t getNumOutputBuffers(Operation *op);

/// Removes any attribute for output buffers.
void removeNumOutputBuffers(Operation *op);

/// Returns the operand index for the first output buffer argument to op.
/// Returnn the op's arity if it has no output buffer arguments.
size_t getFirstOutputBufferOperandIndex(Operation *op);

} // namespace M::MGP

#endif // GENERICML_MGPDIALECT_MGPUTILS_H
