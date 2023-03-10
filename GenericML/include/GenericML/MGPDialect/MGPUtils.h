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

/// Add attribute to op to record it has numOutputs output buffers.
void setNumOutputBuffers(Operation *op, size_t numOutputs);

/// Returns the number of output buffers for op. Will default to zero
/// if the op has not atttribute.
size_t getNumOutputBuffers(Operation *op);

/// Removes any attribute for output buffers.
void removeNumOutputBuffers(Operation *op);

/// Returns true if opOperand is a reading occurrence of a buffer.
bool isBufferReadingOpOperand(OpOperand &opOperand);

/// Returns true if opOperand is a writing occurrence of a buffer.
bool isBufferWritingOpOperand(OpOperand &opOperand);

} // namespace M::MGP

#endif // GENERICML_MGPDIALECT_MGPUTILS_H
