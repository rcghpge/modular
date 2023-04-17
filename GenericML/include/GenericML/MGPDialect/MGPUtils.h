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

// NOTE: It would be much better if we had an MGPWritesBuffersInterface
// for ops so they can self-describe which of their buffer arguments they
// write to. However mop ops have no ODS definition so there's nowhere
// to carry such an interface. Instead we must make do with attributes.

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
