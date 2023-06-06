//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_MGPTRANSFORMS_UTILS_H
#define GENERICML_MGPTRANSFORMS_UTILS_H

#include "GenericML/MGPDialect/MGPOps.h"
#include "GraphRT/GRTDialect/GRTOps.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/SmallPtrSet.h"

namespace M::MGP {

/// Returns the operations in block in a dataflow order, ie s.t. an operation
/// does not have operands defined by operations later in the vector.
SmallVector<Operation *> getDataflowOrder(Block *block);

/// Returns all the OpOperands which are 'reading' or 'writing' occurrences
/// of buffer, depending on the isReading flag.
SmallVector<OpOperand *> allUsingOpOperands(Value buffer, bool isReading);

/// Return the OpOperands for all input chains to op.
SmallVector<OpOperand *, 1> getInputChainOpOperands(Operation *op);

/// Returns the unique writer to the given buffer. Returns nullptr if either
/// no writer or multiple writers.
Operation *getUniqueWriter(Value buffer);

/// Returns the 'earliest' operation (in block operations order) which uses
/// one of results as an operand.
Operation *earliestUsingOp(mlir::ResultRange results);

using ValueSetVector = llvm::SetVector<Value, SmallVector<Value>>;

/// If chain is defined by a grt.chain.merge operation then accumulate
/// all of its operands into primitiveChains (recursively). Otherwise adds
/// chain directly to primitiveChains.
void accumulatePrimitiveChains(ValueSetVector &primitiveChains, Value chain);

/// If targetOp is not before anchorOp, move it. If targetOp then has an
/// operand that is not before itself, move that also, and so on.
/// If your graph is not a DAG you're gonna have a bad time.
/// Though not required for correct MEF execution, it helps the
/// readability of the generated MLIR.
void moveAllBefore(Operation *origTargetOp, Operation *origAnchorOp);

/// Returns a chain value which will be ready only when all of chains (which
/// cannot be empty) are ready.
Value waitForChains(OpBuilder &builder, mlir::Location loc,
                    ArrayRef<Value> chains);

/// Mutates op in-place to ensure its input chain is ready only when all of
/// additionalChains are also ready. Returns failure if op doesn't have any
/// input chain to merge with.
LogicalResult alsoWaitForChains(OpBuilder &builder, Operation *op,
                                ArrayRef<Value> additionalChains);

/// Helper for modifying the 'init' block.
struct InitState {
  InitState(MLIRContext *context, Block *init);

  /// 'Allocates' a slot in the context.
  size_t allocSlot();

  /// Returns the current final 'init' chain result.
  Value getFinalChain();

  /// Update the final 'init' chain result.
  void updateFinalChain(Value newChain);

  /// Returns the 'init' block's context.
  Value getContext();

  Block *init;
  OpBuilder builder;
  ContextCreateOp contextCreateOp;
};

/// Helper for modifying the 'execute' block.
struct ExecuteState {
  ExecuteState(MLIRContext *context, Block *execute);

  /// Returns 'root' chain, ie the first argument to the 'execute' block.
  /// If any non-root chain is ready then the root chain is guaranteed to be
  /// ready, and can thus be ignored.
  Value getRootChain() const;

  /// Returns MGP context argument to the 'execute' block.
  Value getContext() const;

  /// Returns the additional arguments to 'execute' block
  iterator_range<mlir::BlockArgument *> getExtraArgs() const;

  /// Schedule op for later erasure. (We can't always erase eagerly if
  /// we're in the middle of, eg, an iteration over uses/users.)
  void scheduleErase(Operation *op);

  /// Erase all the 'toErase' entries.
  void erasePending();

  Block *execute;
  OpBuilder builder;
  SmallPtrSet<Operation *, 8> toErase;
};

} // namespace M::MGP

#endif // GENERICML_MGPTRANSFORMS_UTILS_H
