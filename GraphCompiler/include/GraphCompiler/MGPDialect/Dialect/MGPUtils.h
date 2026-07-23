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

#include "GraphCompiler/MGPDialect/Dialect/MGPTypes.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/Operation.h"

namespace M::MGP {

/// Lower a dimension to an mlir index.
Value dimensionToIndex(OpBuilder &builder, Location loc, Value dim);
Value indexToDimension(OpBuilder &builder, Location loc, Value index);

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

/// Returns the !mef.chain representing the 'root' chain for block, which can
/// be found from:
///  - a block argument, such as the first argument to the 'execute' block,
///    or a control flow sub-block.
///  - the result of a nullary grt.chain.create in the block, such as would be
///    found in the 'init' block.
///  - the parent block, if in a control flow sub-block which has not
///    carried a !mef.chain downwards. Though using the returned value
///    is illegal in the sub-block, the fixupIsolatedFromAbove can be used
///    to repair the model.
TypedValue<MEF::ChainType> getRootChainForBlock(Block *block);

/// Record the given op has the given in/out semantics.
///
/// Only to be used for ops without an ODS definition. All others should
/// implement the MGPInOutSemanticsOpInterface interface.
void setInOutSignature(Operation *op, InOutSignatureAttr inOutSig);

/// Removes any attribute for in/out semantics added by setInOutSignature.
///
/// Ops without ODS definitions which will be interpreted by MEF cannot
/// retain the attribute without modifying the primitive's implementation.
void removeInOutSignature(Operation *op);

/// Returns the in/out semantics for operand.
InOutSignatureAttr::InOutSemantics getInOutSemantics(OpOperand &opOperand);

/// Returns true if operand is mutated by its op (ie it has either an 'out'
/// or an 'in/out' semantics).
bool isMutated(OpOperand &opOperand);

/// Returns true if operand is read by its op (ie it has either an 'in'
/// or 'in/out' semantics).
///
/// CAUTION: This does not mean 'read only'.
bool isRead(OpOperand &opOperand);

} // namespace M::MGP

#endif // GENERICML_MGPDIALECT_MGPUTILS_H
