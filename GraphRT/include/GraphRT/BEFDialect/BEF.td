//===- BEF.td -------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This is the top level file for the BEF dialect.
//
//===----------------------------------------------------------------------===//

#ifndef BEF_TD
#define BEF_TD

include "GraphRT/BEFDialect/BEFDialect.td"
include "mlir/IR/OpBase.td"

// Base class for the operation in this dialect.
class BEFOp<string mnemonic, list<Trait> traits = []> :
    Op<BEFDialect, mnemonic, traits>;

include "GraphRT/BEFDialect/BEFNonStrict.td"
include "GraphRT/BEFDialect/BEFTypes.td"
include "GraphRT/BEFDialect/BEFOps.td"

#endif // BEF_TD
