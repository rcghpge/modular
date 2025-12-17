//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This defines the base ops in the GRT dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLRT_GRAPHRT_GRTDIALECT_GRTOPSBASE_TD
#define MLRT_GRAPHRT_GRTDIALECT_GRTOPSBASE_TD

include "MLRT/GraphRT/GRTDialect/GRTDialect.td"
include "mlir/IR/OpBase.td"

// Base class for the operation in this dialect.
class GRTOp<string mnemonic, list<Trait> traits = []> :
    Op<GRTDialect, mnemonic, traits>;

#endif // MLRT_GRAPHRT_GRTDIALECT_GRTOPSBASE_TD
