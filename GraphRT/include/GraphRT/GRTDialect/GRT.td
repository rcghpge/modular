//===- GraphRT/GRTDialect/GRT.td ---------------------------*- tablegen -*-===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This is the top level file for the GRT dialect.
//
//===----------------------------------------------------------------------===//

#ifndef GRT_TD
#define GRT_TD

include "mlir/IR/OpBase.td"
include "GraphRT/GRTDialect/GRTDialect.td"
include "GraphRT/BEFDialect/BEFNonStrict.td"

// Base class for the operation in this dialect.
class GRTOp<string mnemonic, list<Trait> traits = []> :
    Op<GRTDialect, mnemonic, traits>;

include "GraphRT/GRTDialect/GRTTypes.td"
include "GraphRT/GRTDialect/GRTAnalyticsOps.td"
include "GraphRT/GRTDialect/GRTIntegerOps.td"
include "GraphRT/GRTDialect/GRTControlFlowOps.td"

#endif // GRT_TD
