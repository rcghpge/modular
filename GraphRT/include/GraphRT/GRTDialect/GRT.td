//===- GRT.td -------------------------------------------------------------===//
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

include "GraphRT/GRTDialect/GRTDialect.td"
include "mlir/IR/OpBase.td"

// Base class for the operation in this dialect.
class GRTOp<string mnemonic, list<Trait> traits = []> :
    Op<GRTDialect, mnemonic, traits>;

include "GraphRT/GRTDialect/GRTBenchmarkingOps.td"
include "GraphRT/GRTDialect/GRTIntegerOps.td"
include "GraphRT/GRTDialect/GRTControlFlowOps.td"
include "GraphRT/GRTDialect/GRTTestOps.td"

#endif // GRT_TD
