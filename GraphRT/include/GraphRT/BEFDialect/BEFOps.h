//===- BEFOps.h - Declare BEF dialect operations ----------------*- C++ -*-===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares the operation classes for the BEF dialect.
//
//===----------------------------------------------------------------------===//

#ifndef GRAPHRT_BEFDIALECT_BEFOPS_H
#define GRAPHRT_BEFDIALECT_BEFOPS_H

#include "GraphRT/BEFDialect/BEFDialect.h"
#include "Support/LLVM.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"

namespace mlir {
class ModuleOp;
}

namespace M {
namespace BEF {
// TODO: Helpers here.
} // namespace BEF
} // namespace M

#define GET_OP_CLASSES
#include "GraphRT/BEFDialect/BEF.h.inc"

#define GET_TYPEDEF_CLASSES
#include "GraphRT/BEFDialect/BEFTypes.h.inc"

#endif // GRAPHRT_BEFDIALECT_BEFOPS_H
