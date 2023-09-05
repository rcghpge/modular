//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares the operation classes for the MEF dialect.
//
//===----------------------------------------------------------------------===//

#ifndef GRAPHRT_MEFDIALECT_MEFOPS_H
#define GRAPHRT_MEFDIALECT_MEFOPS_H

#include "GraphRT/MEFDialect/MEFDialect.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

namespace mlir {
class ModuleOp;
}

#define GET_OP_CLASSES
#include "GraphRT/MEFDialect/MEF.h.inc"

#define GET_TYPEDEF_CLASSES
#include "GraphRT/MEFDialect/MEFTypes.h.inc"

#endif // GRAPHRT_MEFDIALECT_MEFOPS_H
