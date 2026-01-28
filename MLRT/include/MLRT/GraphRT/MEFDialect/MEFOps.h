//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares the operation classes for the MEF dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLRT_GRAPHRT_MEFDIALECT_MEFOPS_H
#define MLRT_GRAPHRT_MEFDIALECT_MEFOPS_H

#include "MLRT/GraphRT/MEFDialect/MEFTypes.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/MDialect/MAttrs.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

namespace mlir {
class ModuleOp;
}

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "MLRT/GraphRT/MEFDialect/MEF.h.inc"

#endif // MLRT_GRAPHRT_MEFDIALECT_MEFOPS_H
