//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares the operation classes for the MOP dialect.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_MOPDIALECT_MOPOPS_H
#define GENERICML_MOPDIALECT_MOPOPS_H

#include "GenericML/MOPDialect/MOPAttrs.h"
#include "GenericML/MOPDialect/MOPDialect.h"
#include "GenericML/MOPDialect/MOPTypes.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"

namespace mlir {
class ModuleOp;
}

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "GenericML/MOPDialect/MOP.h.inc"

#endif // GENERICML_MOPDIALECT_MOPOPS_H
