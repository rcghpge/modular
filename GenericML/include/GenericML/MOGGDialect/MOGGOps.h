//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares the operation classes for the MO dialect.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_MOGGDIALECT_MOGGOPS_H
#define GENERICML_MOGGDIALECT_MOGGOPS_H

#include "GenericML/MODialect/MOOps.h"
#include "GenericML/MODialect/MOTypes.h"
#include "GenericML/MOGGDialect/MOGGDialect.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "GenericML/MOGGDialect/MOGG.h.inc"

#endif // GENERICML_MOGGDIALECT_MOGGOPS_H
