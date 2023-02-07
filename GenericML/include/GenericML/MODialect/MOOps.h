//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares the operation classes for the MO dialect.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_MODIALECT_MOOPS_H
#define GENERICML_MODIALECT_MOOPS_H

#include "GenericML/MODialect/MODialect.h"
#include "GenericML/MODialect/MOInterfaces.h"
#include "GenericML/MODialect/MOTypes.h"
#include "KGEN/KGENDialect/KGENInterfaces.h"
#include "KGEN/KGENDialect/KGENTypes.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir {
class ModuleOp;
} // namespace mlir

namespace M::MO::impl {
LogicalResult resolveSameOperandsAndResultTypeUnknownParameters(
    MOOpParameterizationInterface op, mlir::RewriterBase &rewriter);
} // namespace M::MO::impl

// Needed because the ODS definition of some KGEN ops don't have the namespace.
using M::KGEN::ParamDeclArrayAttr;
using M::KGEN::ParamDeclAttr;
using M::KGEN::SignatureType;

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "GenericML/MODialect/MO.h.inc"

#endif // GENERICML_MODIALECT_MOOPS_H
