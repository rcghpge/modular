//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares the operation classes for the MOSH dialect.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MOSHDIALECT_MOSHOPS_H
#define GENERICML_GRAPHCOMPILER_MOSHDIALECT_MOSHOPS_H

#include "GenericML/GraphCompiler/MOSHDialect/MOSHTypes.h"
#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/KGENInterfaces.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir {
class RewriterBase;
} // namespace mlir

namespace M::MOSH {
class ShapeAttr;
} // namespace M::MOSH

namespace M::KGEN {
class FuncOp;
} // namespace M::KGEN

// Needed because the ODS definition of some KGEN ops don't have the namespace.
using M::KGEN::ParamDeclArrayAttr;
using M::KGEN::ParamDeclAttr;
using M::KGEN::ParamDefValue;
using M::KGEN::ParameterExprArrayAttr;
using M::KGEN::printStringParam;
using M::KGEN::SignatureType;

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "GenericML/GraphCompiler/MOSHDialect/MOSH.h.inc"

#endif // GENERICML_GRAPHCOMPILER_MOSHDIALECT_MOSHOPS_H
