//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares the operation classes for the MAP dialect.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MAPDIALECT_MAPOPS_H
#define GENERICML_GRAPHCOMPILER_MAPDIALECT_MAPOPS_H

#include "GraphCompiler/Interfaces/LowerableToMOGGInterface.h"
#include "GraphCompiler/MAPDialect/Dialect/MAPAttrs.h"
#include "GraphCompiler/MAPDialect/Dialect/MAPDialect.h"
#include "GraphCompiler/MAPDialect/Dialect/MAPOpInterfaces.h"
#include "GraphCompiler/MAPDialect/Dialect/MAPTypes.h"
#include "GraphCompiler/MODialect/MOInterfaces.h"
#include "GraphCompiler/MODialect/MOTypes.h"
#include "GraphCompiler/MOGGDialect/MOGGAttrs.h"
#include "GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "GraphCompiler/MAPDialect/Dialect/MAP.h.inc"

namespace M::MAP {

/// The single `map.store` in a kernel or closure `body`, asserting there is
/// exactly one (every MAP kernel is single-output today -- see the multi-output
/// TODO in input fusion). Prefer this over assuming the store is the body's
/// terminator.
StoreOp getUniqueStore(mlir::Block &body);

/// Merges a MAP kernel's duplicate input operands (a `map.iter.foreach` or
/// `map.iter.opaque`) onto the first occurrence's shadowed block argument and
/// drops the extras, returning whether anything changed. Shared by the
/// `Dedupe*Inputs` canonicalizations and by input/output fusion, which runs it
/// on just the kernel it fused into rather than canonicalizing the whole graph.
bool dedupeKernelInputs(mlir::Operation *kernel, mlir::RewriterBase &rewriter);

} // namespace M::MAP

#endif // GENERICML_GRAPHCOMPILER_MAPDIALECT_MAPOPS_H
