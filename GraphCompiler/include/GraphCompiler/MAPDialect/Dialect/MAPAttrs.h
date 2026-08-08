//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Attributes and their helpers for the MAP dialect.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MAPDIALECT_MAPATTRS_H
#define GENERICML_GRAPHCOMPILER_MAPDIALECT_MAPATTRS_H

#include "GraphCompiler/MAPDialect/Dialect/MAPDialect.h"
#include "GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "GraphCompiler/MAPDialect/Dialect/MAPAttrs.h.inc"

namespace M::MAP {

/// Infers a slice's per-dimension steps from its output shape and static
/// starts/stops bounds: `(stop - start) / outLen`, which is 1 for the common
/// contiguous case. Falls back to `ShapedType::kDynamic` for a dimension
/// whenever the output length isn't statically known, either bound is itself
/// dynamic, or the division isn't exact.
llvm::SmallVector<int64_t> inferSliceSteps(MOSH::ShapeAttr out,
                                           llvm::ArrayRef<int64_t> starts,
                                           llvm::ArrayRef<int64_t> stops);

} // namespace M::MAP

#endif // GENERICML_GRAPHCOMPILER_MAPDIALECT_MAPATTRS_H
