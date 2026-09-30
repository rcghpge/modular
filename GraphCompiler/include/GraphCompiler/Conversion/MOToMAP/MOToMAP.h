//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_CONVERSION_MOTOMAP_H
#define GENERICML_GRAPHCOMPILER_CONVERSION_MOTOMAP_H

#include "GraphCompiler/MODialect/MOUtils.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace M::MOGG {
class MojoRegistry;
} // namespace M::MOGG

namespace M::MO {

#define GEN_PASS_REGISTRATION
#define GEN_PASS_DECL
#include "GraphCompiler/Conversion/MOToMAP/MOToMAP.h.inc"

/// Lowers a single MO kernel op to its MAP form in place, using `registry` to
/// resolve the kernel decl -- the same path the `mo-to-map` pass runs over each
/// op. Returns the created `map.iter.opaque` for the opaque kernel path (the
/// only case a caller needs the op back, e.g. to stamp a write effect), a null
/// `Operation *` for the other successful paths and for an op with no kernel
/// adaptor (left unchanged), or `failure()` on a lowering error.
///
/// Exposed for `NanCheckPass`, which injects `mo.custom` check kernels after
/// `mo-to-map` has already run and must compile them through the same lowering
/// rather than the deleted legacy MO->MOGG builder.
FailureOr<Operation *> lowerMOOpToMAP(Operation *op,
                                      MOGG::MojoRegistry &registry);

} // namespace M::MO

#endif // GENERICML_GRAPHCOMPILER_CONVERSION_MOTOMAP_H
