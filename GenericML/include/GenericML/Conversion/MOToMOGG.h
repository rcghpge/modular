//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_CONVERSION_MOTOMOGG_H
#define GENERICML_CONVERSION_MOTOMOGG_H

#include "GenericML/MGPDialect/MGPOps.h"
#include "GenericML/MODialect/MOOps.h"
#include "GenericML/MOGGDialect/MOGGOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace M::MO {

/// Generate the code for declaring and registering the pass.
#define GEN_PASS_REGISTRATION
#define GEN_PASS_DECL
#include "GenericML/Conversion/MOToMOGG.h.inc"

} // namespace M::MO

#endif // GENERICML_CONVERSION_MOTOMOGG_H
