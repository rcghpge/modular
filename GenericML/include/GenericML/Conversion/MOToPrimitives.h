//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_CONVERSION_MOTOPRIMITIVES_H
#define GENERICML_CONVERSION_MOTOPRIMITIVES_H

#include "mlir/Pass/Pass.h"

namespace M::MGP {

/// Generate the code for declaring the pass.
#define GEN_PASS_DECL_MOTOPRIMITIVES
#include "GenericML/Conversion/MOToPrimitives.h.inc"

/// Generate the code for registering the pass.
#define GEN_PASS_REGISTRATION
#include "GenericML/Conversion/MOToPrimitives.h.inc"

} // namespace M::MGP

#endif // GENERICML_CONVERSION_MOTOPRIMITIVES_H
