//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_CONVERSION_PACKAGEVARIADICSANDSCALARS_H
#define GENERICML_GRAPHCOMPILER_CONVERSION_PACKAGEVARIADICSANDSCALARS_H

#include "mlir/Pass/Pass.h"

namespace M::MOGG {

#define GEN_PASS_REGISTRATION
#define GEN_PASS_DECL
#include "GraphCompiler/Conversion/PackageVariadicsAndScalars/PackageVariadicsAndScalars.h.inc"

} // namespace M::MOGG

#endif // GENERICML_GRAPHCOMPILER_CONVERSION_PACKAGEVARIADICSANDSCALARS_H
