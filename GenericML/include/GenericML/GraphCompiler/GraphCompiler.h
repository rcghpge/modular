//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_MO_GRAPHCOMPILER_H
#define GENERICML_MO_GRAPHCOMPILER_H

#include <mlir/IR/DialectRegistry.h>

namespace M::MO {

void registerDialectsAndPasses(mlir::DialectRegistry &registry);

}

#endif // GENERICML_MO_GRAPHCOMPILER_H
