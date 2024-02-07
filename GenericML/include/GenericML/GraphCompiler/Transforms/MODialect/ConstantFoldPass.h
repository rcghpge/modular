//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_TRANSFORMS_MODIALECT_CONSTANTFOLDPASS_H
#define GENERICML_GRAPHCOMPILER_TRANSFORMS_MODIALECT_CONSTANTFOLDPASS_H

#include "GenericML/GraphCompiler/Passes.h"
#include "mlir/Pass/Pass.h"

namespace M::MO {

/// Creates a constant folding pass, which registers foreign buffer interfaces
/// in a namespace prefixed by `compiledFrameworkLabel`.
std::unique_ptr<mlir::Pass>
createConstantFoldPass(std::string_view compiledFrameworkLabel,
                       const MoToMgpOptions &opts);

} // namespace M::MO

#endif // GENERICML_GRAPHCOMPILER_TRANSFORMS_MODIALECT_CONSTANTFOLDPASS_H
