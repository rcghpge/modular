//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_TRANSFORMS_MODIALECT_CONSTANTFOLDPASS_H
#define GENERICML_GRAPHCOMPILER_TRANSFORMS_MODIALECT_CONSTANTFOLDPASS_H

#include "mlir/Pass/Pass.h"
#include "llvm/ADT/ArrayRef.h"

#include <optional>
#include <string>

namespace M::MO {

/// Creates a constant folding pass, which registers foreign buffer interfaces
/// in a namespace prefixed by `compiledFrameworkLabel`.
///
/// Only `cacheDir` and `kernelLibraryPaths` affect correctness of the
/// constant-fold sub-compilation; all other options are intentionally excluded.
std::unique_ptr<mlir::Pass>
createConstantFoldPass(std::string_view compiledFrameworkLabel,
                       std::optional<std::string> cacheDir,
                       llvm::ArrayRef<std::string> kernelLibraryPaths);

} // namespace M::MO

#endif // GENERICML_GRAPHCOMPILER_TRANSFORMS_MODIALECT_CONSTANTFOLDPASS_H
