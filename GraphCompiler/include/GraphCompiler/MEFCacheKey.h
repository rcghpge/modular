//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_MEFCACHEKEY_H
#define GENERICML_MEFCACHEKEY_H

#include "Support/Context.h"
#include "Support/ErrorOr.h"

#include "llvm/Support/BLAKE3.h"

#include <string>

namespace mlir {
class MLIRContext;
class ModuleOp;
} // namespace mlir

namespace M {

namespace MOF {
struct MOFLoweringOptions;
}
namespace MO {
struct GCPipelineOptions;
}

/// Incremental hasher for MEF cache keys. Holds a precomputed hash state
/// for session-level hash dependencies.
///
/// The MEF cache key depends on
/// - `MOF::MOFLoweringOptions`
/// - The GC pass pipeline, version, and options
/// - KGEN compilation context (Mojo compile-time defines)
/// - All op library MLIR
///
/// It _does not_ depend on location info or specific symbol names.
class MEFCacheKeyHasher {
public:
  static ErrorOr<MEFCacheKeyHasher>
  build(mlir::MLIRContext &mlirCtx, MOF::MOFLoweringOptions &mofLoweringOptions,
        const MO::GCPipelineOptions &gcPipelineOptions, ContextRef maxContext);

  ErrorOr<std::string> keyForModule(mlir::ModuleOp module) const;

private:
  MEFCacheKeyHasher() = default;
  llvm::BLAKE3 prefixHasher;
};

ErrorOr<std::string> computeMEFCacheKey(
    mlir::ModuleOp module, MOF::MOFLoweringOptions &mofLoweringOptions,
    const MO::GCPipelineOptions &gcPipelineOptions, ContextRef maxContext);

} // namespace M

#endif // GENERICML_MEFCACHEKEY_H
