//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "GraphCompiler/GraphCompiler/MODialect/MOOps.h"
#include "mlir/IR/PatternMatch.h"

namespace M::MO {

class DeviceInfoMapping;

/// Cache keyed on `(inputType, kgenParams)` that maps to the packed-weight
/// output `TensorType` produced by a matmul packing shape function. Shared
/// across `MatmulConstBPattern` instances within a single pass run so the
/// shape function is lowered at most once per unique cache key.
class PackParamCache {
public:
  void put(MO::TensorType inputTy, mlir::DictionaryAttr kgenParams,
           MO::TensorType outputTy);

  /// Returns the cached packed weight type, or {} on miss.
  MO::TensorType get(MO::TensorType inputTy, mlir::DictionaryAttr kgenParams);

private:
  llvm::DenseMap<MO::TensorType,
                 llvm::DenseMap<mlir::DictionaryAttr, MO::TensorType>>
      cache;
};

/// Options for MultiHeadAttention pattern matching.
struct MHAOptions {
  bool enable = false;
  bool allowUnsafeBCast = false;
  bool allowInternalReshape = false;
  bool debug = false;
};

/// Imperatively fuses an `mo.allreduce.sum` chain followed by add +
/// rms_norm + quantize_fp8 into a single op. Walks downward from the seed
/// op; no-op if the chain does not match.
void applyFuseAllReduceAddRMSNormQuantFP8(MO::AllReduceSumOp allReduceOp,
                                          M::DeviceRefAttr targetDeviceRef);

/// Cross-boundary fuser for `mo.parallel { mo.bundled.allreduce.sum }`
/// followed by per-device add + rms_norm + quantize_fp8 chains outside
/// the parallel.
void applyFuseParallelBundledAllReduceAddRMSNormQuantFP8(
    MO::BundledAllReduceSumOp bundledAllReduceOp);

/// Adds the main fusion rewrite patterns (LayerNorm, RMSNorm fusions,
/// Matmul folds, RoPE, Conv-const, attention, etc.).
void populateFusionPatterns(mlir::RewritePatternSet &patterns,
                            PackParamCache &cache,
                            M::DeviceRefAttr targetDeviceRef,
                            const MHAOptions &mhaOpts);

/// Adds `MatmulAddPattern`. Gated separately because it is only enabled on
/// SM100-class GPUs by the driver pass.
void populateFusionMatmulAddPatterns(const DeviceInfoMapping &deviceInfoMapping,
                                     mlir::RewritePatternSet &patterns);

/// Adds `MegaFFNNvfp4Pattern`, which fuses the two-op NVFP4 MoE FFN composite
/// chain into a single `mo.composite.mega_ffn_nvfp4`. The pattern arch-gates
/// itself to SM100-class GPUs via `deviceInfoMapping`; the driver pass
/// additionally guards it behind the `mega-ffn-enable` pass option.
void populateFusionMegaFFNPatterns(const DeviceInfoMapping &deviceInfoMapping,
                                   mlir::RewritePatternSet &patterns);

} // namespace M::MO
