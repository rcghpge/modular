//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Lowering helpers shared by the GPU target lowerings (NVPTX, AMDGPU, Metal).
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_KGENTOLLVM_TARGET_GPUCOMMON_H
#define KGEN_KGENTOLLVM_TARGET_GPUCOMMON_H

#include "mlir/IR/Types.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
class Operation;
} // namespace mlir

namespace M::KGEN {

/// If `type` is a non-parameter-pack aggregate that contains a pointer, returns
/// a pointer to it (its memory form); otherwise a null type.
mlir::Type lowerAggregateArgToMemory(mlir::Type type);

/// Returns failure, emitting a diagnostic on `op`, if `op` performs heap
/// allocation or deallocation (a `pop.aligned_alloc`/`pop.aligned_free` op or a
/// non-stack memory blob), which GPU targets do not support.
mlir::LogicalResult verifyNoGPUHeapAllocation(mlir::Operation *op);

} // namespace M::KGEN

#endif // KGEN_KGENTOLLVM_TARGET_GPUCOMMON_H
