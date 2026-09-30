//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_INTERFACES_MAPTOMOGGSTATE_H
#define GENERICML_GRAPHCOMPILER_INTERFACES_MAPTOMOGGSTATE_H

#include "GraphCompiler/MODialect/MODevices.h"
#include "GraphCompiler/MOGGDialect/Analysis/MojoRegistry.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLFunctionalExtras.h"

namespace M::MO {
class InvokeShapeFuncOp;
} // namespace M::MO

namespace M::MAP {

// Shared, per-kernel context threaded through the whole recursive lowering of
// one `map.iter.opaque` by MAPToMOGG (see `LowerableToMOGGInterface`).
// Rebuilt fresh (a new `MAPToMOGGState`) for every top-level kernel -- nothing
// here is meant to survive across kernels.
struct MAPToMOGGState {
  MAPToMOGGState(MOGG::MojoRegistry &registry, mlir::Operation *op)
      : registry(registry), builder(op) {}

  MOGG::MojoRegistry &registry;
  mlir::OpBuilder builder;

  // Whether an epilogue feeding a `_FusedComputeOutputTensor` argument may be
  // lowered as a compute lambda. Mirrors the `EmitMojo` option of the same
  // name.
  bool enableComputeLambdas = true;

  // Substitution for every value reachable from the kernel currently being
  // lowered: this kernel's own input operands and output block args -> the
  // `!mogg.tensor_` `IterOpaqueOp::lowerToMOGG` reconstructed for them (grown
  // by `IterEpilogueOp::lowerToMOGG` too -- an epilogue's own result resolves
  // exactly like a plain kernel output). A nested region (a foreach body's
  // captures/index, an epilogue body's params/captures) remaps its own
  // entries directly before lowering its body.
  mlir::IRMapping mapping;

  // Context per-op `lowerToMOGG` implementations need, set once by
  // `IterOpaqueOp::lowerToMOGG` before any nested op is lowered.
  DeviceRefAttr device;
  // Unique dynamic dims (in first-seen order) -> the `!mogg.dim_` block arg
  // reconstructed for each. Collected as null-valued entries before the
  // kernel's block args exist, then filled in once they do.
  llvm::MapVector<mlir::Attribute, mlir::BlockArgument> dimArgs;
  // The kernel's trailing `!mogg.context` block args. Exactly one for an
  // ordinary single-device kernel; one per device (in
  // `getUniqueDevicesFromKernel` order) for a distributed kernel whose execute
  // signature takes a `DeviceContextArray`. `ConformCallOperands` packs
  // the N-context case into a tuple-of-contexts.
  llvm::SmallVector<mlir::Value> contextArgs;

  // For a dynamic store-index reshape fused into an epilogue: the MAP
  // `map.index.reshape` op -> the two `!mogg.shape` block args (from-shape,
  // to-shape) `IterEpilogueOp::lowerToMOGG` threaded in as shape captures so
  // the store-index transform can be evaluated at runtime inside the functor.
  // Empty unless the epilogue body carries a dynamic reshape.
  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<mlir::Value>>
      reshapeShapeArgs;

  // Lowers `op` through its own `LowerableToMOGGInterface::lowerToMOGG`.
  mlir::LogicalResult lowerOp(mlir::Operation *op);
  // Lowers the ops of `block` in order via `lowerOp`, skipping any for which
  // `shouldLower` returns false. Order is guaranteed, so a predicate that
  // skips an op may rely on everything before it already being mapped.
  mlir::LogicalResult lowerRegionBody(
      mlir::Block &block,
      llvm::function_ref<bool(mlir::Operation *)> shouldLower =
          [](mlir::Operation *) { return true; });
};

/// Lowers a `mo.invoke_shape_func` into a `mogg._kernel`, and erases it.
mlir::LogicalResult lowerInvokeShapeFuncToMOGG(MO::InvokeShapeFuncOp op,
                                               MAPToMOGGState &state);

} // namespace M::MAP

#endif // GENERICML_GRAPHCOMPILER_INTERFACES_MAPTOMOGGSTATE_H
