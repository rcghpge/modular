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
  mlir::Value contextArg;

  // Lowers `op` through its own `LowerableToMOGGInterface::lowerToMOGG`.
  mlir::LogicalResult lowerOp(mlir::Operation *op);
  // Lowers every op in `block` in order via `lowerOp`.
  mlir::LogicalResult lowerRegionBody(mlir::Block &block);
};

} // namespace M::MAP

#endif // GENERICML_GRAPHCOMPILER_INTERFACES_MAPTOMOGGSTATE_H
