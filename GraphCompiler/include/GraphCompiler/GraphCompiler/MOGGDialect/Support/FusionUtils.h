//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
#ifndef GENERICML_GRAPH_COMPILER_MOGGSUPPORT_FUSIONUTILS_H
#define GENERICML_GRAPH_COMPILER_MOGGSUPPORT_FUSIONUTILS_H

#include "GraphCompiler/GraphCompiler/MOGGDialect/MOGGAttrs.h"
#include "GraphCompiler/GraphCompiler/MOGGDialect/MOGGOps.h"
#include "Support/AssertStream.h"

using namespace M;

namespace M::MOGG {

//===----------------------------------------------------------------------===//
// FusionInfo helper functions
//===----------------------------------------------------------------------===//

FusionLeafAttr createFusionLeaf(Operation *op);

template <typename GroupAttr>
FusionInfoAttr createFusionInfo(MLIRContext *ctx, ArrayRef<Operation *> ops,
                                ArrayRef<uint32_t> operandIndices = {}) {

  auto getFusionInfo = [](Operation *op) -> Attribute {
    if (auto info = op->getAttrOfType<FusionInfoAttr>("fusionInfo"))
      return info.getRoot();
    return createFusionLeaf(op);
  };

  SmallVector<Attribute> children;
  SmallVector<uint32_t> mergedIndices(operandIndices.begin(),
                                      operandIndices.end());
  for (auto *op : ops) {
    Attribute child = getFusionInfo(op);
    // Flatten fusion info in cases where multiple same type fusions were
    // applied sequentially.
    // E.g. Elementwise(a, Elementwise(b, c)) -> Elementwise(a, b, c)
    if (auto sameGroup = dyn_cast<GroupAttr>(child)) {
      children.append(sameGroup.getChildren().begin(),
                      sameGroup.getChildren().end());
      if constexpr (std::is_same_v<GroupAttr, FusionEpilogueAttr> ||
                    std::is_same_v<GroupAttr, FusionPrologueAttr>) {
        auto inner = sameGroup.getOperandIndices();
        mergedIndices.append(inner.begin(), inner.end());
      }
      continue;
    }
    children.push_back(child);
  }

  if constexpr (std::is_same_v<GroupAttr, FusionEpilogueAttr> ||
                std::is_same_v<GroupAttr, FusionPrologueAttr>)
    return FusionInfoAttr::get(ctx,
                               GroupAttr::get(ctx, children, mergedIndices));
  else
    return FusionInfoAttr::get(ctx, GroupAttr::get(ctx, children));
}

/// Fuse a topologically-sorted list of GenericCompiledRegionOps into a single
/// region.  All uses of the old regions' results are replaced with the fused
/// region's results.  The caller must erase the old regions afterward.
/// Returns the newly created fused region.
GenericCompiledRegionOp
fuseAndReplaceCompiledRegions(bool allowDuplication,
                              ArrayRef<Operation *> toFuse);

/// Fuse small constant inputs to `kernel` into the kernel, and erase the
/// original constant op if it has no remaining users.
///
/// Example:
///   %const = mo.constant ...
///   %res   = mogg.kernel(%const, ...) {
///     ...
///   }
///
/// ===>
///
///   %res = mogg.kernel(...) {
///     %const = mo.constant ...
///     ...
///   }
void fuseConstant(MOGG::KernelOp kernel);

/// Callback type to intercept inlining on a specific operation.
/// Return true if the operation is handled by the callback, false otherwise.
///
/// Parameters:
///   OpBuilder &: the builder being used
///   IRMapping &: the current IRMapping for the callback to lookup/update
///   Operation *: the operation being inlined
///   GenericCompiledRegionOp src: parent of the operation being inlined
///   GenericCompiledRegionOp dst: region that the operation is inlining into
using fuseRegionInlineCallBackT = llvm::function_ref<bool(
    OpBuilder &, mlir::IRMapping &, Operation *, MOGG::GenericCompiledRegionOp,
    MOGG::GenericCompiledRegionOp)>;

// Aliased constant (introduced in MAXPLAT-70) will not be lifted to init, which
// might reduce the performance gains from certain optimization. We probably
// need to revisit it when the aliased constant is handled correctly.
bool dependsOnNoAliasConstantLikesOnly(mlir::Operation *operation);

/// Checks if `start` depends on `target` transitively.
/// Eg
/// ```
/// %1 = foo(%0)
/// %2 = bar(%1)
/// ```
/// `valueDependsTransitivelyOn(%2, %0) == true`.
/// Also returns true if start == target.
bool valueDependsTransitivelyOn(Value start, Value target);

/// Checks if `start` depends on `target` transitively.
/// Eg
/// ```
/// %1 = foo(%0)
/// %2 = bar(%1)
/// ```
/// `valueDependsTransitivelyOn(%2, foo) == true`.
/// Same as `llvm::any_of(target->getResults(), [](Value targetResult) { return
/// valueDependsTransitivelyOn(start, targetResult)}; )`.
bool valueDependsTransitivelyOn(Value start, mlir::Operation *target);

// As we transition from output to compute lambda, this represent currently the
// 3 different kind of fused tensor types that we can have when build a lambda
// and its corresponding fuser tensor object in the graph compiler.
enum class KernelOutputFusionLambdaType {
  Output, // Classic output fusion lambda with a store (kernel operand is
          // FusedOutputTensor).
  OutputLegacyForCompute, // Classic output fusion lambda with a store for
                          // legacy support of mixed-precision (kernel operand
                          // is _FusedComputeOutputTensor).
  Compute,     // New compute fusion lambda with a return (kernel operand is
               // _FusedComputeOutputTensor).
  ComputeTile, // Tile-based compute fusion lambda with a TileTensor return
               // (kernel operand is _FusedComputeOutputTileTensor).
};

// Returns the FusedTensor kind to be emitted by the Graph Compiler for
// `fusedOp`. Or returns an error if the kernel is not found / the op is
// invalid.
ErrorOr<KernelOutputFusionLambdaType> getKernelOutputFusionLambdaType(
    MojoRegistry &registry, MOGG::TensorFusedOp fusedOp,
    bool enableComputeLambdas, mlir::ModuleOp moduleOp);

} // namespace M::MOGG

#endif // GENERICML_GRAPH_COMPILER_MOGGSUPPORT_FUSIONUTILS_H
