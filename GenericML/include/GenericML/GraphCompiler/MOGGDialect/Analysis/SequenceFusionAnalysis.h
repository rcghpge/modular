//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MOGGDIALECT_SEQUENCEFUSIONANALYSIS_H
#define GENERICML_GRAPHCOMPILER_MOGGDIALECT_SEQUENCEFUSIONANALYSIS_H

#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseSet.h"

#include "GenericML/GraphCompiler/Analysis/MODialect/ParametricTopologyAnalysis.h"
#include "GenericML/GraphCompiler/MODialect/MOOps.h"
#include "GenericML/GraphCompiler/MOSHDialect/MOSHOps.h"
#include "KGEN/KGENDialect/KGENOps.h"

#include <functional>

namespace M::MOGG {

/// Eligibility is defined as ops that should be considered fusible.
using EligibilityFunction = const std::function<bool(mlir::Operation *)> &;

/// Comparability means that the operations are eligible but we should still
/// see if they can be fused together (ex: same device).
using CompatibilityFunction =
    const std::function<bool(mlir::Operation *, mlir::Operation *)> &;

enum class Direction { Up, Down };

/// The returned set contains all operations that are eligible, compatible with
/// one another and all paths to them from the root are void of non eligible
/// ops. A SetVector is returned for determinism. Will not traverse across
/// blocks.
/// Note: The compatibility function is used on the edges during the traversal.
/// The operands must be interpreted as <from, to>.
template <Direction direction, bool allowDuplication = false>
mlir::SetVector<mlir::Operation *>
floodFillFrom(const mlir::SmallVector<mlir::Operation *> &roots,
              EligibilityFunction isEligible,
              CompatibilityFunction areCompatible,
              MO::ParametricTopologyAnalysis &completeDataflowGraph) {
  mlir::SetVector<mlir::Operation *> visitedAsEligible;
  mlir::SetVector<mlir::Operation *> visitedAsNonEligible;

  mlir::Block *block = roots.front()->getBlock();
  llvm::SmallVector<std::tuple<mlir::Operation *, Operation *, bool>> workList;
  for (mlir::Operation *op : roots)
    workList.push_back({op, nullptr, true});

  while (!workList.empty()) {
    auto [op, from, asEligible] = workList.pop_back_val();
    if (op->getBlock() != block)
      continue;

    asEligible &= isEligible(op);
    if (asEligible && from) {
      if constexpr (direction == Direction::Down)
        asEligible &= areCompatible(from, op);
      else
        asEligible &= areCompatible(op, from);
    }

    if (asEligible) {
      // If the current op is eligible, then ensure it was not invalidated or
      // already visited.
      if (visitedAsNonEligible.contains(op))
        continue;

      if (!visitedAsEligible.insert(op))
        continue;
    } else if (allowDuplication || !visitedAsNonEligible.insert(op)) {
      // If the current op is not eligible, ensure it was not already visited
      // and remove it from eligible ops if it was thought to be eligible.
      continue;
    }

    if constexpr (direction == Direction::Down) {
      for (auto user : completeDataflowGraph.users(op))
        workList.push_back({user, op, asEligible});
    } else {
      for (auto provider : completeDataflowGraph.providers(op))
        workList.push_back({provider, op, asEligible});
    }
  }

  visitedAsEligible.set_subtract(visitedAsNonEligible);

  return visitedAsEligible;
}

/// NOTE: Passes that leverage this analysis have to emit deterministic IR
/// meaning two invocations of the same pass over the same input should output
/// the same IR. Therefore, all functions in this analysis guarantee that the
/// returned values are always deterministic (mainly same order).

/// Returns subgraphs of operations that can be fused together while maintaining
/// the following guarantee: Any eligible ops that has a path in G or reverse(G)
/// to another eligible op that does not go through a non eligible op will be
/// fused. Will not fuse across blocks.
llvm::SmallVector<llvm::SmallVector<mlir::Operation *>>
floodFillSequenceFusionAnalysis(
    MO::GraphOp graphOp, EligibilityFunction isEligible,
    CompatibilityFunction areCompatible,
    MO::ParametricTopologyAnalysis &completeDataflowGraph);

/// Operations that should not be considered when preserving parallelism. This
/// means if an additional dependency is created between any of these ops and
/// any other op, it is still parallelism preserving. Rationale: They are not
/// fusible operations but they are still very low in compute which means
/// serializing them is not a big deal.
#define EXCLUSION_LIST                                                         \
  MO::ConstantOp, MO::Shape::ToTensorOp, MO::IndexToTensorOp, MO::RebindOp,    \
      MO::ReshapeOp, MO::Static::ReshapeOp, KGEN::ParamDeclareOp,              \
      MOSH::ParamToValueOp

/// If preserveParallelism is set to true, no additional dependency between non
/// eligible op is created. Otherwise, it fuses anything anything as long as no
/// cycle is created. Will not fuse across blocks.
llvm::SmallVector<llvm::SmallVector<mlir::Operation *>>
transitiveDependencySequenceFusionAnalysis(
    MO::GraphOp graphOp, EligibilityFunction isEligible,
    CompatibilityFunction areCompatible,
    MO::ParametricTopologyAnalysis &completeDataflowGraph,
    bool preserveParallelism);

} // namespace M::MOGG

#endif // GENERICML_GRAPHCOMPILER_MOGGDIALECT_SEQUENCEFUSIONANALYSIS_H
