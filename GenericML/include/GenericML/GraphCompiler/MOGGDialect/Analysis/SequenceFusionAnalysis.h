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
/// see if they can be fused together (ex: same device, fusion interfaces, etc).
using CompatibilityFunction =
    const std::function<bool(mlir::Operation *, mlir::Operation *)> &;

// If an operation matters when preserving parallelism (computationally
// significant).
using ComputationallySignificantFunction =
    const std::function<bool(mlir::Operation *)> &;

using FusionSubGraphs = llvm::SmallVector<llvm::SmallVector<mlir::Operation *>>;

/// NOTE: Passes that leverage this analysis have to emit deterministic IR
/// meaning two invocations of the same pass over the same input should output
/// the same IR. Therefore, all functions in this analysis guarantee that the
/// returned values are always deterministic (mainly same order).

/// Returns subgraphs of operations that can be fused together while maintaining
/// the following guarantee: Any eligible ops that has a path in G or reverse(G)
/// to another eligible op that does not go through a non eligible op will be
/// fused. Will not fuse across blocks.
FusionSubGraphs floodFillSequenceFusionAnalysis(
    Region &region, MO::ParametricTopologyAnalysis &completeDataflowGraph,
    EligibilityFunction isEligible, CompatibilityFunction areCompatible);

/// If preserveParallelism is set to true, no additional dependency between
/// computationally significant ops is created. Otherwise, it fuses anything
/// as long as no cycle is created. Will not fuse across blocks.
FusionSubGraphs transitiveDependencySequenceFusionAnalysis(
    Region &region, MO::ParametricTopologyAnalysis &completeDataflowGraph,
    EligibilityFunction isEligible, CompatibilityFunction areCompatible,
    ComputationallySignificantFunction computationallySignificant,
    bool preserveParallelism);

/// Fuses in compatible operands that would be DCE-ed after fusion. In other
/// words, an operation is fused down if it can fuse with all its users. If
/// allowDuplication is set to false, operations with more than one user can't
/// be fused down.
FusionSubGraphs sourceDestinationSequenceFusionAnalysis(
    Region &region, MO::ParametricTopologyAnalysis &completeDataflowGraph,
    EligibilityFunction isEligible, CompatibilityFunction areCompatible,
    bool allowDuplication);

} // namespace M::MOGG

#endif // GENERICML_GRAPHCOMPILER_MOGGDIALECT_SEQUENCEFUSIONANALYSIS_H
