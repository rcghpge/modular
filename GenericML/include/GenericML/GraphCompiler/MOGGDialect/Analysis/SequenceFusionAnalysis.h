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
#include "KGEN/KGENDialect/KGENOps.h"

#include <functional>

namespace M::MOGG {

/// Eligibility is defined as ops that should be considered fusible.
using EligibilityFunction = const std::function<bool(mlir::Operation *)> &;

/// Comparability means that the operations are eligible but we should still
/// see if they can be fused together (ex: same device).
using CompatibilityFunction =
    const std::function<bool(mlir::Operation *, mlir::Operation *)> &;

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
      KGEN::ParamConstantOp

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
