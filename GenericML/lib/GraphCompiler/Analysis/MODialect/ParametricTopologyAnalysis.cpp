//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/Analysis/MODialect/ParametricTopologyAnalysis.h"
#include "GenericML/GraphCompiler/MODialect/MOOps.h"
#include "KGEN/KGENDialect/KGENParameters.h"

#include "mlir/Pass/AnalysisManager.h"

namespace M::MO {

ParametricTopologyAnalysis::ParametricTopologyAnalysis(mlir::Operation *root) {
  auto paramCache = KGEN::ParameterCollector::Analysis(root);
  for (auto &region : root->getRegions()) {
    KGEN::ParameterUseDefGraph useDefGraph(region);
    useDefGraph.calculate(paramCache);

    auto &rootBlocks = region.getBlocks();
    const auto isARootBlock = [&](mlir::Block *block) {
      for (auto &rootBlock : rootBlocks) {
        if (&rootBlock == block)
          return true;
      }

      return false;
    };

    // For every nested use def graph (this is because an op that implements
    // DeclInterface will not appear in the root useDefGraph).
    llvm::SmallVector<KGEN::ParameterUseDefGraph *> workList = {&useDefGraph};
    while (!workList.empty()) {
      auto &nextUseDefGraph = *workList.pop_back_val();
      for (auto &[op, references] : nextUseDefGraph.opUses) {
        for (auto reference : references) {
          auto whereItIsDeclared =
              nextUseDefGraph.decls.at(reference.getName());
          mlir::Operation *parentOp = op;
          do {
            // Decl <- Use
            metadata[parentOp].parametricProviders.insert(
                whereItIsDeclared.declOp);
            // Decl -> Use
            metadata[whereItIsDeclared.declOp].parametricUsers.insert(parentOp);
            /// ASSUMPTION: The block where the parameter is defined
            /// encompasses the block where it is used. This is the same rule
            /// we have for SSA values.
            if (parentOp->getBlock() == whereItIsDeclared.declOp->getBlock() ||
                isARootBlock(parentOp->getBlock()))
              break;

            parentOp = parentOp->getParentOp();
          } while (parentOp); // parent op of `mo.graph` is nullptr
        }
      }

      for (auto &[_, nestedUseDefGraph] : nextUseDefGraph.nestedScopes)
        workList.push_back(&nestedUseDefGraph);
    }
  }
}

llvm::SmallSetVector<mlir::Operation *, 4>
ParametricTopologyAnalysis::users(mlir::Operation *op) const {
  auto dataflowUsers = op->getUsers();
  llvm::SmallSetVector<mlir::Operation *, 4> ret(dataflowUsers.begin(),
                                                 dataflowUsers.end());

  auto it = metadata.find(op);
  if (it != metadata.end()) {
    ret.insert(it->second.parametricUsers.begin(),
               it->second.parametricUsers.end());
  }

  return ret;
}

llvm::SmallSetVector<mlir::Operation *, 4>
ParametricTopologyAnalysis::providers(mlir::Operation *op) const {
  llvm::SmallSetVector<mlir::Operation *, 4> ret;

  auto it = metadata.find(op);
  if (it != metadata.end()) {
    ret.insert(it->second.parametricProviders.begin(),
               it->second.parametricProviders.end());
  }

  for (mlir::Value operand : op->getOperands()) {
    auto provider = operand.getDefiningOp();
    if (provider)
      ret.insert(provider);
  }

  return ret;
}

/// WARNING: Blatantly copied from
/// https://mlir.llvm.org/doxygen/TopologicalSortUtils_8cpp_source.html#l00107
/// This extends it by using the additional dependencies coming from the
/// parametric world.

/// Return `true` if the given operation is ready to be scheduled.
bool ParametricTopologyAnalysis::isOpReady(
    Operation *op, DenseSet<Operation *> &unscheduledOps) {
  // An operation is ready to be scheduled if all its operands are ready. An
  // operation is ready if:
  const auto isReady = [&](Operation *parent) {
    // it is not defined by an unscheduled op (and also not nested within an
    // unscheduled op).
    do {
      // Stop traversal when op under examination is reached.
      if (parent == op)
        return true;
      if (unscheduledOps.contains(parent))
        return false;
    } while ((parent = parent->getParentOp()));
    // No unscheduled op found.
    return true;
  };

  // An operation is recursively ready to be scheduled of it and its nested
  // operations are ready.
  WalkResult readyToSchedule = op->walk([&](Operation *nestedOp) {
    return llvm::all_of(
               providers(nestedOp),
               [&](mlir::Operation *provider) { return isReady(provider); })
               ? WalkResult::advance()
               : WalkResult::interrupt();
  });
  return !readyToSchedule.wasInterrupted();
}

void ParametricTopologyAnalysis::computeTopologicalSortingImpl(
    mlir::MutableArrayRef<mlir::Operation *> ops) {
  if (ops.empty())
    return;

  // The set of operations that have not yet been scheduled.
  DenseSet<Operation *> unscheduledOps;

  // Mark all operations as unscheduled.
  for (Operation *op : ops)
    unscheduledOps.insert(op);

  unsigned nextScheduledOp = 0;

  while (!unscheduledOps.empty()) {
    bool scheduledAtLeastOnce = false;

    // Loop over the ops that are not sorted yet, try to find the ones "ready",
    // i.e. the ones for which there aren't any operand produced by an op in the
    // set, and "schedule" it (swap it with the op at `nextScheduledOp`).
    for (unsigned i = nextScheduledOp; i < ops.size(); ++i) {
      if (!isOpReady(ops[i], unscheduledOps))
        continue;

      // Schedule the operation by moving it to the start.
      unscheduledOps.erase(ops[i]);
      std::swap(ops[i], ops[nextScheduledOp]);
      scheduledAtLeastOnce = true;
      ++nextScheduledOp;
    }

    // If no operations were scheduled, just schedule the first op and continue.
    if (!scheduledAtLeastOnce)
      unscheduledOps.erase(ops[nextScheduledOp++]);
  }
}

void ParametricTopologyAnalysis::computeTopologicalSorting(
    mlir::MutableArrayRef<mlir::Operation *> ops) {
  computeTopologicalSortingImpl(ops);
}

// NOTE this isn't refactored with the other impl because the differences are
// trivial to lift out, and a refactor might end up becoming over-abstraction.
void ParametricTopologyAnalysis::topologicallySortFirstLevelOpsWithin(
    mlir::Block *block) {
  if (block->empty())
    return;

  // The set of operations that have not yet been scheduled.
  DenseSet<Operation *> unscheduledOps;

  // Mark all operations as unscheduled.
  for (Operation &op : *block)
    unscheduledOps.insert(&op);

  Block::iterator nextScheduledOp = block->begin();
  Block::iterator end = block->end();

  while (!unscheduledOps.empty()) {
    bool scheduledAtLeastOnce = false;

    // Loop over the ops that are not sorted yet, try to find the ones "ready",
    // i.e. the ones for which there aren't any operand produced by an op in the
    // set, and "schedule" it (swap it with the op at `nextScheduledOp`).
    for (Operation &op :
         llvm::make_early_inc_range(llvm::make_range(nextScheduledOp, end))) {
      if (!isOpReady(&op, unscheduledOps))
        continue;

      // NOTE these ops are _required_ to be at the end of their parent block,
      // so we must not reorder them. We can afford to be a little MO specific
      // here since the analysis is under MODialect.
      if (isa<MO::OutputOp, MO::YieldOp>(op))
        continue;

      // Schedule the operation by moving it to the start.
      unscheduledOps.erase(&op);
      op.moveBefore(block, nextScheduledOp);
      scheduledAtLeastOnce = true;
      // Move the iterator forward if we schedule the operation at the front.
      if (&op == &*nextScheduledOp)
        ++nextScheduledOp;
    }

    // If no operations were scheduled, give up and advance the iterator.
    if (!scheduledAtLeastOnce) {
      unscheduledOps.erase(&*nextScheduledOp);
      ++nextScheduledOp;
    }
  }
}

void ParametricTopologyAnalysis::topologicallySortAllOpsWithin(Region *region) {
  for (Block &block : *region) {
    topologicallySortFirstLevelOpsWithin(&block);
    for (Operation &op : block) {
      for (Region &nestedRegion : op.getRegions())
        topologicallySortAllOpsWithin(&nestedRegion);
    }
  }
}

} // namespace M::MO
