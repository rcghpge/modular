//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MODialect/Analysis/ParametricTopologyAnalysis.h"
#include "GraphCompiler/MODialect/MODialect.h"
#include "GraphCompiler/MODialect/MOOps.h"
#include "KGEN/KGENDialect/KGENParameters.h"
#include "Support/AssertStream.h"

#include "mlir/Pass/AnalysisManager.h"

namespace M::MO {

ParametricTopologyAnalysis::ParametricTopologyAnalysis(mlir::Operation *root) {
  // If from is in another block, then all parents of to must also be users of
  // from, up to the last parent that is in the same block as from. When
  // doFirstLevel is set, we will create the dependency between from and to
  // (parametric dependencies are never present in SSA graphs so we need to do
  // it unconditionally).
  const auto applyDependencyToParentOps = [&](Operation *from, Operation *to,
                                              bool doAtLeastOnce) {
    // Parameters don't obey region dominance like SSA values. An operation can
    // use a parameter defined in a child region if that child region is not
    // held by a DeclInterface operation.
    if (to->getParentRegion()->isProperAncestor(from->getParentRegion()))
      return;

    const auto applyDependency = [&]() {
      doAtLeastOnce = true;
      // provider -> user
      metadata[from].parametricUsers.insert(to);
      // user <- provider
      metadata[to].parametricProviders.insert(from);
      to = to->getParentOp();
    };

    while (to && from->getBlock() != to->getBlock())
      applyDependency();

    // If from and to where not in the same block originally, then do the last
    // op. Otherwise, don't do anything (we would end up duplicating the SSA
    // graph in the metadata).
    if (to && doAtLeastOnce)
      applyDependency();
  };

  auto paramCache = KGEN::ParameterCollector::Analysis(root);
  for (auto &region : root->getRegions()) {
    KGEN::ParameterUseDefGraph useDefGraph(region);
    useDefGraph.calculate(paramCache);

    // For every nested use def graph (this is because an op that implements
    // DeclInterface will not appear in the root useDefGraph).
    llvm::SmallVector<KGEN::ParameterUseDefGraph *> workList = {&useDefGraph};
    while (!workList.empty()) {
      auto &nextUseDefGraph = *workList.pop_back_val();
      for (auto &[op, references] : nextUseDefGraph.opUses) {
        for (auto reference : references) {
          auto whereItIsDeclaredIt =
              nextUseDefGraph.decls.find(reference.getName());
          if (whereItIsDeclaredIt == nextUseDefGraph.decls.end())
            continue;
          auto whereItIsDeclared = whereItIsDeclaredIt->second;
          applyDependencyToParentOps(whereItIsDeclared.declOp, op, true);
        }
      }

      for (auto &[_, nestedUseDefGraph] : nextUseDefGraph.nestedScopes)
        workList.push_back(&nestedUseDefGraph);
    }
  }

  root->walk([&](Operation *op) {
    for (auto user : op->getUsers())
      applyDependencyToParentOps(op, user, false);
  });
}

llvm::SmallSetVector<mlir::Operation *, 4>
ParametricTopologyAnalysis::users(mlir::Operation *op) const {
  llvm::SmallSetVector<mlir::Operation *, 4> ret;
  forEachUser(op, [&](mlir::Operation *user) -> bool {
    ret.insert(user);
    return true;
  });

  return ret;
}

llvm::SmallSetVector<mlir::Operation *, 4>
ParametricTopologyAnalysis::providers(mlir::Operation *op) const {
  llvm::SmallSetVector<mlir::Operation *, 4> ret;
  forEachProvider(op, [&](mlir::Operation *provider) -> bool {
    ret.insert(provider);
    return true;
  });

  return ret;
}

/// WARNING: Blatantly copied from
/// https://mlir.llvm.org/doxygen/TopologicalSortUtils_8cpp_source.html#l00107
/// This extends it by using the additional dependencies coming from the
/// parametric world.

/// Return `true` if the given operation is ready to be scheduled.
bool inline ParametricTopologyAnalysis::isOpReady(
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
    bool allOf = true;
    forEachProvider(nestedOp, [&](auto provider) {
      if (!allOf)
        return false;
      allOf &= isReady(provider);
      return true;
    });

    return allOf ? WalkResult::advance() : WalkResult::interrupt();
  });

  return !readyToSchedule.wasInterrupted();
}

void ParametricTopologyAnalysis::computeTopologicalSortingImpl(
    mlir::MutableArrayRef<mlir::Operation *> ops) {
  // The set of operations that have not yet been scheduled.
  DenseSet<Operation *> unscheduledOps;

  // Mark all operations as unscheduled.
  for (Operation *op : ops)
    unscheduledOps.insert(op);

  unsigned nextScheduledOp = 0;

  while (!unscheduledOps.empty()) {
    [[maybe_unused]] bool scheduledAtLeastOnce = false;

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

    ASSERT_STREAM(scheduledAtLeastOnce,
                  << "No op was scheduled, potential cycle in the graph");
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
    [[maybe_unused]] bool scheduledAtLeastOnce = false;

    // Loop over the ops that are not sorted yet, try to find the ones "ready",
    // i.e. the ones for which there aren't any operand produced by an op in the
    // set, and "schedule" it (swap it with the op at `nextScheduledOp`).
    for (Operation &op :
         llvm::make_early_inc_range(llvm::make_range(nextScheduledOp, end))) {
      if (!isOpReady(&op, unscheduledOps))
        continue;

      unscheduledOps.erase(&op);
      scheduledAtLeastOnce = true;
      // Move the iterator forward if we schedule the operation at the front.
      if (&op == &*nextScheduledOp)
        ++nextScheduledOp;

      // NOTE these ops are _required_ to be at the end of their parent block,
      // so we must not reorder them.
      if (op.hasTrait<OpTrait::IsTerminator>())
        continue;

      // Schedule the operation by moving it to the start.
      op.moveBefore(block, nextScheduledOp);
    }

    ASSERT_STREAM(scheduledAtLeastOnce,
                  << "No op was scheduled, potential cycle in the graph");
  }
}

void ParametricTopologyAnalysis::topologicallySortAllOpsWithin(Region *region) {
  auto regionKindInterface =
      dyn_cast<mlir::RegionKindInterface>(region->getParentOp());

  // If we are unable to prove the region is graph region, don't sort since
  // original ordering may matter.
  if (!regionKindInterface)
    return;

  if (regionKindInterface.getRegionKind(region->getRegionNumber()) !=
      RegionKind::Graph)
    return;

  for (Block &block : *region) {
    topologicallySortFirstLevelOpsWithin(&block);
    for (Operation &op : block)
      for (Region &nestedRegion : op.getRegions())
        topologicallySortAllOpsWithin(&nestedRegion);
  }
}

// Return vector of op and any users, users of users, and so forth.
//
// It is assumed all operations are rooted on `op` and it is guaranteed we
// will return the subgraph in a topological order.
SmallVector<Operation *> getRootedComponents(Operation *op) {
  SmallVector<Operation *> fringe;
  fringe.push_back(op);

  DenseSet<Operation *> enqueued;
  enqueued.insert(op);

  // Strategy is to DFS, and then sort later using built-in util.
  SmallVector<Operation *> answer;
  while (!fringe.empty()) {
    Operation *cur = fringe.back();
    fringe.pop_back();
    if (cur == nullptr)
      continue;
    answer.push_back(cur);

    for (Operation *user : cur->getUsers()) {
      if (enqueued.insert(user).second)
        fringe.push_back(user);
    }

    if (cur == op)
      continue;
    for (auto operand : cur->getOperands()) {
      Operation *definingOp = operand.getDefiningOp();
      if (enqueued.insert(definingOp).second)
        fringe.push_back(definingOp);
    }
  }

  MO::ParametricTopologyAnalysis analysis(op->getParentOp());
  analysis.computeTopologicalSorting(answer);
  return answer;
}

} // namespace M::MO
