//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/MOGGDialect/Analysis/SequenceFusionAnalysis.h"

#include "llvm-project/llvm/include/llvm/ADT/SetOperations.h"

namespace M::MOGG {

template <Direction direction>
mlir::SetVector<mlir::Operation *>
floodFillFrom(const mlir::SmallVector<mlir::Operation *> &roots,
              EligibilityFunction isEligible,
              CompatibilityFunction areCompatible,
              MO::ParametricTopologyAnalysis &completeDataflowGraph) {
  mlir::SetVector<mlir::Operation *> visitedAsEligible;
  mlir::SetVector<mlir::Operation *> visitedAsNonEligible;

  mlir::Block *block = roots.front()->getBlock();
  llvm::SmallVector<std::tuple<mlir::Operation *, bool>> workList;
  for (mlir::Operation *op : roots)
    workList.push_back({op, true});

  while (!workList.empty()) {
    auto [op, asEligible] = workList.pop_back_val();
    if (op->getBlock() != block)
      continue;

    asEligible &= isEligible(op) && areCompatible(op, roots.front());

    if (asEligible) {
      // If the current op is eligible, then ensure it was not invalidated or
      // already visited.
      if (visitedAsNonEligible.contains(op))
        continue;

      if (!visitedAsEligible.insert(op))
        continue;
    } else if (!visitedAsNonEligible.insert(op)) {
      // If the current op is not eligible, ensure it was not already visited
      // and remove it from eligible ops if it was thought to be eligible.
      continue;
    }

    if constexpr (direction == Direction::Down) {
      for (auto user : completeDataflowGraph.users(op))
        workList.push_back({user, asEligible});
    } else {
      for (auto provider : completeDataflowGraph.providers(op))
        workList.push_back({provider, asEligible});
    }
  }

  visitedAsEligible.set_subtract(visitedAsNonEligible);

  return visitedAsEligible;
};

llvm::SmallVector<llvm::SmallVector<mlir::Operation *>>
floodFillSequenceFusionAnalysis(
    MO::GraphOp graphOp, EligibilityFunction isEligible,
    CompatibilityFunction areCompatible,
    MO::ParametricTopologyAnalysis &completeDataflowGraph) {

  llvm::SmallVector<llvm::SmallVector<mlir::Operation *>> subgraphs;
  // For every block in the graph (recursively).
  graphOp.walk([&](Block *block) {
    mlir::DenseSet<mlir::Operation *> alreadyProcessedKernelOps;
    // For every operation directly in the block.
    for (mlir::Operation &root : block->getOperations()) {
      if (!isEligible(&root))
        continue;

      if (!alreadyProcessedKernelOps.insert(&root).second)
        continue;

      // Gather all eligible operations that are accessible from this op and
      // not blocked by a non eligible op.
      auto opsToFuseTogetherDownward = floodFillFrom<Direction::Down>(
          {&root}, isEligible, areCompatible, completeDataflowGraph);

      // Find all the leaves from the subgraph that was found by the previous
      // step.
      const auto isALeaf = [&](mlir::Operation *op) {
        return llvm::none_of(op->getUsers(), [&](auto user) {
          return opsToFuseTogetherDownward.contains(user);
        });
      };
      mlir::SmallVector<mlir::Operation *> leaves;
      for (mlir::Operation *op : opsToFuseTogetherDownward) {
        if (isALeaf(op))
          leaves.push_back(op);
      }

      // Gather all eligible operations again but going upward now.
      auto opsToFuseTogetherUpward = floodFillFrom<Direction::Up>(
          leaves, isEligible, areCompatible, completeDataflowGraph);

      // Combine the two sets.
      opsToFuseTogetherDownward.set_union(opsToFuseTogetherUpward);

      for (mlir::Operation *kernelOpInFusionSet : opsToFuseTogetherDownward)
        alreadyProcessedKernelOps.insert(kernelOpInFusionSet);

      subgraphs.push_back(llvm::SmallVector<mlir::Operation *>(
          opsToFuseTogetherDownward.begin(), opsToFuseTogetherDownward.end()));
    }
  });

  return subgraphs;
}

struct OperationDependency {
  /// Position in the bit vectors.
  uint64_t index;
  /// Bit set where 1 means that the operation is (transitively) a provider.
  llvm::BitVector transitiveProviders;
  /// Bit set where 1 means that the operation is (transitively) a user.
  llvm::BitVector transitiveUsers;
};

/// Gathers dependencies of an operation. Should be called in topological order
/// with Up direction and reverse topological order with Down direction. This
/// algorithm is O(n) but with favorable constants, thanks to the BitVector.
template <Direction direction>
static void dependencyWalk(
    mlir::Operation *op,
    llvm::MapVector<mlir::Operation *, OperationDependency>
        &eligibleOpDependencies,
    const mlir::DenseMap<mlir::Operation *, uint64_t> &opToIndexInBitVector,
    MO::ParametricTopologyAnalysis &completeDataflowGraph) {

  mlir::DenseSet<Operation *> visited;
  llvm::SmallVector<Operation *> workList;

  if constexpr (direction == Direction::Up) {
    auto parents = completeDataflowGraph.providers(op);
    workList.append(parents.begin(), parents.end());
  } else {
    auto users = completeDataflowGraph.users(op);
    workList.append(users.begin(), users.end());
  }

  while (!workList.empty()) {
    auto nextOp = workList.pop_back_val();
    // Lookup the operation. If it is not in the index map, then it does not
    // belong to this block. Ignore it.
    auto operationIndexIt = opToIndexInBitVector.find(nextOp);
    if (operationIndexIt == opToIndexInBitVector.end())
      continue;
    uint64_t operationIndex = operationIndexIt->second;

    // Op depends (as provider or user) on nextOp.
    if constexpr (direction == Direction::Up)
      eligibleOpDependencies[op].transitiveProviders[operationIndex] = true;
    else
      eligibleOpDependencies[op].transitiveUsers[operationIndex] = true;

    auto eligibleOpIt = eligibleOpDependencies.find(nextOp);
    bool isEligible = eligibleOpIt != eligibleOpDependencies.end();
    if (isEligible) {
      // If nextOp is eligible and because we are in topological (or reverse)
      // order, we can just grab the dependencies of this op and not recurse.
      if constexpr (direction == Direction::Up) {
        eligibleOpDependencies[op].transitiveProviders |=
            eligibleOpIt->second.transitiveProviders;
      } else {
        eligibleOpDependencies[op].transitiveUsers |=
            eligibleOpIt->second.transitiveUsers;
      }
    } else if (visited.insert(nextOp).second) {
      // If nextOp is not eligible and it was not visited, then recurse. We do
      // not track dependencies of non eligible ops because we don't fuse them.
      // Therefore, we have to recurse.
      if constexpr (direction == Direction::Up) {
        auto parents = completeDataflowGraph.providers(nextOp);
        workList.append(parents.begin(), parents.end());
      } else {
        auto users = completeDataflowGraph.users(op);
        workList.append(users.begin(), users.end());
      }
    }
  }
}

/// Gather dependencies for all operations in the provided block. O(n**2) but
/// with favorable constants, thanks to the BitVector. Using a MapVector to
/// guarantee a deterministic output.
static std::tuple<llvm::MapVector<mlir::Operation *, OperationDependency>,
                  llvm::BitVector>
gatherDependencies(Block *block, EligibilityFunction isEligible,
                   MO::ParametricTopologyAnalysis &completeDataflowGraph) {
  // Maps from any op in the block to it's index in the bit vectors we create.
  mlir::DenseMap<mlir::Operation *, uint64_t> opToIndexInBitVector;
  // The return value which maps an eligible op to its transitive dependencies.
  llvm::MapVector<mlir::Operation *, OperationDependency>
      eligibleOpDependencies;
  llvm::SmallVector<mlir::Operation *> eligibleOps;
  // A mask that can zero-out eligible ops. Useful later, when checking
  // parallelism.
  llvm::BitVector nonEligibleOpMask;
  uint64_t opIndex = 0;
  for (auto &op : block->getOperations()) {
    opToIndexInBitVector[&op] = opIndex++;
    if (isEligible(&op))
      eligibleOps.push_back(&op);
  }

  // Topologically sort the eligible ops and then compute the dependencies in
  // that order. This makes it possible not to traverse the graph again.
  completeDataflowGraph.computeTopologicalSorting(eligibleOps);

  nonEligibleOpMask.resize(opToIndexInBitVector.size(), true);
  for (auto op : eligibleOps) {
    auto &entry = eligibleOpDependencies[op];
    entry.transitiveProviders.resize(opToIndexInBitVector.size(), false);
    entry.transitiveUsers.resize(opToIndexInBitVector.size(), false);
    entry.index = opToIndexInBitVector[op];
    nonEligibleOpMask[entry.index] = false;
  }

  // Ops in the exlcusion list should not be considered as non-eligible when
  // checking for added dependencies. They may or may not be eligible.
  for (auto [op, index] : opToIndexInBitVector) {
    if (llvm::isa<EXCLUSION_LIST>(op))
      nonEligibleOpMask[index] = false;
  }

  for (mlir::Operation *eligibleOp : eligibleOps) {
    dependencyWalk<Direction::Up>(eligibleOp, eligibleOpDependencies,
                                  opToIndexInBitVector, completeDataflowGraph);
  }

  // Same logic for going down, except we use the reverse topological sort.
  for (mlir::Operation *eligibleOp : llvm::reverse(eligibleOps)) {
    dependencyWalk<Direction::Down>(eligibleOp, eligibleOpDependencies,
                                    opToIndexInBitVector,
                                    completeDataflowGraph);
  }

  return {eligibleOpDependencies, nonEligibleOpMask};
}

/// The analysis applies the following observation:
/// M := Set of non eligible ops.
/// S := A subgraph containing eligible ops that are compatible with one
/// another.
/// providers(S) := The set of all operations that transitively feed into an
/// eligible subgraph.
/// users(S) := The set of all operations that use an eligible subgraph.
// A ubgraph of eligible and compatible operations O cannot be added to another
/// subgraph of eligible and compatible operations S iff no cycle is created, no
/// dependency between non eligible ops is created and all operations in both
/// sets are compatible.
/// A cycle is created iff:
///   (providers(S) U providers(O)) intersects (users(S) U users(O))
/// A dependency between non eligible ops is created iff:
///   let S' = S intersection M
///   let O' = O intersection M
///   |providers(S') - providers(O')| != 0 && |users(O') - users(S')| != 0 &&
///   |providers(O') - providers(S')| != 0 && |users(S') - users(O')| != 0 ||
/// The implementation computes the transitive dependencies of all eligible ops.
/// Then, for each op, it checks if there is a set it can fuse with. If it
/// can't, it extends the sets by the current op. O(N**3) but with favorable
/// constants, thanks to the BitVector. The algorithm can also ignore
/// parallelism and only guard for cycles using `preserveParallelism`.
llvm::SmallVector<llvm::SmallVector<mlir::Operation *>>
transitiveDependencySequenceFusionAnalysis(
    MO::GraphOp graphOp, EligibilityFunction isEligible,
    CompatibilityFunction areCompatible,
    MO::ParametricTopologyAnalysis &completeDataflowGraph,
    bool preserveParallelism) {

  const auto wouldCreateANewDependency =
      [&](const OperationDependency &lhs, const OperationDependency &rhs,
          const llvm::BitVector &nonEligibleOpsMask) {
        const auto bitsetDifferenceNonEmpty = [&](const llvm::BitVector &lhs,
                                                  const llvm::BitVector &rhs) {
          // |S - O| != 0 implemented with bitwise ops:
          // lhs - (lhs & rhs) != 0
          static thread_local llvm::BitVector tmp;
          tmp = lhs;
          tmp &= rhs;
          return lhs.test(tmp);
        };

        static thread_local llvm::BitVector tmp1, tmp2, tmp3, tmp4;
        tmp1 = lhs.transitiveProviders;
        tmp2 = rhs.transitiveProviders;
        tmp3 = lhs.transitiveUsers;
        tmp4 = rhs.transitiveUsers;

        // S' = S intersection M
        tmp1 &= nonEligibleOpsMask;
        tmp2 &= nonEligibleOpsMask;
        tmp3 &= nonEligibleOpsMask;
        tmp4 &= nonEligibleOpsMask;

        bool createdNewDependency = (bitsetDifferenceNonEmpty(tmp1, tmp2) &&
                                     bitsetDifferenceNonEmpty(tmp4, tmp3)) ||
                                    (bitsetDifferenceNonEmpty(tmp2, tmp1) &&
                                     bitsetDifferenceNonEmpty(tmp3, tmp4));

        return createdNewDependency;
      };

  const auto wouldCreateACycle = [&](const OperationDependency &lhs,
                                     const OperationDependency &rhs) {
    static thread_local llvm::BitVector tmp1, tmp2;

    // all_providers = providers(lhs) U providers(rhs)
    tmp1 = lhs.transitiveProviders;
    tmp1 |= rhs.transitiveProviders;

    // all_users = users(lhs) U users(rhs)
    tmp2 = lhs.transitiveUsers;
    tmp2 |= rhs.transitiveUsers;

    // tmp1 = intersection(all_providers, all_users)
    tmp1 &= tmp2;

    // Is something a provider and a user at the same time?
    return tmp1.any();
  };

  struct SubgraphEntry {
    llvm::SmallVector<mlir::Operation *> operations;
    OperationDependency dependencies;
  };
  llvm::SmallVector<SubgraphEntry> subgraphEntries;

  graphOp.walk([&](Block *block) {
    auto [eligibleOpsDependencies, nonEligibleOpsMask] =
        gatherDependencies(block, isEligible, completeDataflowGraph);
    for (auto &[eligibleOp, eligibleOpDependencies] : eligibleOpsDependencies) {
      // It is possible that the current operation can fuse with many subgraphs.
      // We just pick the first one. This is because it happens very rarely. It
      // occurs when there are completely disjoint subgraphs (many connected
      // components).
      bool wasInsertedInASubgraph = false;
      for (auto &[operations, subgraphDependencies] : subgraphEntries) {
        if (operations.front()->getBlock() != block)
          continue;

        if (!areCompatible(eligibleOp, operations.front()))
          continue;

        if (wouldCreateACycle(subgraphDependencies, eligibleOpDependencies))
          continue;

        if (preserveParallelism &&
            wouldCreateANewDependency(subgraphDependencies,
                                      eligibleOpDependencies,
                                      nonEligibleOpsMask)) {
          continue;
        }

        subgraphDependencies.transitiveProviders |=
            eligibleOpDependencies.transitiveProviders;
        subgraphDependencies.transitiveUsers |=
            eligibleOpDependencies.transitiveUsers;
        subgraphDependencies.transitiveProviders[eligibleOpDependencies.index] =
            false;
        subgraphDependencies.transitiveUsers[eligibleOpDependencies.index] =
            false;

        operations.push_back(eligibleOp);
        wasInsertedInASubgraph = true;

        break;
      }

      if (!wasInsertedInASubgraph)
        subgraphEntries.push_back(
            {{eligibleOp}, std::move(eligibleOpDependencies)});
    }
  });

  llvm::SmallVector<llvm::SmallVector<mlir::Operation *>> subgraphs;
  for (auto &[operations, _] : subgraphEntries)
    subgraphs.push_back(operations);

  return subgraphs;
}
} // namespace M::MOGG
