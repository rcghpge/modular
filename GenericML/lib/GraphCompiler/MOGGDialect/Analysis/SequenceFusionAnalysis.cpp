//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/MOGGDialect/Analysis/SequenceFusionAnalysis.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGOps.h"
#include "Support/AssertStream.h"

#include "llvm-project/llvm/include/llvm/ADT/SetOperations.h"

namespace M::MOGG {

enum class Direction { Up, Down };

/// The returned set contains all operations that are eligible, compatible with
/// one another and all paths to them from the root are void of non eligible
/// ops. A SetVector is returned for determinism. Will not traverse across
/// blocks.
/// Note: The compatibility function is used on the edges during the traversal.
/// The operands must be interpreted as <from, to>.
template <Direction direction>
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
    } else if (!visitedAsNonEligible.insert(op)) {
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

FusionSubGraphs floodFillSequenceFusionAnalysis(
    Region &region, MO::ParametricTopologyAnalysis &completeDataflowGraph,
    EligibilityFunction isEligible, CompatibilityFunction areCompatible) {
  FusionSubGraphs subgraphs;
  // For every block in the graph (recursively).
  region.walk([&](Block *block) {
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
        return llvm::none_of(completeDataflowGraph.users(op), [&](auto user) {
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

      if (opsToFuseTogetherDownward.size() > 1) {
        subgraphs.push_back(llvm::SmallVector<mlir::Operation *>(
            opsToFuseTogetherDownward.begin(),
            opsToFuseTogetherDownward.end()));
      }
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
    llvm::MapVector<mlir::Operation *, OperationDependency> &dependencies,
    const mlir::DenseMap<mlir::Operation *, uint64_t> &opToIndexInBitVector,
    MO::ParametricTopologyAnalysis &completeDataflowGraph) {
  const auto grabDependencies = [&](auto nextOp) {
    auto &thisOpDependencies = dependencies[op];
    // Lookup the operation. If it is not in the index map, then it does not
    // belong to this block. Ignore it.
    auto operationIndexIt = opToIndexInBitVector.find(nextOp);
    if (operationIndexIt == opToIndexInBitVector.end())
      return;
    uint64_t operationIndex = operationIndexIt->second;

    const auto &dependency = dependencies[nextOp];
    // Op depends (as provider or user) on nextOp.
    if constexpr (direction == Direction::Up) {
      thisOpDependencies.transitiveProviders[operationIndex] = true;
      thisOpDependencies.transitiveProviders |= dependency.transitiveProviders;
    } else {
      thisOpDependencies.transitiveUsers[operationIndex] = true;
      thisOpDependencies.transitiveUsers |= dependency.transitiveUsers;
    }
  };

  if constexpr (direction == Direction::Up)
    completeDataflowGraph.forEachProvider(op, grabDependencies);
  else
    completeDataflowGraph.forEachUser(op, grabDependencies);
}

/// Gather dependencies for all operations in the provided block. O(n**2) but
/// with favorable constants, thanks to the BitVector. Using a MapVector to
/// guarantee a deterministic output.
static std::tuple<llvm::MapVector<mlir::Operation *, OperationDependency>,
                  llvm::BitVector>
gatherDependencies(
    Block *block, EligibilityFunction isEligible,
    ComputationallySignificantFunction isComputationallySignificant,
    MO::ParametricTopologyAnalysis &completeDataflowGraph) {
  // Maps from any op in the block to it's index in the bit vectors we create.
  mlir::DenseMap<mlir::Operation *, uint64_t> opToIndexInBitVector;
  // The return value which maps an eligible op to its transitive dependencies.
  llvm::MapVector<mlir::Operation *, OperationDependency> dependencies;
  // A mask that can zero-out eligible ops. Useful later, when checking
  // parallelism.
  uint64_t opIndex = 0;
  SmallVector<Operation *> allOps;
  for (auto &op : block->getOperations())
    allOps.push_back(&op);
  // Topologically sort the eligible ops and then compute the dependencies in
  // that order. This makes it possible not to traverse the graph again.
  completeDataflowGraph.computeTopologicalSorting(allOps);

  bool atLeastOneEligibleOp = false;
  for (auto op : allOps) {
    opToIndexInBitVector[op] = opIndex++;
    atLeastOneEligibleOp |= isEligible(op);
  }

  if (!atLeastOneEligibleOp)
    return {};

  for (auto op : allOps) {
    auto &entry = dependencies[op];
    entry.transitiveProviders.resize(opToIndexInBitVector.size(), false);
    entry.transitiveUsers.resize(opToIndexInBitVector.size(), false);
    entry.index = opToIndexInBitVector[op];
  }

  llvm::BitVector computationallySignificantMask;
  computationallySignificantMask.resize(opToIndexInBitVector.size(), false);
  for (auto [op, index] : opToIndexInBitVector) {
    if (isComputationallySignificant(op)) {
      GENERICML_ASSERT(!isEligible(op))
          << "Eligible ops are considered computationally insignificant";
      computationallySignificantMask[index] = true;
    }
  }

  for (auto op : allOps) {
    dependencyWalk<Direction::Up>(op, dependencies, opToIndexInBitVector,
                                  completeDataflowGraph);
  }

  // Same logic for going down, except we use the reverse topological sort.
  for (auto op : llvm::reverse(allOps)) {
    dependencyWalk<Direction::Down>(op, dependencies, opToIndexInBitVector,
                                    completeDataflowGraph);
  }

  dependencies.remove_if([&](auto entry) { return !isEligible(entry.first); });

  return {dependencies, computationallySignificantMask};
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
FusionSubGraphs transitiveDependencySequenceFusionAnalysis(
    Region &region, MO::ParametricTopologyAnalysis &completeDataflowGraph,
    EligibilityFunction isEligible, CompatibilityFunction areCompatible,
    ComputationallySignificantFunction isComputationallySignificant,
    bool preserveParallelism) {
  const auto wouldCreateANewDependency =
      [&](const OperationDependency &lhs, const OperationDependency &rhs,
          const llvm::BitVector &computationallySignificantMask) {
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
        tmp1 &= computationallySignificantMask;
        tmp2 &= computationallySignificantMask;
        tmp3 &= computationallySignificantMask;
        tmp4 &= computationallySignificantMask;

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
    llvm::BitVector operationsAskMask;
    OperationDependency dependencies;
  };
  llvm::SmallVector<SubgraphEntry, 4> subgraphEntries;

  region.walk([&](Block *block) {
    auto [eligibleOpsDependencies, computationallySignificantMask] =
        gatherDependencies(block, isEligible, isComputationallySignificant,
                           completeDataflowGraph);
    while (!eligibleOpsDependencies.empty()) {
      auto [eligibleOp, eligibleOpDependencies] =
          std::move(eligibleOpsDependencies.front());
      eligibleOpsDependencies.erase(eligibleOpsDependencies.begin());

      bool wasInsertedInASubgraph = false;
      for (auto &[operations, operationsAskMask, subgraphDependencies] :
           subgraphEntries) {
        if (operations.front()->getBlock() != block)
          continue;

        if (!areCompatible(eligibleOp, operations.front()))
          continue;

        if (wouldCreateACycle(subgraphDependencies, eligibleOpDependencies))
          continue;

        if (preserveParallelism &&
            wouldCreateANewDependency(subgraphDependencies,
                                      eligibleOpDependencies,
                                      computationallySignificantMask)) {
          continue;
        }

        operationsAskMask[eligibleOpDependencies.index] = true;
        subgraphDependencies.transitiveProviders |=
            eligibleOpDependencies.transitiveProviders;
        subgraphDependencies.transitiveUsers |=
            eligibleOpDependencies.transitiveUsers;
        subgraphDependencies.transitiveProviders[eligibleOpDependencies.index] =
            false;
        subgraphDependencies.transitiveUsers[eligibleOpDependencies.index] =
            false;
        operations.push_back(eligibleOp);

        const auto updateDependenciesAfterFusion =
            [](OperationDependency &dependenciesToUpdate,
               const OperationDependency &subgraphDependencies,
               const llvm::BitVector &mask) {
              const bool providerOfTheSubgraph =
                  dependenciesToUpdate.transitiveProviders.anyCommon(mask);
              if (providerOfTheSubgraph) {
                dependenciesToUpdate.transitiveProviders |= mask;
                dependenciesToUpdate.transitiveProviders |=
                    subgraphDependencies.transitiveProviders;
              }
              const bool userOfTheSubgraph =
                  dependenciesToUpdate.transitiveUsers.anyCommon(mask);
              if (userOfTheSubgraph) {
                dependenciesToUpdate.transitiveUsers |= mask;
                dependenciesToUpdate.transitiveUsers |=
                    subgraphDependencies.transitiveUsers;
              }
              GENERICML_ASSERT(!(providerOfTheSubgraph && userOfTheSubgraph))
                  << "Created a cycle";
            };

        // After fusing an operation with a subgraph, update the dependencies of
        // any other op or subgraph to account for the newly created
        // dependencies.
        for (auto &[op, eligibleOpDependencies_] : eligibleOpsDependencies) {
          if (op->getBlock() == block) {
            updateDependenciesAfterFusion(eligibleOpDependencies_,
                                          subgraphDependencies,
                                          operationsAskMask);
          }
        }

        for (auto &[operations_, __, subgraphDependencies_] : subgraphEntries) {
          if (operations_.front()->getBlock() == block) {
            updateDependenciesAfterFusion(
                subgraphDependencies_, subgraphDependencies, operationsAskMask);
          }
        }

        wasInsertedInASubgraph = true;

        break;
      }

      if (!wasInsertedInASubgraph) {
        llvm::BitVector operationsAskMask(
            eligibleOpDependencies.transitiveProviders.size(), false);
        operationsAskMask[eligibleOpDependencies.index] = true;
        subgraphEntries.push_back(
            {{eligibleOp}, operationsAskMask, eligibleOpDependencies});
      }
    }
  });

  FusionSubGraphs subgraphs;
  for (auto &[operations, _, __] : subgraphEntries) {
    if (operations.size() > 1)
      subgraphs.push_back(operations);
  }

  return subgraphs;
}

FusionSubGraphs sourceDestinationSequenceFusionAnalysis(
    Region &region, MO::ParametricTopologyAnalysis &completeDataflowGraph,
    EligibilityFunction isEligible, CompatibilityFunction areCompatible,
    bool allowDuplication) {
  FusionSubGraphs subgraphs;
  region.walk([&](Block *block) {
    SmallVector<Operation *> eligibleOps;
    for (auto &op : block->getOperations()) {
      if (isEligible(&op))
        eligibleOps.push_back(&op);
    }

    // Safe to do over the subset eligibleOps rather than the whole graph
    // because if ops are not connected, they won't fuse anyway.
    completeDataflowGraph.computeTopologicalSorting(eligibleOps);

    DenseSet<Operation *> wouldBeDCE;
    for (auto op : llvm::reverse(eligibleOps)) {
      // If an op was fused, it means it will be DCE-ed later so we do not
      // attempt to fuse things into it.
      if (wouldBeDCE.contains(op))
        continue;

      DenseSet<Operation *> subgraph;
      llvm::SmallVector<std::tuple<mlir::Operation *, Operation *>> workList = {
          {op, nullptr}};

      auto areSameBlockAndCompatible = [&](Operation *parent,
                                           Operation *child) {
        return isEligible(parent) && isEligible(child) &&
               parent->getBlock() == child->getBlock() &&
               areCompatible(parent, child);
      };

      while (!workList.empty()) {
        auto parent_child = workList.pop_back_val();
        // Can't use structured bindings because a lambda captures parent which
        // is possible only in >C++20
        auto parent = std::get<0>(parent_child);
        auto child = std::get<1>(parent_child);

        auto users = completeDataflowGraph.users(parent);
        if (child && !llvm::all_of(users, [&](auto user) {
              return areSameBlockAndCompatible(parent, user);
            }))
          continue;

        // If duplication is not allowed, then the only user of the current op
        // must be in the fusion subgraph already. Note that if there are more
        // than one user, this will fail because one of the users will be in
        // another subgraph from a past iteration.
        if (child && !allowDuplication && !llvm::all_of(users, [&](auto user) {
              return subgraph.contains(user);
            }))
          continue;

        subgraph.insert(parent);
        wouldBeDCE.insert(parent);

        for (auto grandParent : completeDataflowGraph.providers(parent))
          workList.push_back({grandParent, parent});
      }

      if (subgraph.size() > 1)
        subgraphs.emplace_back(subgraph.begin(), subgraph.end());
    }
  });
  return subgraphs;
}

} // namespace M::MOGG
