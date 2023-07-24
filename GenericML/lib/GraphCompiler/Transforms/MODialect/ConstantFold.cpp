//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "./ConstantFold.h"
#include "GenericML/GraphCompiler/Transforms/MODialect/MOTransforms.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/Hashing.h"

namespace M::MO {
#define GEN_PASS_DEF_CONSTANTFOLD
#include "GenericML/GraphCompiler/Transforms/MODialect/MOTransforms.h.inc"

struct ValueCompare {
  bool operator()(const Value &lhs, const Value &rhs) const {
    return hash_value(lhs) < hash_value(rhs);
  }
};

SmallVector<Subgraph> ConstantSubgraphs(MO::GraphOp graph) {
  // Recurse down consumers ConstantOps, marking ops with all constant inputs as
  // constant. If inputs have not been visited yet, then we recurse up the use
  // chain on inputs until we determine if they are const or not.

  // Set containing values that are known to not be constant. That is one or
  // more inputs to the operation creating the values are not constant.
  llvm::DenseSet<Value> non_const;
  // Disjoint set containing constant values. A constant value is either a
  // mo.constant or a value whose defining operation has all constant value
  // inputs. Each equivalence class is a connected DAG of values which depend on
  // each other.
  llvm::EquivalenceClasses<Value, ValueCompare> const_subgraphs;
  // Stack of values that need to be visited. Values may be popped and then
  // pushed back on the queue if all parent values have not been resolved. New
  // values are added when the parents of an operation are unknown or the users
  // of a value when it has been determined to be constant.
  llvm::SmallVector<Value> work_queue;
  graph.walk([&](ConstantOp op) {
    for (auto user : op.getResult().getUsers())
      for (auto result : user->getResults())
        work_queue.push_back(result);
    const_subgraphs.insert(op.getResult());
  });

  // Run for a fixed amount of iterations. This avoids unexpected infinite
  // recursion within a graphop.
  size_t iter = 0;
  size_t bound_for_maximum_iterations = 100000;
  for (; iter < bound_for_maximum_iterations; iter++) {
    if (work_queue.empty())
      break;

    Value v = work_queue.back();
    work_queue.pop_back();

    // Value is already non-const, skip to next on work queue.
    if (non_const.contains(v))
      continue;

    // already visited
    if (const_subgraphs.findValue(v) != const_subgraphs.end())
      continue;

    auto op = v.getDefiningOp();
    // If there is no defining op, then this value is not constant.
    if (!op) {
      non_const.insert(v);
      continue;
    }

    // Don't constant fold any op with regions.
    if (op->getNumRegions() != 0) {
      non_const.insert(v);
      continue;
    }

    llvm::DenseSet<Value> unvisited;
    for (auto operand : op->getOperands()) {
      // If any operand is non-const then this op/value is also non-const
      if (non_const.contains(operand)) {
        non_const.insert(v);
        continue;
      }
      if (const_subgraphs.findValue(operand) == const_subgraphs.end()) {
        // Operand has not been visited yet.
        unvisited.insert(operand);
      }
    }

    if (!unvisited.empty()) {
      // All operands have not been visited, push the current value, then the
      // unvisited operands (so they get handled first).
      work_queue.push_back(v);
      for (auto operand : unvisited)
        work_queue.push_back(operand);
      continue;
    }

    // All operands are const, union this value with all operands.
    for (auto operand : v.getDefiningOp()->getOperands())
      const_subgraphs.unionSets(v, operand);
  }
  assert(iter < bound_for_maximum_iterations &&
         "Hit bound on maximum number of iterations for detecting constant "
         "subgraphs. Either this graph op is VERY large (in which case this "
         "bound should be increased), or else somehow there is infinite "
         "recursion (which is a bug)");

  // Collect subgraphs from equivalence sets.
  SmallVector<Subgraph, 0> subgraphs;
  for (auto it = const_subgraphs.begin(), end = const_subgraphs.end();
       it != end; ++it) {
    if (!it->isLeader())
      continue;
    // Ignore subgraphs that only contain constant ops.
    if (llvm::all_of(
            llvm::iterator_range(const_subgraphs.member_begin(it),
                                 const_subgraphs.member_end()),
            [](Value v) { return dyn_cast<ConstantOp>(v.getDefiningOp()); }))
      continue;
    llvm::SetVector<Value> members(const_subgraphs.member_begin(it),
                                   const_subgraphs.member_end());

    llvm::SetVector<Value> outputs;
    for (auto m : members) {
      for (auto user : m.getUsers()) {
        if (user->getNumResults() == 0 ||
            const_subgraphs.findValue(user->getResult(0)) ==
                const_subgraphs.end()) {
          // If any user is not const, then this is an output.
          outputs.insert(m);
          break;
        }
      }
    }
    subgraphs.push_back(Subgraph(members, outputs));
  }

  return subgraphs;
}

struct ConstantFoldPass
    : public M::MO::impl::ConstantFoldBase<ConstantFoldPass> {
  using ConstantFoldBase::ConstantFoldBase;

  void runOnOperation() override;
};
void ConstantFoldPass::runOnOperation() { return signalPassFailure(); }

} // namespace M::MO
