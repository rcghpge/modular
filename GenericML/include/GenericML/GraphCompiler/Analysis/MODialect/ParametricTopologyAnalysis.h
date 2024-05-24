//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MODIALECT_PARAMETRICTOPOLOGYANALYSIS_H
#define GENERICML_GRAPHCOMPILER_MODIALECT_PARAMETRICTOPOLOGYANALYSIS_H

#include "mlir/IR/Operation.h"
#include "llvm/ADT/SetVector.h"

namespace M::MO {

// (1) Parameters cannot be discovered using MLIR's builtin dataflow API such as
// op->getUsers() because they are attributes. Operations (value) can create
// parameters (attributes) that other operations (value) consume. This
// dependency is not directly observable via the dataflow API. This analysis
// aims at extending it.
// (2) Furthermore, operations that aren't isolated from above can contain
// nested operations that depend on other operations in superseding blocks
// (contained in). Therefore, it is not directly obvious if an operation
// provides another one if it is only providing nested operations. This analysis
// aims at extending the regular SSA dataflow with this information.
// Example:
// %0 = ...
// %1 = op1 {
//   %2 = op2(%0)
// }
// Here, %0 is a provider of both op2 AND op1.
class ParametricTopologyAnalysis {
public:
  // Construct the state necessary rooted at the given operation.
  ParametricTopologyAnalysis(mlir::Operation *root);
  ~ParametricTopologyAnalysis() = default;

  // Returns {op->getUsers()} U {parametricUsers(op)}.
  /// NOTE: Passes that leverage this analysis have to emit deterministic IR
  /// meaning two invocations of the same pass over the same input should output
  /// the same IR. Therefore, the output of this function is a SmallSetVector
  /// which preserves insertion order.
  llvm::SmallSetVector<mlir::Operation *, 4> users(mlir::Operation *op) const;

  template <typename Func>
  inline void forEachUser(mlir::Operation *op, Func &&func) const {
    for (auto user : op->getUsers())
      func(user);

    auto it = metadata.find(op);
    if (it != metadata.end()) {
      for (auto user : it->second.parametricUsers)
        func(user);
    }
  }

  // Returns {dataflowProviders(op)} U {parametricProviders(op)}.
  /// NOTE: Passes that leverage this analysis have to emit deterministic IR
  /// meaning two invocations of the same pass over the same input should output
  /// the same IR. Therefore, the output of this function is a SmallSetVector
  /// which preserves insertion order.
  llvm::SmallSetVector<mlir::Operation *, 4>
  providers(mlir::Operation *op) const;

  template <typename Func>
  void forEachProvider(mlir::Operation *op, Func &&func) const {
    auto it = metadata.find(op);
    if (it != metadata.end()) {
      for (auto provider : it->second.parametricProviders)
        func(provider);
    }

    for (auto operand : op->getOperands()) {
      if (auto provider = operand.getDefiningOp())
        func(provider);
    }
  }

  // Will compute the topological sort of a set of operations but accounting
  // for parametric dependencies.
  void computeTopologicalSorting(mlir::MutableArrayRef<mlir::Operation *> ops);

  // Topologically sort operations within `region` in place, accounting for
  // parametric dependencies and ops within nested regions.
  void topologicallySortAllOpsWithin(mlir::Region *region);

private:
  void
  computeTopologicalSortingImpl(mlir::MutableArrayRef<mlir::Operation *> ops);

  // Topologically sort operations within `block` in place (excluding ones in
  // nested region), accounting for parametric dependencies.
  void topologicallySortFirstLevelOpsWithin(mlir::Block *block);

  bool isOpReady(mlir::Operation *op,
                 mlir::DenseSet<mlir::Operation *> &unscheduledOps);

  struct OperationRankAndDependency {
    llvm::SmallSetVector<mlir::Operation *, 4> parametricProviders;
    llvm::SmallSetVector<mlir::Operation *, 4> parametricUsers;
  };

  mlir::DenseMap<mlir::Operation *, OperationRankAndDependency> metadata;
};

} // namespace M::MO

#endif
