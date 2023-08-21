//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_TRANSFORMS_MODIALECT_CONSTANTFOLD_H
#define GENERICML_GRAPHCOMPILER_TRANSFORMS_MODIALECT_CONSTANTFOLD_H

#include "GenericML/GraphCompiler/MODialect/MOOps.h"
#include "KGEN/KGENDialect/KGENParameters.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/IR/Value.h"

namespace M::MO {
struct Subgraph {
  llvm::SetVector<Operation *> ops;
  // Outputs are Value in the constant subgraph that have uses outside of the
  // subgraph. Outputs may depend on other outputs.
  llvm::SetVector<Operation *> outputs;
  Subgraph(llvm::SetVector<Operation *> ops) : ops(ops) { findOutputs(); }

  void remove(Operation *op) {
    ops.remove(op);
    outputs.clear();
    findOutputs();
  }

  /// Is the Value completely internal to this subgraph.
  bool fullyContains(Value v) const {
    return llvm::all_of(v.getUsers(),
                        [this](Operation *user) { return ops.contains(user); });
  }

  /// List of parameters declared and used in this subgraph. Parameters are
  /// returned in topological order.
  llvm::SetVector<StringAttr> paramDeclarations() const;

private:
  // Set outputs to those Values in vals which have uses outside the subgraph.
  void findOutputs() {
    for (Operation *op : ops) {
      // Constant ops cant be outputs
      if (dyn_cast<ConstantOp>(op))
        continue;
      for (auto user : op->getUsers()) {
        if (dyn_cast<MOControlOpInterface>(user) || !ops.contains(user)) {
          // If any user is not const, then this is an output.
          outputs.insert(op);
          break;
        }
      }
    }
  }
};
SmallVector<Subgraph>
constantSubgraphs(MO::GraphOp graph,
                  KGEN::ParameterCollector::Analysis &paramCache);

} // namespace M::MO

#endif // GENERICML_GRAPHCOMPILER_TRANSFORMS_MODIALECT_CONSTANTFOLD_H
