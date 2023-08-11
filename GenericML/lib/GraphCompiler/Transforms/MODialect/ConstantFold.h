//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_TRANSFORMS_MODIALECT_CONSTANTFOLD_H
#define GENERICML_GRAPHCOMPILER_TRANSFORMS_MODIALECT_CONSTANTFOLD_H

#include "GenericML/GraphCompiler/MODialect/MOOps.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/IR/Value.h"

namespace M::MO {
struct Subgraph {
  llvm::SetVector<Value> vals;
  // Outputs are Value in the constant subgraph that have uses outside of the
  // subgraph. Outputs may depend on other outputs.
  llvm::SetVector<Value> outputs;
  Subgraph(llvm::SetVector<Value> vals) : vals(vals) { findOutputs(); }

  void remove(Value v) {
    vals.remove(v);
    outputs.clear();
    findOutputs();
  }

private:
  // Set outputs to those Values in vals which have uses outside the subgraph.
  void findOutputs() {
    for (auto v : vals) {
      // Constant ops cant be outputs
      if (v.getDefiningOp<ConstantOp>())
        continue;
      for (auto user : v.getUsers()) {
        // If one result is not in the constant subgraph, then they are all not.
        if (user->getNumResults() == 0 || !vals.contains(user->getResult(0))) {
          // If any user is not const, then this is an output.
          outputs.insert(v);
          break;
        }
      }
    }
  }
};
SmallVector<Subgraph> ConstantSubgraphs(MO::GraphOp graph);

} // namespace M::MO

#endif // GENERICML_GRAPHCOMPILER_TRANSFORMS_MODIALECT_CONSTANTFOLD_H
