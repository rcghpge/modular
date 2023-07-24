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
  llvm::SetVector<Value> outputs;
  Subgraph(llvm::SetVector<Value> vals, llvm::SetVector<Value> outputs)
      : vals(vals), outputs(outputs) {}
};
SmallVector<Subgraph> ConstantSubgraphs(MO::GraphOp graph);

} // namespace M::MO

#endif // GENERICML_GRAPHCOMPILER_TRANSFORMS_MODIALECT_CONSTANTFOLD_H
