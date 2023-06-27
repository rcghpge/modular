//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/MGPTransforms/MGPTransforms.h"

#include "GenericML/MGPDialect/MGPOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"

#include "./AsyncDataflowDependencies.h"

using namespace mlir;

namespace M::MGP {

// Generate support code for AsyncDataflowDependenciesTestPass
#define GEN_PASS_DEF_ASYNCDATAFLOWDEPENDENCIESTESTPASS
#include "GenericML/MGPTransforms/MGPTransforms.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// AsyncDataflowDependenciesTestPass
//===----------------------------------------------------------------------===//

class AsyncDataflowDependenciesTestPass
    : public impl::AsyncDataflowDependenciesTestPassBase<
          AsyncDataflowDependenciesTestPass> {
  using AsyncDataflowDependenciesTestPassBase::
      AsyncDataflowDependenciesTestPassBase;

  void runOnOperation() override {
    ModelOp modelOp = getOperation();
    IndexSet<Value> valueIndex = createValueIndexSet(modelOp);
    IndexSet<Operation *> operationIndex = createOperationIndexSet(modelOp);
    AsyncDataflowDependencies dependencies(&valueIndex, &operationIndex);

    valueIndex.printModel(llvm::outs());
    dependencies.print(llvm::outs());
  }
};

} // namespace
} // namespace M::MGP
