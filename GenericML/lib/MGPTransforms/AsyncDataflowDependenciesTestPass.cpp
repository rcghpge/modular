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
#include "./Utils.h"

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
    ValueTestPrinter printer(getOperation());

    printer.printModel(llvm::outs());

    AsyncDataflowDependencies dependencies;
    printer.printRelation(llvm::outs(), "sameAs", [&](Value lhs, Value rhs) {
      return dependencies.sameAs(lhs, rhs);
    });
    printer.printPredicate(llvm::outs(), "isAlwaysReady", [&](Value value) {
      return dependencies.isAlwaysReady(value);
    });
    printer.printRelation(llvm::outs(), "impliesReady",
                          [&](Value lhs, Value rhs) {
                            return dependencies.impliesReady(lhs, rhs);
                          });
  }
};

} // namespace
} // namespace M::MGP
