//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
#include "ConstantFold.h"
#include "GenericML/GraphCompiler/MODialect/MOOps.h"
#include "GenericML/GraphCompiler/Transforms/MODialect/MOTransforms.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
namespace M::MO {
#define GEN_PASS_DEF_CONSTANTFOLDSUBGRAPHTEST
#include "GenericML/GraphCompiler/Transforms/MODialect/MOTransforms.h.inc"

class ConstantFoldSubgraphTestPass
    : public impl::ConstantFoldSubgraphTestBase<ConstantFoldSubgraphTestPass> {
  using ConstantFoldSubgraphTestBase::ConstantFoldSubgraphTestBase;

  void runOnOperation() override {
    auto &paramCache = getAnalysis<KGEN::ParameterCollector::Analysis>();
    getOperation().walk([&](GraphOp op) {
      llvm::outs() << op << "\n";
      for (Subgraph &subgraph : constantSubgraphs(op, paramCache)) {
        llvm::outs() << "subgraph ";
        std::vector<std::string> vars;
        for (Operation *op : subgraph.ops) {
          for (auto v : op->getResults()) {
            std::string s;
            llvm::raw_string_ostream ss(s);
            v.printAsOperand(ss, mlir::OpPrintingFlags());
            vars.push_back(ss.str());
          }
        }
        // sort so values always come out in the same order
        std::sort(vars.begin(), vars.end());
        for (auto v : vars)
          llvm::outs() << v << " ";
        llvm::outs() << "\n";
        llvm::outs() << "outputs ";
        std::vector<std::string> outs;
        for (Operation *op : subgraph.outputs) {
          for (auto v : op->getResults()) {
            std::string s;
            llvm::raw_string_ostream ss(s);
            v.printAsOperand(ss, mlir::OpPrintingFlags());
            outs.push_back(ss.str());
          }
        }
        // sort so values always come out in the same order
        std::sort(outs.begin(), outs.end());
        for (auto v : outs)
          llvm::outs() << v << " ";
        llvm::outs() << "\n";
      }
    });
  }
};
} // namespace M::MO
