//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGPasses.h"
#include "KGEN/KGENDialect/KGENOps.h"

#include "llvm/ADT/StringSet.h"

#include <optional>

using namespace M;

namespace M::MOGG {

#define GEN_PASS_DEF_SIMPLIFYKERNELOPERANDS
#include "GenericML/GraphCompiler/MOGGDialect/MOGGPasses.h.inc"

namespace {

struct SimplifyKernelOperands
    : public impl::SimplifyKernelOperandsBase<SimplifyKernelOperands> {
  using SimplifyKernelOperandsBase::SimplifyKernelOperandsBase;

  void runOnOperation() override {
    MO::GraphOp graphOp = getOperation();

    graphOp->walk([](MOGG::KernelOp kernelOp) {
      DenseMap<Value, BlockArgument> map;
      for (int idx = kernelOp.getInputs().size() - 1; idx >= 0; idx--) {
        auto [it, inserted] =
            map.insert({kernelOp.getInputs()[idx],
                        kernelOp.getExecBody().getArgument(idx)});
        if (!inserted) {
          // Have duplication.
          kernelOp.getExecBody().getArgument(idx).replaceAllUsesWith(
              it->second);
          kernelOp.eraseInputsAt(idx);
        }
      }

      kernelOp.walk([](MOGG::BindLambdaOp bindOp) {
        DenseMap<Value, BlockArgument> map;
        for (int idx = bindOp.getNumOperands() - 1; idx >= 0; idx--) {
          auto [it, inserted] = map.insert(
              {bindOp.getOperand(idx), bindOp.getBody()->getArgument(idx)});
          if (!inserted) {
            // Have duplication.
            bindOp.getBody()->getArgument(idx).replaceAllUsesWith(it->second);
            bindOp.getBody()->eraseArgument(idx);
            bindOp->eraseOperand(idx);
          }
        }
      });
    });
  }
};

} // namespace

} // namespace M::MOGG
