//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/MGPDialect/BufferBuilder.h"
#include "GenericML/MGPDialect/MGPOps.h"
#include "GenericML/MGPDialect/MGPTensorInterface.h"
#include "GenericML/MGPTransforms/MGPTransforms.h"
#include "GenericML/MGPTransforms/Utils.h"

namespace M::MGP {

#define GEN_PASS_DEF_FIXUPOVERHANGSPASS
#include "GenericML/MGPTransforms/MGPTransforms.h.inc"

namespace {

class FixupOverhangsPass
    : public impl::FixupOverhangsPassBase<FixupOverhangsPass> {
  using FixupOverhangsPassBase::FixupOverhangsPassBase;

  void runOnOperation() override {
    // Identify hanging chains in each block (i.e., chains with no uses) and
    // insert dependencies on the block terminator to ensure all chains are
    // awaited.

    ModelOp modelOp = getOperation();
    BufferBuilderContext builderCtx =
        TensorBufferDialectInterface::getAllBufferBuilders(modelOp);

    modelOp->walk([&](Block *block) {
      auto outputOp = block->getTerminator();

      SmallVector<Value> hangingChains;

      for (Operation &op : block->getOperations()) {
        for (auto result : op.getResults()) {
          const bool isChain =
              isa<MEF::MEFChainTypeInterface>(result.getType());
          if (isChain && result.use_empty())
            hangingChains.push_back(result);
        }
      }

      if (hangingChains.empty())
        return;

      OpBuilder builder(outputOp);
      if (failed(
              alsoWaitForChains(builder, builderCtx, outputOp, hangingChains)))
        outputOp->emitError("Could not wait on hanging chains");
    });
  }
};

} // namespace
} // namespace M::MGP
