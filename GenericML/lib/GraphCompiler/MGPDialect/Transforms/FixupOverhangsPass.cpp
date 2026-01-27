//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/MGPDialect/Dialect/BufferBuilder.h"
#include "GenericML/GraphCompiler/MGPDialect/Dialect/MGPOps.h"
#include "GenericML/GraphCompiler/MGPDialect/Dialect/MGPTensorInterface.h"
#include "GenericML/GraphCompiler/MGPDialect/Transforms/MGPTransforms.h"
#include "GenericML/GraphCompiler/MGPDialect/Transforms/Utils.h"

namespace M::MGP {

#define GEN_PASS_DEF_FIXUPOVERHANGSPASS
#include "GenericML/GraphCompiler/MGPDialect/Transforms/MGPTransforms.h.inc"

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
          // All async ref results will be set to error upon failure, thus one
          // async value per operation is sufficient to propagate error
          // correctly.
          if (result.use_empty()) {
            hangingChains.push_back(result);
            break;
          }
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
