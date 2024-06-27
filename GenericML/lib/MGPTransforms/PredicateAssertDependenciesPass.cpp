//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/MGPTransforms/MGPTransforms.h"

#include "GenericML/MGPDialect/BufferBuilder.h"
#include "GenericML/MGPDialect/MGPOps.h"
#include "GenericML/MGPDialect/MGPTensorInterface.h"
#include "GenericML/MGPTransforms/Utils.h"
#include "GraphRT/GRTDialect/GRTOps.h"
#include "GraphRT/MEFDialect/MEFTypes.h"
#include "GraphRT/MIPDialect/MIPOps.h"
#include "llvm-project/llvm/include/llvm/ADT/SetOperations.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

#define EXCLUDED_TYPES MGP::ContextType, MEF::MEFChainTypeInterface

using namespace mlir;

namespace M::MGP {

// Generate support code for PredicateAssertDependenciesPass.
#define GEN_PASS_DEF_PREDICATEASSERTDEPENDENCIESPASS
#include "GenericML/MGPTransforms/MGPTransforms.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// PredicateAssertDependenciesPass
//===----------------------------------------------------------------------===//

/// After this pass, any operation that takes in a chain and uses a value that
/// contributed to at least one assert will have to depend on the chains of
/// those asserts.
class PredicateAssertDependenciesPass
    : public impl::PredicateAssertDependenciesPassBase<
          PredicateAssertDependenciesPass> {
  using PredicateAssertDependenciesPassBase::
      PredicateAssertDependenciesPassBase;

  void runOnOperation() override {
    auto &ctx = getContext();
    OpBuilder builder(&ctx);

    // TODO(#19098): The rewrite is done over the blocks in the IR. There is
    // discontinuity between blocks meaning no uses are tracked across blocks.
    // This means we are not yet tracking dependencies across blocks which may
    // cause errors that would have been prevented with the assertion.
    ModelOp model = getOperation();
    BufferBuilderContext builderCtx =
        TensorBufferDialectInterface::getAllBufferBuilders(model);

    auto perBlockRewrite = [&](Block *block) {
      // First, build a mapping of values and the assertions that they
      // contribute to.
      // TODO(#19098): This map is valid within a block.
      llvm::MapVector<Value, llvm::SetVector<AssertOp>> valuesThatFeedAsserts;
      for (auto assertOp : block->getOps<AssertOp>()) {
        DenseSet<Operation *> visited;
        std::function<void(Value)> propagateUp = [&](Value val) {
          Operation *op = val.getDefiningOp();

          // Block arguments
          if (!op) {
            valuesThatFeedAsserts[val].insert(assertOp);
            return;
          }

          // Skip if already visited.
          if (!visited.insert(op).second)
            return;

          // All results of an operation should be considered, not just the
          // value that went into an assert.
          for (auto result : op->getResults())
            valuesThatFeedAsserts[result].insert(assertOp);

          for (auto operand : op->getOperands())
            propagateUp(operand);
        };

        for (auto operand : assertOp->getOperands())
          propagateUp(operand);
      }

      llvm::MapVector<Operation *, llvm::SetVector<AssertOp>>
          operationsThatDependOnAsserts;
      const auto storeAssertionDependenciesForOp =
          [&](Operation *op, const SetVector<AssertOp> &asserts) {
            auto &assertsItDependsOn = operationsThatDependOnAsserts[op];
            uint64_t before = assertsItDependsOn.size();
            assertsItDependsOn.insert(asserts.begin(), asserts.end());

            // This is the equivalent of checking if the op was already
            // visited with that set of asserts.
            if (assertsItDependsOn.size() == before)
              return false;

            // This does not guarantee that the created chains are optimal (no
            // need for simplification) because the order in which we visit
            // these operations is arbitrary. SimplifyChains pass can take
            // care of simplification. The purpose of this statement is to
            // avoid recursing when we know for sure we do not need to. This
            // is because the dependent operations will naturally inherit the
            // dependencies of this operation. If we predicate at this
            // operation, no need to visit descendant operations.
            auto inputChains = getInputChainOpOperands(op);
            if (!inputChains.empty())
              return false;

            return true;
          };

      // TODO(#19098): Propagate the assert dependencies on all users of the
      // values that contributed to the assertions in the current block.
      const std::function<void(Value, llvm::SetVector<AssertOp>)>
          propagateDown = [&](Value val, const SetVector<AssertOp> &asserts) {
            // These types should be excluded because they will create false
            // dependencies.
            auto op = val.getDefiningOp();
            if (op && !storeAssertionDependenciesForOp(op, asserts))
              return;

            if (isa<EXCLUDED_TYPES>(val.getType()))
              return;

            for (auto user : val.getUsers()) {
              if (!user->getNumResults())
                storeAssertionDependenciesForOp(user, asserts);
              else {
                for (auto result : user->getResults())
                  propagateDown(result, asserts);
              }
            }
          };

      for (auto &[val, asserts] : valuesThatFeedAsserts)
        propagateDown(val, asserts);

      for (auto &[op, asserts] : operationsThatDependOnAsserts) {
        // An operation that does not take in a chain will be ignored.
        auto inputChains = getInputChainOpOperands(op);
        if (inputChains.empty())
          continue;

        llvm::SetVector<AssertOp> diff = asserts;
        if (op->getNumResults()) {
          auto it = valuesThatFeedAsserts.find(op->getResult(0));
          if (it != valuesThatFeedAsserts.end())
            diff = llvm::set_difference(it->second, asserts);
        }

        // This is equivalent to |assertsYouDependOn - assertsYouContributedTo|
        if (diff.empty())
          continue;

        // For readability.
        builder.setInsertionPoint(op);
        SmallVector<Value> chains;
        for (AssertOp assert : diff)
          chains.emplace_back(assert.getResult());
        if (failed(alsoWaitForChains(builder, builderCtx, op, chains)))
          op->emitError("Could not wait on additional chains");
      }
    };

    model->walk(perBlockRewrite);
  }
};

} // namespace
} // namespace M::MGP
