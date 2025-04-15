//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/MGPDialect/MGPOps.h"
#include "GenericML/MGPTransforms/MGPTransforms.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"

#define DEBUG_TYPE "globalize-slot-indices"

namespace M::MGP {

#define GEN_PASS_DEF_GLOBALIZESLOTINDICESPASS
#include "GenericML/MGPTransforms/MGPTransforms.h.inc"

}; // namespace M::MGP

using namespace M;
using namespace MGP;

//===----------------------------------------------------------------------===//
// GlobalizeSlotIndicesPass
//===----------------------------------------------------------------------===//

/// Offsets the specific slot index attribute of various MGP operations.
///
/// This function checks the type of the input `op` and updates the
/// corresponding attribute that holds its primary slot index
/// (e.g., `bBufferSlot` for `BufferCacheOp`, `aRuntimeSlot` for
/// `RuntimeCreateOp`, etc.) by adding the provided `slotOffset`.
/// This is necessary because different operations use different attribute names
/// for their associated slot indices.
///
/// @param op The operation whose slot index should be offset.
/// @param slotOffset The value to add to the existing slot index.
static void offsetRuntimeSlot(Operation *op, uint64_t slotOffset) {
  if (auto bufferCacheOp = dyn_cast<BufferCacheOp>(op)) {
    uint64_t oldSlot = bufferCacheOp.getBBufferSlot();
    bufferCacheOp.setBBufferSlot(oldSlot + slotOffset);
  }
  if (auto runtimeCreateOp = dyn_cast<RuntimeCreateOp>(op)) {
    uint64_t oldSlot = runtimeCreateOp.getARuntimeSlot();
    runtimeCreateOp.setARuntimeSlot(oldSlot + slotOffset);
  }
  if (auto bufferGetCachedOp = dyn_cast<BufferGetCachedOp>(op)) {
    uint64_t oldSlot = bufferGetCachedOp.getBBufferSlot();
    bufferGetCachedOp.setBBufferSlot(oldSlot + slotOffset);
  }
  if (auto bufferRemoveCachedOp = dyn_cast<BufferRemoveCachedOp>(op)) {
    uint64_t oldSlot = bufferRemoveCachedOp.getBufferSlot();
    bufferRemoveCachedOp.setBufferSlot(oldSlot + slotOffset);
  }
  if (auto moduleLoadOp = dyn_cast<ModuleLoadOp>(op)) {
    uint64_t oldSlot = moduleLoadOp.getARuntimeSlot();
    moduleLoadOp.setARuntimeSlot(oldSlot + slotOffset);
  }
  if (auto executeGenericOp = dyn_cast<ExecuteGenericOp>(op)) {
    uint64_t oldSlot = executeGenericOp.getARuntimeSlot();
    executeGenericOp.setARuntimeSlot(oldSlot + slotOffset);
  }
}

namespace {

struct GlobalizeSlotIndicesPass
    : public impl::GlobalizeSlotIndicesPassBase<GlobalizeSlotIndicesPass> {
  using GlobalizeSlotIndicesPassBase::GlobalizeSlotIndicesPassBase;

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();

    // Keep track of the global maximum slot index across all models.
    uint64_t globalMaxSlot = 0;

    // Collect all models first to avoid issues while iterating and modifying.
    SmallVector<ModelOp> allModels;
    moduleOp.walk([&](ModelOp modelOp) { allModels.push_back(modelOp); });

    // Go through every model, offset its slot indices, and update the global
    // max.
    for (auto model : allModels) {
      auto initBlock = model.getInitBlock();
      auto executeBlock = model.getExecuteBlock();

      // Find the number of slots this specific model requires.
      ContextCreateOp contextCreateOp;
      initBlock->walk([&](ContextCreateOp op) { contextCreateOp = op; });

      // MGP::ModelOp::verify ensures there's only one ContextCreateOp, so this
      // should never happen.
      if (!contextCreateOp) {
        model.emitError("Model does not contain a ContextCreateOp in its init "
                        "block.");
        signalPassFailure();
        return;
      }

      uint64_t modelSlotsUsed = contextCreateOp.getNumSlots();

      // Go through the init block and modify all the slot indices using the
      // current global offset.
      initBlock->walk(
          [&](Operation *op) { offsetRuntimeSlot(op, globalMaxSlot); });
      // Do the same for the execute block.
      executeBlock->walk(
          [&](Operation *op) { offsetRuntimeSlot(op, globalMaxSlot); });

      // Add the slots used by this model to the global count.
      globalMaxSlot += modelSlotsUsed;
    }

    // After processing all models and calculating the total slots needed,
    // update every model's context create op to reserve the full amount.
    // This is necessary because while the *uses* of slots within each model
    // have been offset, the ContextCreateOp itself still only reflects the
    // local number of slots originally required by that model. Updating it
    // ensures the runtime context is allocated with sufficient space for all
    // globally assigned slots.
    moduleOp.walk([&](ContextCreateOp contextCreateOp) {
      contextCreateOp.setNumSlots(globalMaxSlot);
    });
  }
};

} // namespace
