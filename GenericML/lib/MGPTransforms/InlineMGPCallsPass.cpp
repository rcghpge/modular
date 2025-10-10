//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "AsyncRT/CompilerSupport/Context.h"
#include "GenericML/MGPDialect/MGPBufferLifetimeAnalysis.h"
#include "GenericML/MGPDialect/MGPOps.h"
#include "GenericML/MGPTransforms/MGPTransforms.h"
#include "KGEN/TransformUtils/CallGraphUtils.h"
#include "KGEN/TransformUtils/SCCUtils.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"

#include "GenericML/MGPDialect/BufferBuilder.h"
#include "GenericML/MGPDialect/MGPTensorInterface.h"
#include "GenericML/MGPTransforms/Utils.h"

#define DEBUG_TYPE "inline-mgp-calls"

namespace M::MGP {

#define GEN_PASS_DEF_INLINEMGPCALLSPASS
#include "GenericML/MGPTransforms/MGPTransforms.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Slot offsetting helper functions
//===----------------------------------------------------------------------===//

/// Offsets the specific slot index attribute of various MGP operations.
void offsetRuntimeSlot(Operation *op, uint64_t slotOffset) {
  op->walk([&](Operation *op) {
    if (auto bufferCacheOp = dyn_cast<BufferCacheOp>(op)) {
      uint64_t oldSlot = bufferCacheOp.getBBufferSlot();
      bufferCacheOp.setBBufferSlot(oldSlot + slotOffset);
    } else if (auto runtimeCreateOp = dyn_cast<RuntimeCreateOp>(op)) {
      uint64_t oldSlot = runtimeCreateOp.getARuntimeSlot();
      runtimeCreateOp.setARuntimeSlot(oldSlot + slotOffset);
    } else if (auto bufferRemoveCachedOp = dyn_cast<BufferRemoveCachedOp>(op)) {
      uint64_t oldSlot = bufferRemoveCachedOp.getBufferSlot();
      bufferRemoveCachedOp.setBufferSlot(oldSlot + slotOffset);
    } else if (auto moduleLoadOp = dyn_cast<ModuleLoadOp>(op)) {
      uint64_t oldSlot = moduleLoadOp.getARuntimeSlot();
      moduleLoadOp.setARuntimeSlot(oldSlot + slotOffset);
    } else if (auto executeGenericOp = dyn_cast<ExecuteGenericOp>(op)) {
      uint64_t oldSlot = executeGenericOp.getARuntimeSlot();
      executeGenericOp.setARuntimeSlot(oldSlot + slotOffset);
    }
  });
}

/// Updates the name attribute of mgp.buffer.constant.external operations
/// that have placeholder=true by prepending the given prefix.
void updatePlaceholderNames(Operation *op, StringRef prefix) {
  if (auto bufferConstExtOp = dyn_cast<BufferConstantExternalOp>(op)) {
    if (bufferConstExtOp.getHIsPlaceholder() && !prefix.empty()) {
      bufferConstExtOp.setBNameAttr(StringAttr::get(
          op->getContext(), Twine(prefix) + bufferConstExtOp.getBName()));
      bufferConstExtOp.setHIsPlaceholder(false);
    }
  }
}

} // namespace

//===----------------------------------------------------------------------===//
// Call Graph Definition for MGP Models
//===----------------------------------------------------------------------===//

struct MGPCallGraphNode
    : public KGEN::SCCNode<MGPCallGraphNode, ModelOp, CallOp> {
  using SCCNode::SCCNode;

  LogicalResult inlineSubgraph(SCCNode::EdgeT call,
                               uint64_t &modelSlotUsage) const;

  bool visited = false;

  ContextCreateOp contextCreateOp;
  uint64_t slotsUsed = 0;

  /// Number of times the current node was inlined.
  /// This is used to determine if the node was an inlining candidate and can
  /// be deleted from the graph after inlining.
  uint64_t numTimesInlined = 0;
};

LogicalResult MGPCallGraphNode::inlineSubgraph(MGPCallGraphNode::EdgeT edge,
                                               uint64_t &modelSlotUsage) const {
  // High-level steps for inlining a subgraph:
  // 1. Prepend call prefix to unique external tensors (placeholder names)
  // 2. Record initial slot usage in parent model
  // 3. Offset slots in subgraph so parent mgp.context machinery is correct
  // 4. Clone init block operations from subgraph into parent init
  // 5. Clone execute block operations from subgraph into parent at call site
  // 6. Map return values and clean up the call operation

  ModelOp subgraph = edge.node->func;
  ModelOp parentModelOp = this->func;
  CallOp callOp = edge.call;

  // Find the target subgraph model
  if (!subgraph)
    return callOp.emitError()
           << "failed to find symbol @" << callOp.getModelAttrName();

  // Get the prefix from the call operation
  StringRef callPrefix = callOp.getPrefix();

  // Find the parent's context (needed for both slot tracking and mapping)
  ContextCreateOp parentContextCreateOp = parentModelOp.getInitContext();

  // Find the number of slots this specific subgraph requires
  ContextCreateOp subgraphContextCreateOp = subgraph.getInitContext();

  uint64_t subgraphSlotsUsed = edge.node->slotsUsed;

  // The offset for this callsite should start after the parent's current
  // slots
  uint64_t slotOffset = modelSlotUsage;

  // First, inline the init block for this specific callsite
  Block *parentInit = parentModelOp.getInitBlock();

  // Set up builder for init region
  OpBuilder initBuilder(&parentInit->back()); // Insert before the terminator

  // Create mapping for init args
  IRMapping initMap;

  // Map context. Assume there is only one mgp.context.create op per model.
  if (parentContextCreateOp && subgraphContextCreateOp)
    initMap.map(subgraphContextCreateOp.getResult(),
                parentContextCreateOp.getResult());

  // Map weights registry
  initMap.map(subgraph.getInitBlock()->getArgument(0),
              parentInit->getArgument(0));

  // Map devices
  for (auto subgraphArg : subgraph.getInitBlock()->getArguments())
    if (isa<DeviceContextType>(subgraphArg.getType()))
      for (auto parentArg : parentInit->getArguments())
        if (subgraphArg.getType() == parentArg.getType())
          initMap.map(subgraphArg, parentArg);

  // Check if any devices were left unmapped and emit an error if so.
  for (auto subgraphArg : subgraph.getInitBlock()->getArguments())
    if (isa<DeviceContextType>(subgraphArg.getType()))
      if (!initMap.contains(subgraphArg))
        return subgraph.emitError()
               << "failed to map device context for argument "
               << subgraphArg.getArgNumber();

  // Clone operations from subgraph's init block, excluding context.create
  // and terminator, and apply slot offsetting and prefix handling
  for (Operation *subOp : getDataflowOrder(subgraph.getInitBlock())) {
    // Skip the context.create op as parent already has one
    if (isa<ContextCreateOp>(subOp))
      continue;

    // Ensure subgraph init output chain is waited upon by the parent's init
    // terminator, but don't clone the op.
    if (isa<MEF::OutputOp>(subOp)) {
      Value subgraphOutputChain = initMap.lookup(subOp->getOperand(0));
      assert(subgraphOutputChain &&
             "Subgraph init block is missing an output chain");

      OpOperand &parentOutputChain =
          parentInit->getTerminator()->getOpOperand(0);
      auto newInit = initBuilder.create<GRT::ChainCreateOp>(
          parentInit->getTerminator()->getLoc(),
          ValueRange{parentOutputChain.get(), subgraphOutputChain});

      parentOutputChain.set(newInit);
      continue;
    }

    // Clone the operation
    Operation *clonedOp = initBuilder.clone(*subOp, initMap);

    // Apply slot offsetting to the cloned operation
    offsetRuntimeSlot(clonedOp, slotOffset);

    // Update placeholder names with prefix
    updatePlaceholderNames(clonedOp, callPrefix);
  }

  // Now inline the execute region at the call site
  Block *subgraphExec = subgraph.getExecuteBlock();

  // Create builder for execute region
  OpBuilder execBuilder(callOp);

  // Set up mapping for the arguments
  IRMapping valueMap;

  // Map the call operands to the subgraph execute arguments
  valueMap.map(subgraphExec->getArguments(), callOp.getOperands());

  // Clone operations from the subgraph's execute block, excluding the
  // terminator, and apply slot offsetting and prefix handling
  for (Operation *subOp : getDataflowOrder(subgraphExec)) {
    if (subOp == subgraphExec->getTerminator())
      continue;

    // Clone the operation with remapped operands
    Operation *clonedOp = execBuilder.clone(*subOp, valueMap);

    // Apply slot offsetting to the cloned operation
    offsetRuntimeSlot(clonedOp, slotOffset);

    // Update placeholder names with prefix
    updatePlaceholderNames(clonedOp, callPrefix);
  }

  // Map the return values of the subgraph to the results of the call
  auto subgraphTerminator = cast<MEF::OutputOp>(subgraphExec->getTerminator());

  auto newResults =
      llvm::map_range(subgraphTerminator.getOperands(),
                      [&](Value v) { return valueMap.lookup(v); });

  callOp.replaceAllUsesWith(newResults);

  // Update the parent's slot usage to include this inlined subgraph
  modelSlotUsage += subgraphSlotsUsed;

  // Erase the call operation
  callOp.erase();
  return success();
}

struct MGPCallGraph : public KGEN::SCCGraph<MGPCallGraph, MGPCallGraphNode,
                                            /*BottomUpRewrite=*/true> {
  using Base = KGEN::SCCGraph<MGPCallGraph, MGPCallGraphNode, true>;
  using Base::Base;

  bool doAnalysis(MGPCallGraphNode *node);
  void doRewrite(const MGPCallGraphNode *node);

  std::atomic<int64_t> numErrors{0};
};

bool MGPCallGraph::doAnalysis(MGPCallGraphNode *node) {
  if (node->visited)
    return false;

  // Error checking.
  if (node->scc->hasRecursion) {
    auto scc = node->scc;

    ModuleOp module = node->func.getParentOp();

    mlir::InFlightDiagnostic diag =
        module.emitError("Cycle detected in MGP call graph, cannot inline");
    diag << ". Involved models: ";
    llvm::interleaveComma(scc->nodes, diag, [&](MGPCallGraphNode *node) {
      diag << "@" << node->func.getSymName();
      // Emit a note attached to the specific ModelOp involved in the cycle.
      node->func->emitError() << "Model involved in the call graph cycle.";
    });

    numErrors++;
  }

  // Capture the context create operation and the number of context slots used
  // by the model.
  {
    node->contextCreateOp = node->func.getInitContext();
    if (!node->contextCreateOp) {
      node->func.emitError("unable to find context");
      numErrors++;
    } else {
      node->slotsUsed = node->contextCreateOp.getNumSlots();
    }
  }

  node->visited = true;
  return true;
}

void MGPCallGraph::doRewrite(const MGPCallGraphNode *node) {
  // If we've encountered errors during analysis or rewriting of child nodes,
  // do not bother processing.
  if (numErrors > 0)
    return;

  uint64_t modelSlotsUsage = node->slotsUsed;
  for (EdgeT edge : node->callsites) {
    if (failed(node->inlineSubgraph(edge, modelSlotsUsage))) {
      edge.call->emitError("unable to inline model call operation");
      numErrors++;
      return;
    }

    edge.node->numTimesInlined++;
  }

  // Update the model slots used for the current model.
  ContextCreateOp op = node->contextCreateOp;
  op.setNumSlots(modelSlotsUsage);
}

} // namespace M::MGP

using namespace M;
using namespace M::MGP;

namespace {

//===----------------------------------------------------------------------===//
// InlineMGPCallsPass
//===----------------------------------------------------------------------===//

struct InlineMGPCallsPass
    : public impl::InlineMGPCallsPassBase<InlineMGPCallsPass> {
  using InlineMGPCallsPassBase::InlineMGPCallsPassBase;

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    // No work to do if there's only one model.
    auto models = moduleOp.getOps<ModelOp>();
    if (models.empty() || llvm::hasSingleElement(models))
      return;

    AsyncRT::Runtime &runtime =
        *loadContext(&getContext())->get<AsyncRT::Runtime>();

    SymbolTable symtab(moduleOp);

    MGPCallGraph callGraph;
    callGraph.build(moduleOp, symtab);
    callGraph.run(runtime);

    if (callGraph.numErrors > 0)
      return signalPassFailure();

    // After inlining everything, delete subgraphs that are no longer used.
    // Iterate using the original list of functions in the module before
    // erasing.
    for (auto &[model, node] : callGraph.nodes) {
      // Only consider erasing models that were targets of inlining.
      if (node.numTimesInlined == 0)
        continue;

      // Check if this subgraph has any remaining uses. If it's marked
      // isSubgraph and has no uses, erase it.
      if (SymbolTable::symbolKnownUseEmpty(model, moduleOp) &&
          model.getIsSubgraph()) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Erasing inlined model @" << model.getSymName() << "\n");
        model.erase();
      } else {
        // This might happen if the model is still called from outside the
        // current set of models being processed or if it's not marked
        // isSubgraph.
        LLVM_DEBUG(
            llvm::dbgs()
            << "Model @" << model.getSymName()
            << " still has uses or is public after inlining, not erasing.\n");
      }
    }
  }
};

} // namespace
