//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/MGPDialect/MGPOps.h"
#include "GenericML/MGPTransforms/MGPTransforms.h"
#include "KGEN/TransformUtils/CallGraphUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/SCCIterator.h"

#include "GenericML/MGPDialect/BufferBuilder.h"
#include "GenericML/MGPDialect/MGPTensorInterface.h"
#include "GenericML/MGPTransforms/Utils.h"

#define DEBUG_TYPE "inline-mgp-calls"

namespace M::MGP {

#define GEN_PASS_DEF_INLINEMGPCALLSPASS
#include "GenericML/MGPTransforms/MGPTransforms.h.inc"

//===----------------------------------------------------------------------===//
// Call Graph Definition for MGP Models
//===----------------------------------------------------------------------===//

struct MGPCallGraphNode
    : KGEN::CallGraphNodeBase<MGPCallGraphNode, ModelOp, CallOp> {
  explicit MGPCallGraphNode(ModelOp func)
      : KGEN::CallGraphNodeBase<MGPCallGraphNode, ModelOp, CallOp>(func) {}
};

struct MGPCallGraph : KGEN::CallGraphBase<MGPCallGraph, MGPCallGraphNode> {
  using KGEN::CallGraphBase<MGPCallGraph, MGPCallGraphNode>::CallGraphBase;

  explicit MGPCallGraph() : externalNode(nullptr) {}

  // Custom build method for MGP Models, as they don't have a single body
  // region.
  template <typename... NodeArgTs>
  void build(ModuleOp module, const SymbolTable &symtab, NodeArgTs &&...args) {
    for (auto func : llvm::make_early_inc_range(module.getOps<ModelOp>())) {
      nodes.insert(std::make_pair(
          func, MGPCallGraphNode(func, std::forward<NodeArgTs>(args)...)));
    }

    for (auto &[_, node] : nodes) {
      auto callerNode = &node;

      Block *executeBlock = callerNode->func.getExecuteBlock();

      executeBlock->walk([&](CallOp call) {
        Operation *calleeOp = symtab.lookup(call.getModel().getLeafReference());
        assert(calleeOp && "Call target not found in symbol table");

        // Only add the edge if the symbol we found is an MGP Model.
        auto callee = dyn_cast<ModelOp>(calleeOp);
        assert(callee && "Callee is not an MGP model");

        auto it = nodes.find(callee);
        assert(it != nodes.end() && "Callee is not in the call graph");

        MGPCallGraphNode *calleeNode = &it->second;
        assert(calleeNode &&
               "Callee node pointer must not be null before adding edge");

        callerNode->callsites.emplace_back(call, calleeNode);
        LLVM_DEBUG(llvm::dbgs()
                   << "Added callsite @" << call.getModel().getLeafReference()
                   << " to @" << callerNode->func.getSymName() << "\n");

        calleeNode->callers.push_back(callerNode);
        LLVM_DEBUG(llvm::dbgs()
                   << "Added caller @" << callerNode->func.getSymName()
                   << " to callee @" << calleeNode->func.getSymName() << "\n");
      });
    }
  }

  /// Builds the MGP call graph and computes a reverse topological ordering
  /// suitable for inlining. Detects cycles (recursion) in the call graph.
  ///
  /// @param module The ModuleOp containing the MGP models.
  /// @param symTab A symbol table for the module.
  /// @return An ErrorOr containing either a SmallVector of MGPCallGraphNode
  ///         pointers in reverse topological (inlining) order, or an Error if
  ///         a cycle is detected in the call graph.
  ErrorOr<SmallVector<MGPCallGraphNode *>> getInliningOrder(ModuleOp module,
                                                            SymbolTable symTab);

  /// External node that has all the functions that do not have a caller in the
  /// Module as callees. This node is the entry node of the CallGraph for
  /// computing SCCs.
  MGPCallGraphNode externalNode;
};

} // namespace M::MGP

using namespace M;
using namespace MGP;

// Provide GraphTraits specializations for MGPCallGraph to work with llvm
// iterators.
namespace llvm {

template <>
struct GraphTraits<M::MGP::MGPCallGraphNode *> {
  using NodeT = M::MGP::MGPCallGraphNode;
  using NodeRef = NodeT *;

  using ChildIteratorType =
      llvm::mapped_iterator<typename NodeT::EdgeListT::iterator,
                            std::function<NodeT *(typename NodeT::EdgeT)>>;

  static NodeRef getEntryNode(NodeT *node) { return node; }

  static NodeT *edgeToNode(typename NodeT::EdgeT edge) { return edge.node; }

  static ChildIteratorType child_begin(NodeRef node) {
    std::function<NodeT *(typename NodeT::EdgeT)> mapFunc = edgeToNode;
    return ChildIteratorType(node->begin(), mapFunc);
  }
  static ChildIteratorType child_end(NodeRef node) {
    std::function<NodeT *(typename NodeT::EdgeT)> mapFunc = edgeToNode;
    return ChildIteratorType(node->end(), mapFunc);
  }
};

template <>
struct GraphTraits<M::MGP::MGPCallGraph *>
    : public GraphTraits<M::MGP::MGPCallGraph::BaseT *> {};

} // namespace llvm

ErrorOr<SmallVector<MGPCallGraphNode *>>
MGPCallGraph::getInliningOrder(ModuleOp module, SymbolTable symTab) {
  build(module, symTab);

  // Connect externalNode to all nodes with no callers (entry points).
  for (auto &[func, node] : nodes)
    if (node.callers.empty())
      externalNode.callsites.emplace_back(/*call=*/nullptr, &node);

  // Use SCC iterator to get a reverse topological order and detect cycles.
  SmallVector<MGPCallGraphNode *> inliningOrder;
  inliningOrder.reserve(nodes.size());

  // Process SCCs in reverse topological order (bottom-up)
  bool cycleDetected = false;
  for (auto sccIt = llvm::scc_begin(this); sccIt != llvm::scc_end(this);
       ++sccIt) {
    auto scc = *sccIt;

    if (sccIt.hasCycle()) {
      cycleDetected = true;
      mlir::InFlightDiagnostic diag =
          module.emitError("Cycle detected in MGP call graph, cannot inline");
      diag << ". Involved models: ";
      llvm::interleaveComma(scc, diag, [&](MGPCallGraphNode *node) {
        diag << "@" << node->func.getSymName();
        // Emit a note attached to the specific ModelOp involved in the cycle.
        node->func->emitError() << "Model involved in the call graph cycle.";
      });
      continue; // Skip adding cyclic nodes to inlining order
    }

    if (cycleDetected)
      return Error("Cycle detected in MGP call graph, cannot inline");

    // Each SCC should be a single node since we've verified there are no
    // cycles
    assert(scc.size() == 1 && "Expected no cycles in MGP call graph");
    MGPCallGraphNode *node = scc.front();

    // Skip the external node
    if (node == &externalNode)
      continue;

    inliningOrder.push_back(node);
    LLVM_DEBUG(llvm::dbgs() << "Added node @" << node->func.getSymName()
                            << " to inlining order\n");
  }

  return inliningOrder;
}

namespace {

//===----------------------------------------------------------------------===//
// Slot offsetting helper functions
//===----------------------------------------------------------------------===//

/// Offsets the specific slot index attribute of various MGP operations.
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

  // Recurse into regions (for example, grt.if)
  for (Region &region : op->getRegions()) {
    for (Block &block : region) {
      for (Operation &nestedOp : block)
        offsetRuntimeSlot(&nestedOp, slotOffset);
    }
  }
}

/// Updates the name attribute of mgp.buffer.constant.external operations
/// that have placeholder=true by prepending the given prefix.
static void updatePlaceholderNames(Operation *op, StringRef prefix) {
  if (auto bufferConstExtOp = dyn_cast<BufferConstantExternalOp>(op)) {
    if (bufferConstExtOp.getHIsPlaceholder() && !prefix.empty()) {
      bufferConstExtOp.setBNameAttr(StringAttr::get(
          op->getContext(), Twine(prefix) + bufferConstExtOp.getBName()));
      bufferConstExtOp.setHIsPlaceholder(false);
    }
  }
}

/// Finds the ContextCreateOp in the given model's init block with error
/// reporting. Returns the ContextCreateOp if found, emits an error and returns
/// nullptr otherwise. MGP::ModelOp::verify ensures there's only one
/// ContextCreateOp per init block.
static ContextCreateOp findContextCreateOp(ModelOp model) {
  ContextCreateOp contextCreateOp;
  model.getInitBlock()->walk([&](ContextCreateOp op) { contextCreateOp = op; });
  ASSERT_STREAM(contextCreateOp,
                "Model does not contain a ContextCreateOp in its init block.");
  return contextCreateOp;
}

//===----------------------------------------------------------------------===//
// InlineMGPCallsPass
//===----------------------------------------------------------------------===//

struct InlineMGPCallsPass
    : public impl::InlineMGPCallsPassBase<InlineMGPCallsPass> {
  using InlineMGPCallsPassBase::InlineMGPCallsPassBase;

  // Track slot usage per model as we inline into them
  DenseMap<ModelOp, uint64_t> modelSlotUsage;

  LogicalResult inlineSubgraph(CallOp callOp) {
    // High-level steps for inlining a subgraph:
    // 1. Prepend call prefix to unique external tensors (placeholder names)
    // 2. Record initial slot usage in parent model
    // 3. Offset slots in subgraph so parent mgp.context machinery is correct
    // 4. Clone init block operations from subgraph into parent init
    // 5. Clone execute block operations from subgraph into parent at call site
    // 6. Map return values and clean up the call operation

    ModuleOp moduleOp = getOperation();
    ModelOp subgraph = moduleOp.lookupSymbol<ModelOp>(callOp.getModel());
    ModelOp parentModelOp = callOp->getParentOfType<ModelOp>();

    // Find the target subgraph model
    if (!subgraph)
      return callOp.emitError()
             << "failed to find symbol @" << callOp.getModel();

    // Get the prefix from the call operation
    StringRef callPrefix = callOp.getPrefix();

    // Find the parent's context (needed for both slot tracking and mapping)
    ContextCreateOp parentContextCreateOp = findContextCreateOp(parentModelOp);
    if (!parentContextCreateOp)
      return failure();

    // Initialize parent's slot usage if not already tracked
    if (modelSlotUsage.find(parentModelOp) == modelSlotUsage.end())
      modelSlotUsage[parentModelOp] = parentContextCreateOp.getNumSlots();

    // Get the current slot usage of the parent (including any previously
    // inlined subgraphs)
    uint64_t parentCurrentSlotUsage = modelSlotUsage[parentModelOp];

    // Find the number of slots this specific subgraph requires
    ContextCreateOp subgraphContextCreateOp = findContextCreateOp(subgraph);
    if (!subgraphContextCreateOp)
      return failure();

    uint64_t subgraphSlotsUsed = subgraphContextCreateOp.getNumSlots();

    // The offset for this callsite should start after the parent's current
    // slots
    uint64_t slotOffset = parentCurrentSlotUsage;

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

        BufferBuilderContext builderCtx =
            TensorBufferDialectInterface::getAllBufferBuilders(parentModelOp);

        if (failed(alsoWaitForChains(initBuilder, builderCtx,
                                     parentInit->getTerminator(),
                                     {subgraphOutputChain}))) {
          return subgraph.emitError()
                 << "failed to wait for inlined init chain";
        }

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
    for (unsigned i = 0; i < callOp.getOperands().size(); ++i)
      valueMap.map(subgraphExec->getArgument(i), callOp.getOperand(i));

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
    auto subgraphTerminator =
        cast<MEF::OutputOp>(subgraphExec->getTerminator());
    for (unsigned i = 0; i < callOp.getNumResults(); ++i) {
      Value mappedValue = valueMap.lookup(subgraphTerminator.getOperand(i));
      callOp.getResult(i).replaceAllUsesWith(mappedValue);
    }

    // Update the parent's slot usage to include this inlined subgraph
    modelSlotUsage[parentModelOp] += subgraphSlotsUsed;

    // Erase the call operation
    callOp.erase();
    return success();
  }

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    SymbolTable symtab(moduleOp);

    // No work to do if there's only one model.
    auto models = moduleOp.getOps<ModelOp>();
    if (models.empty() || std::next(models.begin()) == models.end())
      return;

    MGPCallGraph callGraph;
    auto inliningOrder = callGraph.getInliningOrder(moduleOp, symtab);

    if (inliningOrder.isError()) {
      // Couldn't find inlining order due to cycle in graph
      signalPassFailure();
      return;
    }

    // Set to track subgraphs that we've inlined
    DenseSet<StringRef> inlinedSubgraphs;

    // Inline functions in the determined reverse topological order.
    for (MGPCallGraphNode *node : inliningOrder.get()) {
      [[maybe_unused]] ModelOp modelOp = node->func;
      // Make a copy of callsites as inlining modifies the operation list.
      SmallVector<KGEN::CallGraphEdgeBase<MGPCallGraphNode, ModelOp, CallOp>>
          callsitesToInline;
      callsitesToInline = node->callsites;

      for (auto &edge : callsitesToInline) {
        CallOp callOp = edge.call;
        // If the call op is null, it's an edge from the synthetic external
        // node, skip it.
        if (!callOp)
          continue;

        StringRef calleeName = edge.node->func.getSymName();
        LLVM_DEBUG(llvm::dbgs() << "Inlining call to @" << calleeName
                                << " into @" << modelOp.getSymName() << "\n");

        if (failed(inlineSubgraph(callOp))) {
          signalPassFailure();
          return;
        }

        // Mark the callee as potentially removable.
        inlinedSubgraphs.insert(calleeName);
      }
    }

    // After inlining everything, update each model's ContextCreateOp to use
    // the correct slot count based on what was actually inlined into it.
    for (const auto &entry : modelSlotUsage) {
      ModelOp model = entry.first;
      uint64_t slotUsage = entry.second;
      ContextCreateOp contextCreateOp = findContextCreateOp(model);
      if (contextCreateOp)
        contextCreateOp.setNumSlots(slotUsage);
    }

    // After inlining everything, delete subgraphs that are no longer used.
    // Iterate using the original list of functions in the module before
    // erasing.
    SmallVector<ModelOp> originalModels;
    for (auto model : moduleOp.getOps<ModelOp>())
      originalModels.push_back(model);

    for (ModelOp subgraph : originalModels) {
      // Only consider erasing models that were targets of inlining.
      if (!inlinedSubgraphs.contains(subgraph.getSymName()))
        continue;

      // Check if this subgraph has any remaining uses. If it's marked
      // isSubgraph and has no uses, erase it.
      if (SymbolTable::symbolKnownUseEmpty(subgraph, moduleOp) &&
          subgraph.getIsSubgraph()) {
        LLVM_DEBUG(llvm::dbgs() << "Erasing inlined model @"
                                << subgraph.getSymName() << "\n");
        subgraph.erase();
      } else {
        // This might happen if the model is still called from outside the
        // current set of models being processed or if it's not marked
        // isSubgraph.
        LLVM_DEBUG(
            llvm::dbgs()
            << "Model @" << subgraph.getSymName()
            << " still has uses or is public after inlining, not erasing.\n");
      }
    }
  }
};

} // namespace
