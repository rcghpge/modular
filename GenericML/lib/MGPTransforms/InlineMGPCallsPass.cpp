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
// InlineMGPCallsPass
//===----------------------------------------------------------------------===//

struct InlineMGPCallsPass
    : public impl::InlineMGPCallsPassBase<InlineMGPCallsPass> {
  using InlineMGPCallsPassBase::InlineMGPCallsPassBase;

  // Set of models whose inits have been inlined.
  DenseSet<StringRef> inlinedInits;

  LogicalResult inlineSubgraph(CallOp callOp) {
    ModuleOp moduleOp = getOperation();
    ModelOp subgraph = moduleOp.lookupSymbol<ModelOp>(callOp.getModel());
    ModelOp parentModelOp = callOp->getParentOfType<ModelOp>();

    // Find the target subgraph model
    if (!subgraph)
      return callOp.emitError()
             << "failed to find symbol @" << callOp.getModel();

    // Check if we need to inline init
    StringRef modelName = callOp.getModel().getRootReference();
    if (!inlinedInits.contains(modelName)) {
      inlinedInits.insert(modelName);

      // Get the init blocks
      Block *parentInit = parentModelOp.getInitBlock();
      Block *subgraphInit = subgraph.getInitBlock();

      // Set up builder for init region
      OpBuilder initBuilder(
          &parentInit->back()); // Insert before the terminator

      // Create mapping for init args
      IRMapping initMap;

      // Map context. Assume there is only one mgp.context.create op per model.
      for (auto op : parentInit->getOps<ContextCreateOp>()) {
        for (auto subOp : subgraphInit->getOps<ContextCreateOp>()) {
          initMap.map(subOp.getResult(), op.getResult());
          break;
        }
        break;
      }

      // Map weights registry
      initMap.map(subgraphInit->getArgument(0), parentInit->getArgument(0));

      // Map devices
      for (auto subgraphArg : subgraphInit->getArguments())
        if (isa<DeviceContextType>(subgraphArg.getType()))
          for (auto parentArg : parentInit->getArguments())
            if (subgraphArg.getType() == parentArg.getType())
              initMap.map(subgraphArg, parentArg);

      // Check if any devices were left unmapped and emit an error if so.
      for (auto subgraphArg : subgraphInit->getArguments())
        if (isa<DeviceContextType>(subgraphArg.getType()))
          if (!initMap.contains(subgraphArg))
            return subgraph.emitError()
                   << "failed to map device context for argument "
                   << subgraphArg.getArgNumber();

      // Clone operations from subgraph's init block, excluding context.create
      // and terminator
      for (Operation &subOp : subgraphInit->getOperations()) {
        // Skip the context.create op as parent already has one
        if (isa<ContextCreateOp>(subOp))
          continue;

        // Ensure subgraph init output chain is waited upon by the parent's init
        // terminator, but don't clone the op.
        if (isa<MEF::OutputOp>(subOp)) {
          Value subgraphOutputChain = initMap.lookup(subOp.getOperand(0));
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

        initBuilder.clone(subOp, initMap);
      }
    }

    // Now inline the execute region at the call site
    // Get the execute blocks
    Block *subgraphExec = subgraph.getExecuteBlock();

    // Create builder for execute region
    OpBuilder execBuilder(callOp);

    // Set up mapping for the arguments
    IRMapping valueMap;

    // Map the call operands to the subgraph execute arguments
    for (unsigned i = 0; i < callOp.getOperands().size(); ++i)
      valueMap.map(subgraphExec->getArgument(i), callOp.getOperand(i));

    // Clone operations from the subgraph's execute block, excluding the
    // terminator
    for (auto &subOp : llvm::make_range(subgraphExec->begin(),
                                        std::prev(subgraphExec->end()))) {
      // Clone the operation with remapped operands
      Operation *clonedOp = execBuilder.clone(subOp, valueMap);

      // Add mappings for the results
      for (unsigned i = 0; i < subOp.getNumResults(); ++i)
        valueMap.map(subOp.getResult(i), clonedOp->getResult(i));
    }

    // Map the return values of the subgraph to the results of the call
    auto subgraphTerminator =
        cast<MEF::OutputOp>(subgraphExec->getTerminator());
    for (unsigned i = 0; i < callOp.getNumResults(); ++i) {
      Value mappedValue = valueMap.lookup(subgraphTerminator.getOperand(i));
      callOp.getResult(i).replaceAllUsesWith(mappedValue);
    }

    // Erase the call operation
    callOp.erase();
    return success();
  }

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    SymbolTable symtab(moduleOp);

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
      ModelOp modelOp = node->func;
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
