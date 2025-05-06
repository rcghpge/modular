//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/MODialect/MODialect.h"
#include "GenericML/GraphCompiler/MODialect/MOOps.h"
#include "GenericML/GraphCompiler/Transforms/MODialect/MOTransforms.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <KGEN/KGENDialect/KGENOps.h>

#define DEBUG_TYPE "hoist-constant-subgraphs-into-parent"

using namespace mlir;

namespace M::MO {
#define GEN_PASS_DEF_HOISTCONSTANTSUBGRAPHSINTOPARENT
#include "GenericML/GraphCompiler/Transforms/MODialect/MOTransforms.h.inc"
} // namespace M::MO

using namespace M;
using namespace M::MO;

namespace {

//===----------------------------------------------------------------------===//
// HoistConstantSubgraphsIntoParent Pass
//===----------------------------------------------------------------------===//
//
// This pass optimizes graphs by hoisting constant operations from
// subgraphs into their parent graphs, enabling optimizations like
// ExecInvariantCodeMotion to hoist them into the init block. For each subgraph
// called with constant arguments, it analyzes which operations in the subgraph
// are constant-derived (operations whose results only depend on constants), and
// hoists these constant operations into the parent at each call site.

//
// The pass performs the following steps:
// 1. Build a call graph of all subgraphs in the module
// 2. Topologically sort the subgraphs (callees before callers) for processing
// 3. For each subgraph, identify which arguments are constant across all
// callsites
// 4. Perform dataflow analysis to find operations with constant-derived results
// 5. Find boundary values between constant and non-constant parts of the graph
// 6. Hoist constant operations to parent graphs at each callsite
// 7. Update subgraph definitions by:
//    - Removing constant arguments
//    - Removing constant operations
//    - Adding new arguments for boundary values
//
// Note: The pass only operates on the acyclic portion of the call graph, and
// ignores any cycles or disconnected components.
//===----------------------------------------------------------------------===//

// Lattice for the data-flow analysis. Each value can be in one of two states.
enum class ConstLattice {
  NonConst, // Not constant-derived
  Const     // Constant-derived
};

// Helper that determines if a value is constant-derived in its containing
// block. Used for analyzing values in the parent that are passed to callsites.
bool isConstantDerived(Value value) {
  Operation *definingOp = value.getDefiningOp();
  if (!definingOp)
    return false;

  // Base case: if the value is directly a constant
  if (isa<MO::ConstantLike>(definingOp))
    return true;

  // Recursive case: check if all operands are constant-derived
  // If any operand is not constant-derived, this op isn't either
  return llvm::all_of(definingOp->getOperands(), isConstantDerived);
}

struct HoistConstantSubgraphsIntoParentPass
    : public M::MO::impl::HoistConstantSubgraphsIntoParentBase<
          HoistConstantSubgraphsIntoParentPass> {
  using HoistConstantSubgraphsIntoParentBase::
      HoistConstantSubgraphsIntoParentBase;

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    LLVM_DEBUG(llvm::dbgs() << "Starting pass\n");

    // Maps to store call graph information
    llvm::DenseMap<GraphOp, SmallVector<CallOp>> graphToCallsites;
    SmallVector<GraphOp> sortedGraphs;

    // STEP 1-2: Build call graph and topologically sort the subgraphs
    buildCallGraphAndSortGraphs(moduleOp, graphToCallsites, sortedGraphs);

    // STEP 3-7: Process graphs and optimize in topological order
    for (GraphOp graphOp : sortedGraphs) {
      auto &callsites = graphToCallsites[graphOp];
      StringRef graphName = graphOp.getName();

      // Skip graphs with no callsites - nothing to optimize
      if (callsites.empty()) {
        LLVM_DEBUG(llvm::dbgs() << "Skipping graph with no callsites: "
                                << graphName << "\n");
        continue;
      }

      LLVM_DEBUG(llvm::dbgs() << "Processing graph: " << graphName << " with "
                              << callsites.size() << " callsites\n");

      processGraph(graphOp, callsites);
    }
  }

private:
  // Build call graph and topologically sort the subgraphs
  void buildCallGraphAndSortGraphs(
      ModuleOp module,
      llvm::DenseMap<GraphOp, SmallVector<CallOp>> &graphToCallsites,
      SmallVector<GraphOp> &sortedGraphs) {

    // Map of symbol names to GraphOps for subgraphs
    llvm::DenseMap<StringRef, GraphOp> symbolToGraphOp;
    // Call graph: graph → graphs it calls (for topological ordering)
    llvm::DenseMap<GraphOp, llvm::SetVector<GraphOp>> callGraph;

    // First collect all subgraphs and build the symbol-to-graph map
    SmallVector<GraphOp> graphs;
    SmallVector<GraphOp> entryPoints;

    for (GraphOp graphOp : module.getOps<GraphOp>()) {
      StringRef name = graphOp.getName();
      symbolToGraphOp[name] = graphOp;
      graphs.push_back(graphOp);

      // Also collect entry points
      if (!graphOp.getIsSubgraph())
        entryPoints.push_back(graphOp);

      LLVM_DEBUG(llvm::dbgs() << "Found graph: " << name << "\n");
    }

    // Then find all call sites by walking the module
    module.walk([&](CallOp callOp) {
      StringRef calleeName = callOp.getSymbol().getLeafReference();
      auto it = symbolToGraphOp.find(calleeName);
      if (it == symbolToGraphOp.end()) {
        callOp.emitError() << "Could not find callee graph named '"
                           << calleeName << "'";
        signalPassFailure();
        return;
      }

      GraphOp callee = it->second;
      graphToCallsites[callee].push_back(callOp);

      // Find the parent graph if this CallOp is inside a graph
      GraphOp parentGraph = callOp->getParentOfType<GraphOp>();
      if (!parentGraph) {
        callOp.emitError() << "Call op must be contained within a GraphOp";
        signalPassFailure();
        return;
      }

      callGraph[parentGraph].insert(callee);

      LLVM_DEBUG(llvm::dbgs() << "Found callsite for graph: " << calleeName
                              << " at " << callOp.getLoc() << "\n");
    });

    LLVM_DEBUG({
      llvm::dbgs() << "Entry points in call graph:\n";
      for (GraphOp entry : entryPoints)
        llvm::dbgs() << "  " << entry.getName() << "\n";
    });

    // Simple DFS to build topological order from entry points
    llvm::DenseSet<GraphOp> visited;

    std::function<void(GraphOp)> dfs = [&](GraphOp node) {
      if (!visited.insert(node).second)
        return;

      // Visit all callees first
      if (auto it = callGraph.find(node); it != callGraph.end())
        llvm::for_each(it->second, dfs);

      sortedGraphs.push_back(node);
    };

    // Process each entry point
    for (GraphOp entry : entryPoints)
      dfs(entry);

    // Reverse to get bottom-up order (callees before callers)
    std::reverse(sortedGraphs.begin(), sortedGraphs.end());

    LLVM_DEBUG({
      llvm::dbgs() << "Processing graphs in order:\n";
      for (GraphOp g : sortedGraphs)
        llvm::dbgs() << "  " << g.getName() << "\n";
    });
  }

  // Main function to process a single graph
  void processGraph(GraphOp graphOp, ArrayRef<CallOp> callsites) {
    unsigned numArgs = graphOp.getBody().getArguments().size();
    if (numArgs == 0) {
      LLVM_DEBUG(llvm::dbgs() << "Graph has no arguments, skipping\n");
      return;
    }

    // STEP 3: Identify which arguments are constant across all callsites
    llvm::BitVector constantMask =
        identifyConstantArguments(graphOp, callsites);

    // Skip if no constant arguments
    if (!constantMask.any()) {
      LLVM_DEBUG(llvm::dbgs() << "No constant arguments found for "
                              << graphOp.getName() << "\n");
      return;
    }

    // STEP 4: Perform dataflow analysis
    llvm::DenseMap<Value, ConstLattice> lattice;
    performDataflowAnalysis(graphOp, constantMask, lattice);

    // STEP 5: Find constant operations and boundary values
    llvm::SmallSetVector<Operation *, 16> constantOps;
    SetVector<Value> boundaryValues;
    llvm::BitVector preservedArgs = constantMask;
    preservedArgs.flip(); // Invert constantMask: true for non-constant args
    SmallVector<Operation *> sortedConstantOps;

    findConstantOpsAndBoundaryValues(graphOp, lattice, constantMask,
                                     constantOps, boundaryValues, preservedArgs,
                                     sortedConstantOps);

    // Skip if no boundary values - nothing to hoist
    if (boundaryValues.empty()) {
      LLVM_DEBUG(llvm::dbgs() << "No boundary values to hoist for "
                              << graphOp.getName() << "\n");
      return;
    }

    // STEP 6: Hoist constant operations to parent at each callsite
    hoistConstantOperations(graphOp, callsites, preservedArgs, constantMask,
                            constantOps, boundaryValues, sortedConstantOps);

    // STEP 7: Update the subgraph definition
    updateSubgraphDefinition(graphOp, preservedArgs, boundaryValues,
                             sortedConstantOps);
  }

  // Identify which arguments are constant across all callsites
  llvm::BitVector identifyConstantArguments(GraphOp graphOp,
                                            ArrayRef<CallOp> callsites) {

    unsigned numArgs = graphOp.getBody().getArguments().size();
    llvm::BitVector constantMask(numArgs,
                                 true); // Start assuming all args are constant

    // For each argument position, check if it's a constant across all callsites
    for (unsigned i = 0; i < numArgs; ++i) {
      for (CallOp callOp : callsites) {
        Value arg = callOp.getOperand(i);
        if (!isConstantDerived(arg)) {
          constantMask[i] = false;
          LLVM_DEBUG(llvm::dbgs()
                     << "Arg " << i << " is not constant in callsite at "
                     << callOp.getLoc() << "\n");
          break;
        }
      }
    }

    LLVM_DEBUG({
      llvm::dbgs() << "Constant arguments: ";
      for (unsigned i = 0; i < numArgs; ++i) {
        if (constantMask[i])
          llvm::dbgs() << i << " ";
      }
      llvm::dbgs() << "\n";
    });

    return constantMask;
  }

  // Perform dataflow analysis to find constant-derived operations
  void performDataflowAnalysis(GraphOp graphOp,
                               const llvm::BitVector &constantMask,
                               llvm::DenseMap<Value, ConstLattice> &lattice) {

    Block *body = graphOp.getBodyBlock();
    SmallVector<Operation *> worklist;

    // Initialize lattice states for block arguments
    for (auto [idx, arg] : llvm::enumerate(body->getArguments())) {
      lattice[arg] =
          constantMask[idx] ? ConstLattice::Const : ConstLattice::NonConst;
    }

    // Initialize worklist with all operations in the block
    for (Operation &op : body->without_terminator())
      worklist.push_back(&op);

    // Process the worklist until fixed point
    while (!worklist.empty()) {
      Operation *op = worklist.back();
      worklist.pop_back();

      // Determine the result state for this operation
      ConstLattice newState = isa<MO::ConstantLike>(op)
                                  ? ConstLattice::Const
                                  : ConstLattice::NonConst;

      // If not a constant-like op, check if it's constant-derived from its
      // operands
      if (newState == ConstLattice::NonConst) {
        bool allOperandsConst = true;
        for (Value operand : op->getOperands()) {
          auto it = lattice.find(operand);
          if (it == lattice.end() || it->second != ConstLattice::Const) {
            allOperandsConst = false;
            break;
          }
        }

        if (allOperandsConst)
          newState = ConstLattice::Const;
      }

      // Update lattice states for all results
      bool changed = false;
      for (Value result : op->getResults()) {
        auto it = lattice.find(result);
        if (it == lattice.end() || it->second != newState) {
          lattice[result] = newState;
          changed = true;
        }
      }

      // If any result state changed, add all users to the worklist
      if (changed) {
        for (Value result : op->getResults()) {
          for (Operation *user : result.getUsers()) {
            if (!user->hasTrait<OpTrait::IsTerminator>() &&
                !llvm::is_contained(worklist, user)) {
              worklist.push_back(user);
            }
          }
        }
      }
    }
  }

  // Find constant operations and boundary values
  void findConstantOpsAndBoundaryValues(
      GraphOp graphOp, const llvm::DenseMap<Value, ConstLattice> &lattice,
      const llvm::BitVector &constantMask,
      llvm::SmallSetVector<Operation *, 16> &constantOps,
      SetVector<Value> &boundaryValues, llvm::BitVector &preservedArgs,
      SmallVector<Operation *> &sortedConstantOps) {

    Block *body = graphOp.getBodyBlock();

    // Identify operations with all constant results
    for (Operation &op : *body) {
      if (op.hasTrait<OpTrait::IsTerminator>())
        continue;

      // Check if all results are constant
      bool allResultsConst = llvm::all_of(op.getResults(), [&](Value result) {
        auto it = lattice.find(result);
        return it != lattice.end() && it->second == ConstLattice::Const;
      });

      if (allResultsConst)
        constantOps.insert(&op);
    }

    // Determine which constant arguments are used by non-constant operations
    for (unsigned i : constantMask.set_bits()) {
      BlockArgument arg = body->getArgument(i);

      // Check if this constant argument is used by any non-constant operation
      for (Operation *user : arg.getUsers()) {
        if (!constantOps.contains(user)) {
          // This argument is used by a non-constant operation, so we need to
          // preserve it
          preservedArgs[i] = true;
          LLVM_DEBUG(llvm::dbgs()
                     << "Arg " << i
                     << " is constant but used by non-constant op: " << *user
                     << ", keeping it\n");
          break;
        }
      }
    }

    LLVM_DEBUG({
      llvm::dbgs() << "Arguments to preserve: ";
      for (unsigned i : preservedArgs.set_bits())
        llvm::dbgs() << i << " ";
      llvm::dbgs() << "\n";
    });

    // Sort constantOps in topological order
    sortedConstantOps.reserve(constantOps.size());
    llvm::DenseSet<Operation *> visited;

    std::function<void(Operation *)> visitOp = [&](Operation *op) {
      if (visited.contains(op))
        return;

      // Visit operand definitions first (if they are constant ops)
      for (Value operand : op->getOperands()) {
        if (Operation *defOp = operand.getDefiningOp()) {
          if (constantOps.contains(defOp))
            visitOp(defOp);
        }
      }

      // After visiting all dependencies, add this op
      visited.insert(op);
      sortedConstantOps.push_back(op);
    };

    // Visit all constant ops to build topological ordering
    for (Operation *op : constantOps) {
      if (!visited.contains(op))
        visitOp(op);
    }

    LLVM_DEBUG({
      llvm::dbgs() << "Constant ops in topological order:\n";
      for (Operation *op : sortedConstantOps)
        llvm::dbgs() << "  " << *op << "\n";
    });

    // Find boundary values - constant values used by non-constant ops
    for (Operation *op : constantOps) {
      for (Value result : op->getResults()) {
        for (Operation *user : result.getUsers()) {
          if (!constantOps.contains(user) &&
              !user->hasTrait<OpTrait::IsTerminator>()) {
            boundaryValues.insert(result);
            break;
          }
        }
      }
    }

    LLVM_DEBUG({
      llvm::dbgs() << "Found " << constantOps.size() << " constant ops and "
                   << boundaryValues.size() << " boundary values\n";

      llvm::dbgs() << "Boundary values: ";
      for (Value boundaryVal : boundaryValues)
        llvm::dbgs() << boundaryVal << " ";

      llvm::dbgs() << "\n";
    });
  }

  // Hoist constant operations to parent at each callsite
  void hoistConstantOperations(
      GraphOp graphOp, ArrayRef<CallOp> callsites,
      const llvm::BitVector &preservedArgs, const llvm::BitVector &constantMask,
      const llvm::SmallSetVector<Operation *, 16> &constantOps,
      const SetVector<Value> &boundaryValues,
      const SmallVector<Operation *> &sortedConstantOps) {

    Block *body = graphOp.getBodyBlock();
    unsigned numArgs = body->getArguments().size();

    for (CallOp callOp : callsites) {
      OpBuilder builder(callOp);
      IRMapping valueMapping;

      // Map constant arguments
      for (unsigned i = 0; i < numArgs; ++i)
        if (constantMask[i])
          valueMapping.map(body->getArgument(i), callOp.getOperand(i));

      // Clone constant-derived ops in topological order
      for (Operation *op : sortedConstantOps) {
        Operation *clonedOp = builder.clone(*op, valueMapping);

        // Special case for param decls - we need to unique the name
        if (auto paramDecl = dyn_cast<KGEN::ParamDeclareOp>(*op)) {
          // Get the parameter declaration attribute
          KGEN::ParamDeclAttr decl = paramDecl.getParamDecl();
          StringRef origName = decl.getName();

          // Find the GraphOp parent to help with name uniquing
          auto parentGraphOp = builder.getInsertionBlock()
                                   ->getParentOp()
                                   ->getParentOfType<MO::GraphOp>();

          assert(parentGraphOp && "expected parent graph op");

          // Create a new unique parameter name
          std::string uniqueName = parentGraphOp.getNewParamName(origName);

          // Create a new parameter declaration with the unique name and same
          // type
          auto newDecl = KGEN::ParamDeclAttr::get(uniqueName, decl.getType());

          // Replace the cloned operation with a new one that uses the unique
          // name
          auto newParamOp = builder.create<KGEN::ParamDeclareOp>(
              clonedOp->getLoc(), newDecl,
              cast<KGEN::ParamDeclareOp>(clonedOp).getValue());

          // Map the results appropriately
          for (unsigned i = 0; i < clonedOp->getNumResults(); ++i)
            valueMapping.map(op->getResult(i), newParamOp->getResult(i));

          // Erase the original cloned operation since we've replaced it
          clonedOp->erase();

          // Use the new operation for subsequent operations
          clonedOp = newParamOp;
        }

        // Map the results for use by subsequent ops
        for (unsigned i = 0; i < op->getNumResults(); ++i)
          valueMapping.map(op->getResult(i), clonedOp->getResult(i));
      }

      // Prepare new operands for the updated call
      SmallVector<Value> newCallOperands;

      // Add preserved arguments
      for (unsigned i : preservedArgs.set_bits())
        newCallOperands.push_back(callOp.getOperand(i));

      // Add boundary values as new arguments
      for (Value boundaryVal : boundaryValues) {
        Value mappedVal = valueMapping.lookup(boundaryVal);
        newCallOperands.push_back(mappedVal);
      }

      // Create new call with updated operands
      auto newCallOp =
          builder.create<CallOp>(callOp.getLoc(), callOp.getResultTypes(),
                                 newCallOperands, callOp.getSymbol());

      // Replace uses of original call results
      callOp.replaceAllUsesWith(newCallOp.getResults());

      LLVM_DEBUG(llvm::dbgs() << "Replaced call at " << callOp.getLoc()
                              << " with new call\n");

      // Erase the original call
      callOp.erase();
    }
  }

  // Update subgraph definition by removing constants and adding new arguments
  void
  updateSubgraphDefinition(GraphOp graphOp,
                           const llvm::BitVector &preservedArgs,
                           const SetVector<Value> &boundaryValues,
                           const SmallVector<Operation *> &sortedConstantOps) {

    Block *body = graphOp.getBodyBlock();
    unsigned numArgs = body->getArguments().size();

    // Get the list of arguments to delete (in reverse order for safe erasure)
    SmallVector<unsigned> argsToDelete;
    for (int i = numArgs - 1; i >= 0; --i)
      if (!preservedArgs[i])
        argsToDelete.push_back(i);

    // Update boundary values to use new block arguments
    for (Value boundaryVal : boundaryValues) {
      BlockArgument newArg =
          body->addArgument(boundaryVal.getType(), graphOp.getLoc());
      boundaryVal.replaceAllUsesWith(newArg);
      LLVM_DEBUG(llvm::dbgs()
                 << "Replaced boundary value with new arg " << newArg << "\n");
    }

    // Erase constant operations (in reverse topological order)
    for (auto it = sortedConstantOps.rbegin(); it != sortedConstantOps.rend();
         ++it) {
      LLVM_DEBUG(llvm::dbgs() << "Erasing constant op: " << **it << "\n");
      (*it)->erase();
    }

    // Erase non-preserved arguments
    for (unsigned idx : argsToDelete) {
      LLVM_DEBUG(llvm::dbgs() << "Erasing non-preserved arg: " << idx << "\n");
      body->eraseArgument(idx);
    }

    // Update function type and signature
    synchronizeMetadataUpdates(graphOp);
  }
};

} // namespace
