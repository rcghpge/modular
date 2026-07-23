//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MODialect/Analysis/ParametricTopologyAnalysis.h"
#include "GraphCompiler/MODialect/MODialect.h"
#include "GraphCompiler/MODialect/MOInterfaces.h"
#include "GraphCompiler/MODialect/MOOps.h"
#include "GraphCompiler/MODialect/MOTypes.h"
#include "GraphCompiler/MODialect/Transforms/MOTransforms.h"
#include "KGEN/KGENDialect/AttrTypeMangler.h"
#include "KGEN/KGENDialect/KGENOps.h"

#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "hoist-constant-subgraphs-into-parent"

using namespace mlir;

namespace M::MO {
#define GEN_PASS_DEF_HOISTCONSTANTSUBGRAPHSINTOPARENT
#include "GraphCompiler/MODialect/Transforms/MOTransforms.h.inc"
} // namespace M::MO

using namespace M;
using namespace M::MO;

namespace {

//===----------------------------------------------------------------------===//
// HoistConstantSubgraphsIntoParent Pass
//===----------------------------------------------------------------------===//
//
// This pass optimizes graphs by hoisting callsite-invariant operations from
// subgraphs into their parent graphs, enabling optimizations like
// ExecInvariantCodeMotion to hoist them into the init block. For each subgraph
// called with callsite-invariant arguments (arguments that are the same SSA
// value across all callsites), it analyzes which operations in the subgraph are
// invariant-derived (operations whose results only depend on callsite-invariant
// arguments), and hoists these operations into the parent at each call site.

//
// The pass performs the following steps:
// 1. Build a call graph of all subgraphs in the module
// 2. Topologically sort the subgraphs (callees before callers) for processing
// 3. For each subgraph, identify which arguments are effectively invariant
// across all callsites
// 4. Perform dataflow analysis to find operations with invariant-derived
// results
// 5. Find boundary values between invariant-derived and non-invariant-derived
// parts of the graph
// 6. Hoist invariant-derived operations to parent graphs at each callsite
// 7. Update subgraph definitions by:
//    - Removing callsite-invariant arguments
//    - Removing invariant-derived operations
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

// Container holding the lattice state for every program point tracked by
// our analysis.
struct LatticeState {

  // A program point corresponds to either:
  //
  // 1. A block argument for tracking the 'constness' of graph inputs.
  // 2. An operation which tracks the 'constness' of every value produced by
  //    that operation.
  using ProgramPoint = llvm::PointerUnion<BlockArgument, Operation *>;

  ProgramPoint getKey(Value v) const {
    if (auto arg = dyn_cast<BlockArgument>(v))
      return arg;
    return v.getDefiningOp();
  }

  ConstLattice query(Value v) const {
    return state.lookup_or(getKey(v), ConstLattice::NonConst);
  }

  ConstLattice query(Operation *v) const {
    return state.lookup_or(v, ConstLattice::NonConst);
  }

  void set(Value v, ConstLattice l) { state[getKey(v)] = l; }

  void set(Operation *v, ConstLattice l) { state[v] = l; }

  bool update(Operation *v, ConstLattice l) {
    auto oldValue = query(v);
    if (oldValue != l) {
      set(v, l);
      return true;
    }

    return false;
  }

  llvm::DenseMap<ProgramPoint, ConstLattice> state;
};

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
      [[maybe_unused]] StringRef graphName = graphOp.getName();

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
      StringRef calleeName = callOp.getCallee();
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

    // STEP 3: Identify which arguments are effectively invariant across all
    // callsites
    llvm::BitVector effectiveInvariantArgMask =
        computeEffectiveInvariantArgMask(graphOp, callsites);

    // Skip if no effectively invariant arguments
    if (!effectiveInvariantArgMask.any()) {
      LLVM_DEBUG(llvm::dbgs() << "No effectively invariant arguments found for "
                              << graphOp.getName() << "\n");
      return;
    }

    // STEP 4: Perform dataflow analysis
    LatticeState lattice;
    performDataflowAnalysis(graphOp, effectiveInvariantArgMask, lattice);

    // STEP 5: Find constant operations and boundary values
    llvm::SmallSetVector<Operation *, 16> constantOps;
    SetVector<Value> boundaryValues;
    llvm::BitVector preservedArgs = effectiveInvariantArgMask;
    preservedArgs.flip(); // Invert mask: true for non-invariant args initially
    SmallVector<Operation *> sortedConstantOps;

    findConstantOpsAndBoundaryValues(
        graphOp, lattice, effectiveInvariantArgMask, constantOps,
        boundaryValues, preservedArgs, sortedConstantOps);

    // Skip if no boundary values - nothing to hoist
    if (boundaryValues.empty()) {
      LLVM_DEBUG(llvm::dbgs() << "No boundary values to hoist for "
                              << graphOp.getName() << "\n");
      return;
    }

    // STEP 6: Hoist constant operations to parent at each callsite
    auto newCallsites = hoistConstantOperations(
        graphOp, callsites, preservedArgs, effectiveInvariantArgMask,
        constantOps, boundaryValues, sortedConstantOps);

    // STEP 7: Update the subgraph definition
    updateSubgraphDefinition(graphOp, preservedArgs, boundaryValues,
                             sortedConstantOps);

    // STEP 8: Push chains to the back of the subgraph's signature and update
    // call sites.
    moveChainsToTheEnd(graphOp, newCallsites);
  }

  void moveChainsToTheEnd(GraphOp subgraphGraphOp, ArrayRef<CallOp> callsites) {
    SmallVector<uint64_t> argumentsToErase;
    auto block = subgraphGraphOp.getBodyBlock();
    // Clone arguments to the end of the argument list and do the use-def
    // replacement.
    for (auto &&arg : llvm::make_early_inc_range(block->getArguments())) {
      if (isa<ChainType>(arg.getType())) {
        auto newArg = block->insertArgument(block->getNumArguments(),
                                            arg.getType(), arg.getLoc());
        arg.replaceAllUsesWith(newArg);
        argumentsToErase.push_back(arg.getArgNumber());
      }
    }

    // Erase the old arguments.
    for (uint64_t idx : llvm::reverse(argumentsToErase))
      block->eraseArgument(idx);

    synchronizeMetadataUpdates(subgraphGraphOp);

    // Update the call sites.
    for (CallOp callOp : callsites) {
      OpBuilder builder(callOp);
      SmallVector<Value> newOperands;
      SmallVector<Value> chains;
      for (auto &&operand : callOp->getOperands()) {
        if (isa<ChainType>(operand.getType()))
          chains.push_back(operand);
        else
          newOperands.push_back(operand);
      }
      newOperands.insert(newOperands.end(), chains.begin(), chains.end());

      auto newCallOp = CallOp::create(
          builder, callOp->getLoc(), callOp->getResultTypes(), newOperands,
          callOp.getCalleeAttr(), callOp.getPrefixAttr(),
          /*outputParamDecls=*/{},
          /*arg_attrs=*/{},
          /*res_attrs=*/{});
      callOp->replaceAllUsesWith(newCallOp.getResults());
      callOp->erase();
    }
  }

  // Identify which arguments are callsite-invariant across all callsites.
  // An argument is "callsite-invariant" if all corresponding operands
  // refer to the exact same SSA Value in the parent across all callsites.
  llvm::BitVector computeEffectiveInvariantArgMask(GraphOp graphOp,
                                                   ArrayRef<CallOp> callsites) {
    unsigned numArgs = graphOp.getBody().getArguments().size();
    if (numArgs == 0)
      return llvm::BitVector(); // No arguments to check

    llvm::BitVector effectiveInvariantMask(numArgs,
                                           false); // Initialize to all false

    // This function is called after a check ensures `callsites` is not empty.
    assert(!callsites.empty() && "Callsites should not be empty here.");

    for (unsigned i = 0; i < numArgs; ++i) {
      // Check if all callsite operands are the exact same SSA Value
      Value firstOperandValue = callsites[0]->getOperand(i);
      bool allSameValue = true;
      for (size_t cIdx = 1; cIdx < callsites.size(); ++cIdx) {
        if (callsites[cIdx]->getOperand(i) != firstOperandValue) {
          allSameValue = false;
          break;
        }
      }

      if (allSameValue && callsites.size() > 1) {
        // If all callsites pass the exact same Value for this argument,
        // it's invariant from the subgraph's perspective.
        effectiveInvariantMask[i] = true;
        LLVM_DEBUG(llvm::dbgs()
                   << "Arg " << i
                   << " is callsite-invariant (all same parent value: "
                   << firstOperandValue << ").\n");
      } else {
        effectiveInvariantMask[i] = false;
        LLVM_DEBUG(llvm::dbgs()
                   << "Arg " << i << " is NOT callsite-invariant.\n");
      }
    }

    LLVM_DEBUG({
      llvm::dbgs() << "Callsite-invariant arguments mask for graph "
                   << graphOp.getName() << ": ";
      for (unsigned i : effectiveInvariantMask.set_bits())
        llvm::dbgs() << i << " ";
      llvm::dbgs() << "\n";
    });

    return effectiveInvariantMask;
  }

  // Perform dataflow analysis to find callsite-invariant-derived operations
  void performDataflowAnalysis(GraphOp graphOp,
                               const llvm::BitVector &effectiveInvariantArgMask,
                               LatticeState &lattice) {
    ParametricTopologyAnalysis topological(graphOp);

    auto allInputsConstant = [&](Operation *op) {
      return topological.forEachProvider(op, [&](Operation *input) {
        if (input == graphOp)
          return true;
        return lattice.query(input) == ConstLattice::Const;
      }) && llvm::all_of(op->getOperands(), [&](Value v) {
               return lattice.query(v) == ConstLattice::Const;
             });
    };

    Block *body = graphOp.getBodyBlock();
    SmallVector<Operation *> worklist;

    // Initialize lattice states for block arguments
    for (auto [idx, arg] : llvm::enumerate(body->getArguments())) {
      lattice.set(arg, effectiveInvariantArgMask[idx] ? ConstLattice::Const
                                                      : ConstLattice::NonConst);
    }

    // Initialize worklist with all operations in the block
    for (Operation &op : body->without_terminator())
      worklist.push_back(&op);

    // Process the worklist until fixed point
    while (!worklist.empty()) {
      Operation *op = worklist.back();
      worklist.pop_back();

      // Determine the result state for this operation
      // An operation is considered "const" (invariant-derived) if all its
      // operands are invariant-derived (come from callsite-invariant arguments
      // or other invariant-derived operations)
      bool allOperandsConst =
          (op->getNumOperands() > 0 || isa<KGEN::ParamDeclareOp>(op)) &&
          allInputsConstant(op);

      ConstLattice newState =
          allOperandsConst ? ConstLattice::Const : ConstLattice::NonConst;

      // If any result state changed, add all users to the worklist
      if (lattice.update(op, newState)) {
        topological.forEachUser(op, [&](Operation *user) {
          if (!user->hasTrait<OpTrait::IsTerminator>() &&
              !llvm::is_contained(worklist, user))
            worklist.push_back(user);

          return true;
        });
      }
    }
  }

  // Find callsite-invariant-derived operations and boundary values
  void findConstantOpsAndBoundaryValues(
      GraphOp graphOp, const LatticeState &lattice,
      const llvm::BitVector &effectiveInvariantArgMask,
      llvm::SmallSetVector<Operation *, 16> &constantOps,
      SetVector<Value> &boundaryValues, llvm::BitVector &preservedArgs,
      SmallVector<Operation *> &sortedConstantOps) {

    Block *body = graphOp.getBodyBlock();

    // Identify operations with all invariant-derived results
    for (Operation &op : body->without_terminator()) {
      // Check if all results are invariant-derived
      if (lattice.query(&op) == ConstLattice::Const)
        constantOps.insert(&op);
    }

    // Determine which callsite-invariant arguments are used by
    // non-invariant-derived operations An invariant argument needs to be
    // preserved if it's used by a non-hoistable operation.
    for (unsigned i : effectiveInvariantArgMask.set_bits()) {
      BlockArgument arg = body->getArgument(i);

      // Check if this invariant argument is used by any non-invariant-derived
      // operation
      for (Operation *user : arg.getUsers()) {
        if (!constantOps.contains(user)) {
          // This argument is used by a non-invariant-derived operation, so we
          // need to preserve it
          preservedArgs[i] = true;
          LLVM_DEBUG(
              llvm::dbgs()
              << "Arg " << i
              << " is callsite-invariant but used by non-invariant-derived op: "
              << *user << ", keeping it\n");
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
      llvm::dbgs() << "Invariant-derived ops in topological order:\n";
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
      llvm::dbgs() << "Found " << constantOps.size()
                   << " invariant-derived ops and " << boundaryValues.size()
                   << " boundary values\n";

      llvm::dbgs() << "Boundary values: ";
      for (Value boundaryVal : boundaryValues)
        llvm::dbgs() << boundaryVal << "\n";

      llvm::dbgs() << "\n";
    });
  }

  // Hoist callsite-invariant-derived operations to parent at each callsite
  SmallVector<CallOp> hoistConstantOperations(
      GraphOp subgraphGraphOp, ArrayRef<CallOp> callsites,
      const llvm::BitVector &preservedArgs,
      const llvm::BitVector &effectiveInvariantArgMask,
      const llvm::SmallSetVector<Operation *, 16> &constantOps,
      const SetVector<Value> &boundaryValues,
      const SmallVector<Operation *> &sortedConstantOps) {

    Block *body = subgraphGraphOp.getBodyBlock();
    KGEN::ParameterUseDefGraph subgraphParamUseDefGraph(
        subgraphGraphOp.getBody());
    KGEN::ParameterCollector::Analysis analysisCache;
    subgraphParamUseDefGraph.calculate(analysisCache);
    DenseMap<StringAttr, KGEN::ParamDeclaration> &subgraphDecls =
        subgraphParamUseDefGraph.decls;
    llvm::SetVector<StringAttr> subgraphDeclNames;
    subgraphDeclNames.insert_range(
        llvm::make_first_range(llvm::make_filter_range(
            subgraphDecls,
            [&](const std::pair<StringAttr, KGEN::ParamDeclaration> &decl) {
              return decl.second.declOp != subgraphGraphOp;
            })));

    SmallVector<CallOp> newCallSites;
    for (CallOp callOp : callsites) {
      OpBuilder builder(callOp);
      IRMapping valueMapping;
      GraphOp parentGraphOp = callOp->getParentOfType<MO::GraphOp>();
      Region &parentGraphRegion = parentGraphOp.getBody();
      KGEN::AttrTypeMangler::Cache manglerCache;
      KGEN::AttrTypeMangler mangler(manglerCache);
      KGEN::ParameterUseDefGraph paramGraph(parentGraphRegion);
      paramGraph.calculate(analysisCache);
      KGEN::NameUniquer uniquer(paramGraph, paramGraph);
      mangler.populate(builder, uniquer, subgraphDeclNames, paramGraph);

      assert(parentGraphOp && "expected parent graph op");

      // Map callsite-invariant arguments
      for (unsigned i : effectiveInvariantArgMask.set_bits())
        valueMapping.map(body->getArgument(i), callOp.getOperand(i));

      // Clone invariant-derived ops in topological order
      for (Operation *op : sortedConstantOps) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Cloning invariant-derived operation: " << *op << "\n");
        Operation *clonedOp = builder.clone(*op, valueMapping);

        if (auto itf = dyn_cast<KGEN::ParamOpInterface>(*clonedOp)) {
          LLVM_DEBUG(llvm::dbgs()
                     << "Processing ParamOpInterface for operation: "
                     << *clonedOp << "\n");
          SmallVector<KGEN::ParamDeclAttr> newDecls;
          itf.walkDeclarations([&](KGEN::ParamDeclAttr decl) {
            LLVM_DEBUG(llvm::dbgs()
                       << "Walking declaration: " << decl.getName() << "\n");
            newDecls.push_back(
                mangler.mangleDecl(decl, /*needsMangling=*/true));
          });
          cast<KGEN::ParamOpInterface>(clonedOp).renameDeclarations(newDecls);
          // Populate the new declarations into the call scope graph.
          for (KGEN::ParamDeclAttr decl : newDecls) {
            LLVM_DEBUG(llvm::dbgs() << "Adding declaration to param graph: "
                                    << decl.getName() << "\n");
            paramGraph.decls.try_emplace(
                decl.getName(), KGEN::ParamDeclaration{decl.getType(), clonedOp,
                                                       &parentGraphRegion});
          }
          parentGraphOp.setCounter(parentGraphOp.getCounter() +
                                   newDecls.size());
        }
        LLVM_DEBUG(llvm::dbgs()
                   << "Mangling elements in operation: " << *clonedOp << "\n");
        mangler.mangleElementsIn(clonedOp);

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
      auto newCallOp = CallOp::create(
          builder, callOp.getLoc(), callOp.getResultTypes(), newCallOperands,
          callOp.getCalleeAttr(), callOp.getPrefixAttr(),
          /*outputParamDecls=*/{},
          /*arg_attrs=*/{},
          /*res_attrs=*/{});

      // Replace uses of original call results
      callOp.replaceAllUsesWith(newCallOp.getResults());

      LLVM_DEBUG(llvm::dbgs() << "Replaced call at " << callOp.getLoc()
                              << " with new call\n");

      // Erase the original call
      newCallSites.push_back(newCallOp);
      callOp.erase();
    }

    return newCallSites;
  }

  // Update subgraph definition by removing constants and adding new arguments
  void
  updateSubgraphDefinition(GraphOp graphOp,
                           const llvm::BitVector &preservedArgs,
                           const SetVector<Value> &boundaryValues,
                           const SmallVector<Operation *> &sortedConstantOps) {

    Block *body = graphOp.getBodyBlock();
    unsigned numArgs = body->getArguments().size();

    // Get the list of arguments to delete (in reverse order for safe erasure).
    // llvm::BitVector doesn't have a reverse iterator, so we can't use
    // set_bits().
    SmallVector<unsigned> argsToDelete;
    for (int i = numArgs - 1; i >= 0; --i)
      if (!preservedArgs[i])
        argsToDelete.push_back(i);

    unsigned insertionIndex = numArgs;
    for (Value boundaryVal : boundaryValues) {
      BlockArgument newArg = body->insertArgument(
          insertionIndex++, boundaryVal.getType(), graphOp.getLoc());
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
