//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinOps.h"

#include "GenericML/GraphCompiler/MODialect/MOOps.h"
#include "GenericML/GraphCompiler/MODialect/MOTypes.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGPasses.h"
#include "GenericML/Support/Tensor.h"

#include "mlir/IR/IRMapping.h"

#include "LambdaUtils.h"

using namespace M;

namespace M::MOGG {

#define GEN_PASS_DEF_FUSEELEMENTWISE
#include "GenericML/GraphCompiler/MOGGDialect/MOGGPasses.h.inc"

namespace {

// A graph chained elementwise kernels. Each node is an elementwise operations
// (with view operations allowed). The leaves of the graph are the operations
// which do not feed into other elementwise ops. The roots are operations which
// do not consume elementwise or view changing inputs.
struct ElementwiseGraph {
  void buildGraph(MOGG::KernelOp kernel);

  llvm::SmallVector<MOGG::KernelOp> roots;
  llvm::SmallVector<MOGG::KernelOp> leaves;
};

void ElementwiseGraph::buildGraph(MOGG::KernelOp start) {
  llvm::SmallDenseSet<MOGG::KernelOp> seenNodes;

  // All kernels in the graph must be in the same region.
  auto region = start->getParentRegion();

  SmallVector<MOGG::KernelOp> searchList;
  searchList.push_back(start);

  // We will process outputs and add to the leaves and process inputs and add to
  // the roots if the current op is the end of the chain.
  auto process = [&searchList, &seenNodes,
                  region](KernelOp current, Operation *userOrProducer,
                          SmallVector<KernelOp> &endpoint) {
    // Must be in same region to be connected.
    if (region != userOrProducer->getParentRegion())
      endpoint.push_back(current);

    if (auto useOrP = dyn_cast<KernelOp>(userOrProducer)) {
      // If the user or producer is anythig but an elementwise kernel then we
      // are a leaf/root. Elementwise = is considered implementable just by
      // application of an elementwise lambda across all elements.
      if (!useOrP.getFusionBodyLambda()) {
        endpoint.push_back(current);
      } else {
        if (!seenNodes.contains(useOrP))
          searchList.push_back(useOrP);
      }
    } else {
      // All other ops are exits.
      endpoint.push_back(current);
    }
  };

  while (!searchList.empty()) {
    MOGG::KernelOp op = searchList.back();
    searchList.pop_back();
    seenNodes.insert(op);

    // Traverse down the graph. If any operation exits into a non-elementwise
    // kernel then it is a leaf node.
    for (Operation *user : op->getUsers())
      process(op, user, leaves);

    // Traverse upwards.
    for (Value operand : op.getOperands()) {
      Operation *edge = operand.getDefiningOp();

      // Block arguments will appear as null.
      if (edge)
        process(op, edge, roots);
      else
        roots.push_back(op);
    }
  }
}

// Implements fusion by applying the algorithm:
// 1. Build a graph of elementwise kernels by looking through the outputs and
//    inputs of each elementwise kernel. This gives us a view of which
//    elementwise kernel is dominated by all elementwise kernels in a cluster of
//    kernels.
// 2. For each leaf node in that graph, if the input is an elementwise kernel
//    fuse it into the body of the leaf kernel.

// Bear in mind what a kernel looks like:

// clang-format off
/*
%2 = mogg.kernel (%0, %1) : (%arg0: !mo.tensor<[4], f32>, %arg1: !mo.tensor<[4],f32>) -> !mo.tensor<[4], f32>{ 
  mogg.lambda @input_0_fn_0 ... 
  mogg.lambda @input_1_fn_1 ...
  mogg.lambda @output_0_fn ...
  mogg.lambda @elemwise_compute_lambda ... 
} {fusionBodyLambda = "elemwise_compute_lambda",
  inputFusionInterface = ["input_0_fn_0", "input_1_fn"],
  outputFusionInterface = ["output_0_fn"]}
*/
// clang-format on
// The `fusionBodyLambda` is used to identify the body we fuse and the
// `inputFusionInterface` are used to identify the lambdas assosiated for a
// given input. It tells us that accesses to operand 0 (%0) are implemented by
// an invoke of the lambda `input_0_fn_0` so to fuse the producer of `%0` into
// this kernel we need to replace those invokes.

// The fusion itself is applied by the following steps. The kernel being fused
// into the leaf is referred to by 'other'.
// 1. Add all operands of the other kernel to the leaf kernel, and add new args
//    to the body.
// 2. For each input lambda on the other kernel clone it into the leaf kernel
//    and remap to refer to the added arguments.
// 3. Take the elementwise lambda from the other kernel and replace the lambda
//    invoke of the argument we are replacing (remember we are fusing the other
//    body into a use of the other kernel). Remap each input invocation to point
//    to the cloned input lambdas.
// 4. Remove the old argument and associated lambda.
// 5. Update the attributes to reflect the new inputs.

struct FuseElementwisePass
    : public impl::FuseElementwiseBase<FuseElementwisePass> {
  using FuseElementwiseBase::FuseElementwiseBase;

  void runOnOperation() override {
    MO::GraphOp graph = getOperation();

    // Build a graph of all reachable elementwise kernels from each one. After
    // the graph is created it is fused but the old kernel is not removed, it
    // just will just no longer be used in that kernel.
    for (MOGG::KernelOp kernel : graph.getOps<MOGG::KernelOp>()) {
      // Skip non elementwise kernels.
      if (!kernel.getFusionBodyLambda() || kernel.getIsViewOp().value_or(false))
        continue;

      // The graph of reachable elementwise ops from this kernel
      ElementwiseGraph reachableOps;
      reachableOps.buildGraph(kernel);

      // Traverse up from each operation and fuse all connected operations
      // downwards into it.
      for (MOGG::KernelOp leaf : reachableOps.leaves) {
        // Map of the operands we've already fused in. This is for the case of:
        // %0 = op(...)
        // %1 = op(%0)
        // %2 = op(%0, %1)
        // We don't want to have multiple copies of %0 fused in.

        DenseMap<Value, Value> alreadyFused;

        // On each leaf in the graph we will continously fuse until nothing is
        // left to fuse down.
        while (true) {
          int32_t idx = getInputToFuse(leaf);
          if (idx == -1)
            break;

          fuseInputAtIndex(leaf, idx, alreadyFused);
        }
      }
    }
  }

  /// Return the index of the input to fuse or -1 if nothing can be fused.
  int32_t getInputToFuse(MOGG::KernelOp leaf) {
    for (int32_t index = 0, e = leaf.getNumOperands(); index < e; ++index) {
      // Block arguments have no defining op and obviously cant be fused.
      Operation *edge = leaf.getOperand(index).getDefiningOp();
      if (!edge)
        continue;

      if (edge->getParentRegion() != leaf->getParentRegion())
        continue;

      // If the input operation is a kernel and is implemented by a lambda then
      // we can fuse it.
      if (auto otherKernel = dyn_cast<KernelOp>(edge)) {
        // If the other kernel has been marked a view operation we don't fuse it
        // here.
        if (otherKernel.getIsViewOp().value_or(false))
          return -1;

        if (otherKernel.getFusionBodyLambda())
          return index;
      }
    }
    return -1;
  }

  /// Remove the input interface attr for the given index.
  llvm::SmallVector<StringRef>
  removeIndexFromInputInterface(MOGG::KernelOp leaf, int32_t index) {
    SmallVector<StringRef> newHooks;
    ArrayAttr leafHooks = leaf.getInputFusionInterfaceAttr();

    int32_t hookIndex = 0;
    // Add each of the current input hooks ones to the list.
    for (StringAttr name : leafHooks.getAsRange<StringAttr>()) {
      if (hookIndex != index)
        newHooks.push_back(name.strref());
      hookIndex++;
    }

    return newHooks;
  }

  /// Fuse the input of the given index of the kernel into this kernel.
  void fuseInputAtIndex(MOGG::KernelOp leaf, int32_t idx,
                        DenseMap<Value, Value> &alreadyFused) {
    IRMapping remapper;
    Value operandBeingReplaced = leaf.getOperand(idx);

    // We already check everything in the getInputToFuse method so we know this
    // index is a KernelOp
    KernelOp producerKernel =
        cast<KernelOp>(operandBeingReplaced.getDefiningOp());

    // The input lambda we are going to replace with the fused body.
    ArrayAttr leafHooks = leaf.getInputFusionInterfaceAttr();
    StringAttr inputLambdaToReplace = cast<StringAttr>(leafHooks[idx]);

    // Each input on the kernel has an assosiated input lambda which loads that
    // operand. When we remove the operand we must remove the lamdba and update
    // the list of exposed lambdas to reflect that.
    SmallVector<StringRef> newHooks = removeIndexFromInputInterface(leaf, idx);

    // Symbol tables for the both the incoming kernel and the kernel we are
    // fusing into.
    mlir::SymbolTable leafSymTable{leaf};
    mlir::SymbolTable producerSymTable{producerKernel};

    Block &oldBlock = producerKernel.getBody().front();
    Block &leafBlock = leaf.getBody().front();

    // Add the new lambdas to the top of the leaf kernel.
    mlir::OpBuilder inserter{&leafBlock, leafBlock.begin()};

    // Get the main elementwise block in the kernel we are fusing into.
    StringRef leafMainBody = *leaf.getFusionBodyLambda();
    auto leafElemwiseLambda = cast<LambdaOp>(leafSymTable.lookup(leafMainBody));

    // If we've already fused then early exit by just reusing the value we have
    // replaced.
    auto itr = alreadyFused.find(operandBeingReplaced);
    if (itr != alreadyFused.end()) {
      replaceLoadWithExisitingUse(leaf, inputLambdaToReplace,
                                  leafElemwiseLambda, inserter, itr->second);

      // Expose the new updated input hooks.
      leaf.setInputFusionInterfaceAttr(inserter.getStrArrayAttr(newHooks));

      // Remove the argument.
      leaf->eraseOperand(idx);
      leafBlock.eraseArgument(idx);
      return;
    }

    llvm::StringMap<StringRef> oldNameToNew;

    // Clone each of the input lambdas for the inputs we don't already have.
    ArrayAttr otherInputLambdas = producerKernel.getInputFusionInterfaceAttr();
    ArrayAttr leafInputLambdas = leaf.getInputFusionInterfaceAttr();

    llvm::SmallDenseMap<Value, size_t> leafOperands;
    for (size_t i = 0, e = leaf->getNumOperands(); i < e; ++i) {
      Value operand = leaf->getOperand(i);
      leafOperands[operand] = i;
    }

    // Add the operands to the graph level op and remap the values from the
    // incoming kernel.
    for (size_t i = 0, e = producerKernel->getNumOperands(); i < e; ++i) {
      // The values on the kernel being fused in.
      Value otherOperand = producerKernel->getOperand(i);
      Value otherBlockArg = oldBlock.getArgument(i);

      // Get the lambda which loads this operand in the other kernel.
      StringRef name = cast<StringAttr>(otherInputLambdas[i]).getValue();
      auto otherLambdaLoad = cast<LambdaOp>(producerSymTable.lookup(name));

      auto itr = leafOperands.find(otherOperand);

      // If we already have this operand then reuse the current copy. Otherwise
      // add a new argument and operand.
      if (itr != leafOperands.end()) {
        size_t leafInIndex = itr->second;
        remapper.map(otherBlockArg, leafBlock.getArgument(leafInIndex));

        // All loads from the old lambda should point to the lambda which exists
        // in this kernel.
        StringRef leafName =
            cast<StringAttr>(leafInputLambdas[leafInIndex]).getValue();
        oldNameToNew[otherLambdaLoad.getSymName()] = leafName;
      } else {
        // Add the new operand to the kernel + body.
        leaf->insertOperands(leaf->getNumOperands(), ValueRange{otherOperand});
        Value leafBlockArg = leafBlock.addArgument(otherBlockArg.getType(),
                                                   otherBlockArg.getLoc());
        remapper.map(otherBlockArg, leafBlockArg);

        // As we have added a new input we need to clone the lambda which loads
        // that input.
        auto newLoadLambda = cast<LambdaOp>(otherLambdaLoad->clone(remapper));

        // Add the lambda to the kernel.
        leafSymTable.insert(newLoadLambda, inserter.getInsertionPoint());

        // Add to the input interfaces so they can then be fused into later.
        newHooks.push_back(newLoadLambda.getSymName());

        // Remap so any uses of the old lambda can be remapped onto the new
        // lambda when we fuse the main elementwise body.
        oldNameToNew[otherLambdaLoad.getSymName()] = newLoadLambda.getSymName();
      }
    }

    // Expose the new updated input hooks.
    leaf.setInputFusionInterfaceAttr(inserter.getStrArrayAttr(newHooks));

    StringRef mainBody = *producerKernel.getFusionBodyLambda();
    auto otherElemwiseLambda =
        cast<LambdaOp>(producerSymTable.lookup(mainBody));

    // Clone the block.
    auto newOp = cast<LambdaOp>(otherElemwiseLambda->clone(remapper));

    // Track the calls to the output lambda[s] so they can be removed.
    llvm::SmallDenseSet<StringRef> outputFns;
    ArrayAttr outputHooks = leaf.getOutputFusionInterfaceAttr();
    for (StringRef name : outputHooks.getAsValueRange<StringAttr>())
      outputFns.insert(name);

    auto returnOp =
        cast<MOGG::OutputOp>(newOp.getBody().front().getTerminator());

    // When we inline the new body we will have lambda invokes representing
    // loads from the input lambdas. These symbols will be references to lambdas
    // in the other kernel so we need to remap onto the ones in this kernel.
    for (auto invokeOp :
         llvm::make_early_inc_range(newOp.getOps<MOGG::LambdaInvokeOp>())) {
      StringRef oldName = invokeOp.getLambda().getLeafReference().strref();

      // Remove all references to the old stores.
      if (outputFns.contains(oldName)) {
        // Add the values we previously stored as the output.
        for (Value arg : invokeOp.getOperands()) {
          // Add everything but indices.
          if (!isa<MOGG::IndicesType>(arg.getType()))
            returnOp->insertOperands(returnOp.getNumOperands(), arg);
        }

        invokeOp.erase();
      } else {
        auto itr = oldNameToNew.find(oldName);
        if (itr != oldNameToNew.end()) {
          invokeOp.setLambdaAttr(
              FlatSymbolRefAttr::get(invokeOp.getContext(), itr->second));
        }
      }
    }

    // Add the new lambda to the table.
    leafSymTable.insert(newOp, inserter.getInsertionPoint());
    inserter.setInsertionPointAfter(newOp);

    // Replace all uses of the old input lambda inside the main body with the
    // inlined body of the replacement lambda. The only reason we don't replace
    // it with a call is because it returns void so we would need to create a
    // new one for a legal call.
    auto uses =
        *leafSymTable.getSymbolUses(inputLambdaToReplace, leafElemwiseLambda);
    for (auto use : llvm::make_early_inc_range(uses)) {
      auto toReplace = cast<MOGG::LambdaInvokeOp>(use.getUser());

      // Inline the first invoke and then reuse those values.
      auto itr = alreadyFused.find(operandBeingReplaced);
      if (itr == alreadyFused.end()) {
        auto newValues = inlineInvoke(toReplace, newOp);
        assert(
            newValues.size() == 1 &&
            "Currently assuming one return from elementwise ops when fusing");
        alreadyFused[operandBeingReplaced] = newValues[0];
      } else {
        toReplace->replaceAllUsesWith(ValueRange{itr->second});
      }

      // Remove the lambda if we didn't already remove it.
      auto op = SymbolTable::lookupSymbolIn(leaf, toReplace.getLambda());
      if (op)
        op->erase();

      // Remove the call to the old input lambda.
      toReplace.erase();
    }

    // All uses have been inlined so get rid of the new lambda.
    newOp.erase();

    // Remove the argument we have effectively inlined.
    leaf->eraseOperand(idx);
    leafBlock.eraseArgument(idx);
  }

  // Replace all loads from `inputLambdaToReplace` in `leafElemwiseLambda` with
  // the existing value `newVal`.
  void replaceLoadWithExisitingUse(MOGG::KernelOp leaf,
                                   StringAttr inputLambdaToReplace,
                                   MOGG::LambdaOp leafElemwiseLambda,
                                   mlir::OpBuilder &builder, Value newVal) {
    mlir::SymbolTable leafSymTable{leaf};

    // The input lambda we should remove.
    auto lambdaToRemove =
        cast<LambdaOp>(leafSymTable.lookup(inputLambdaToReplace));

    // Replace all uses inside the body of this lambda with the value we
    // already fused in.
    auto uses = *leafSymTable.getSymbolUses(lambdaToRemove, leafElemwiseLambda);
    for (auto use : llvm::make_early_inc_range(uses)) {
      auto toReplace = cast<MOGG::LambdaInvokeOp>(use.getUser());
      toReplace.replaceAllUsesWith(ValueRange{newVal});
      toReplace.erase();
    }

    lambdaToRemove.erase();
  }
};

} // namespace

} // namespace M::MOGG
