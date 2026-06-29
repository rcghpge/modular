//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements the device-graph operations in the MGP dialect: the
// `mgp.device_graph.create`/`mgp.device_graph.execute` surface, its
// `mgp.device_graph.yield` build-body terminator, and the transient
// `mgp.device_graph.staging`/`mgp.device_graph.dependencies` scaffolds (with
// their yield terminators) used during device-graph synthesis. The
// declarations for these ops live in MGPDeviceGraphOps.td (included by
// MGPOps.td).
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/GraphCompiler/MGPDialect/Dialect/MGPOps.h"
#include "GraphCompiler/GraphCompiler/MGPDialect/Dialect/MGPTypes.h"
#include "GraphCompiler/GraphCompiler/MOGGDialect/MOGGTypes.h"
#include "MLRT/GraphRT/MEFDialect/MEFTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SetVector.h"

using namespace M;
using namespace MGP;

//===----------------------------------------------------------------------===//
// DeviceGraphCreateOp
//===----------------------------------------------------------------------===//

LogicalResult DeviceGraphCreateOp::verify() {
  Block &body = *getBodyBuilderBlock();

  // The body must terminate with a `mgp.device_graph.yield`. Because the op
  // uses `SingleBlock` rather than `SingleBlockImplicitTerminator`, the
  // terminator is not auto-inserted -- the trait only guarantees the region
  // has exactly one block.
  if (body.empty() || !llvm::isa<DeviceGraphYieldOp>(body.back()))
    return emitOpError(
        "build body must terminate with `mgp.device_graph.yield`");

  // Block argument types are [builder, inputs..., outputs...]. The op creates
  // the builder from its `deviceContext` operand and passes it in as the first
  // block arg; it instantiates the recorded builder into `outGraph` itself, so
  // the body neither creates the builder nor yields the graph.
  //
  // Like `mogg.generic.compiled.region`, the input/output arguments arrive as
  // `!mogg.async<T>` wrappers around the corresponding operand type `T`: the
  // region boundary wraps the buffers/tensors into async values on entry and
  // the body unpacks them via `mogg.async.unpack`. An argument consumed
  // directly as the plain operand type `T` is also accepted (mirroring the
  // region's ABI-lowering flexibility).
  size_t numValueArgs = getInputs().size() + getOutputs().size();
  if (body.getNumArguments() != 1 + numValueArgs)
    return emitOpError("build body has ")
           << body.getNumArguments() << " block argument(s), expected "
           << (1 + numValueArgs);

  if (!llvm::isa<DeviceGraphBuilderType>(body.getArgument(0).getType()))
    return emitOpError("build body argument #0 must be a "
                       "`!mgp.device_graph.builder`, but is ")
           << body.getArgument(0).getType();

  // inputs then outputs, matched against the body args after the builder.
  SmallVector<Type> valueOperandTypes(getInputs().getTypes());
  llvm::append_range(valueOperandTypes, getOutputs().getTypes());
  for (auto [idx, operandType] : llvm::enumerate(valueOperandTypes)) {
    Type argType = body.getArgument(idx + 1).getType();
    if (argType == operandType)
      continue;
    auto asyncType = llvm::dyn_cast<MOGG::AsyncType>(argType);
    if (!asyncType || asyncType.getElementType() != operandType)
      return emitOpError("build body argument #")
             << (idx + 1) << " has type " << argType << ", expected "
             << operandType << " or a `!mogg.async` of it";
  }

  // The result device_graph's signature must agree with the op-level operand
  // grouping: signature.inputs == inputs, signature.results == outputs.
  FunctionType sig = getOutGraph().getType().getSignature();
  if (!llvm::equal(sig.getInputs(), getInputs().getTypes()))
    return emitOpError("result `!mgp.device_graph` signature inputs do not "
                       "match the op's `inputs` operand types");
  if (!llvm::equal(sig.getResults(), getOutputs().getTypes()))
    return emitOpError("result `!mgp.device_graph` signature results do not "
                       "match the op's `outputs` operand types");

  return success();
}

LogicalResult DeviceGraphCreateOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, Adaptor adaptor,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  // The device graph's signature mirrors the op's operand grouping:
  // function inputs == `inputs`, function results == `outputs`.
  auto signature = FunctionType::get(context, adaptor.getInputs().getTypes(),
                                     adaptor.getOutputs().getTypes());

  // Results are (outChain, outGraph) in declaration order.
  inferredReturnTypes.push_back(MEF::ChainType::get(context));
  inferredReturnTypes.push_back(DeviceGraphType::get(context, signature));
  return success();
}

//===----------------------------------------------------------------------===//
// DeviceGraphExecuteOp
//===----------------------------------------------------------------------===//

LogicalResult DeviceGraphExecuteOp::verify() {
  FunctionType sig = getGraph().getType().getSignature();

  if (!llvm::equal(getInputs().getTypes(), sig.getInputs()))
    return emitOpError("`inputs` operand types do not match the graph's "
                       "signature inputs");

  if (!llvm::equal(getOutputs().getTypes(), sig.getResults()))
    return emitOpError("`outputs` operand types do not match the graph's "
                       "signature results");

  return success();
}

//===----------------------------------------------------------------------===//
// DeviceGraphDependenciesOp
//===----------------------------------------------------------------------===//

DeviceGraphDependenciesYieldOp DeviceGraphDependenciesOp::getYieldOp() {
  return cast<DeviceGraphDependenciesYieldOp>(
      getBody().front().getTerminator());
}

BlockArgument DeviceGraphDependenciesOp::getBuilderBlockArgument() {
  return getBody().front().getArgument(0);
}

LogicalResult DeviceGraphDependenciesOp::verify() {
  // The body has exactly one block argument: the `!mgp.device_graph.builder`
  // (arg #0) the unit's device-context uses are re-sourced from. It maps onto
  // the enclosing staging/create builder.
  Block &body = getBody().front();
  if (body.getNumArguments() != 1)
    return emitOpError("body has ")
           << body.getNumArguments()
           << " block argument(s), expected exactly 1 (the "
              "`!mgp.device_graph.builder`)";
  if (!llvm::isa<DeviceGraphBuilderType>(body.getArgument(0).getType()))
    return emitOpError("body argument #0 must be a "
                       "`!mgp.device_graph.builder`, but is ")
           << body.getArgument(0).getType();

  // SingleBlockImplicitTerminator guarantees a single block terminated by a
  // DeviceGraphDependenciesYieldOp; the yield's operands carry the former
  // region's results out, so their types must match this op's `bodyResults`.
  DeviceGraphDependenciesYieldOp yieldOp = getYieldOp();
  if (!llvm::equal(yieldOp.getOperands().getTypes(),
                   getBodyResults().getTypes()))
    return emitOpError(
        "`mgp.device_graph.dependencies.yield` operand types must "
        "match the op's body result types");
  return success();
}

LogicalResult DeviceGraphDependenciesOp::fold(FoldAdaptor,
                                              SmallVectorImpl<OpFoldResult> &) {
  llvm::SmallSetVector<Value, 8> deduped;
  deduped.insert_range(getDependencies());

  // Nothing to do if every dependency edge is already distinct.
  if (deduped.size() == getDependencies().size())
    return failure();

  getDependenciesMutable().assign(deduped.getArrayRef());
  return success();
}

void DeviceGraphDependenciesOp::build(
    OpBuilder &builder, OperationState &result, Value graphBuilder,
    ValueRange dependencies,
    llvm::function_ref<void(OpBuilder &, Location, Value)> bodyBuilder) {
  OpBuilder::InsertionGuard guard(builder);

  // Operand order mirrors `arguments`: the builder, then the dependency edges.
  result.addOperands(graphBuilder);
  result.addOperands(dependencies);

  // The synthesized node is always the first result.
  result.addTypes(DeviceGraphNodeType::get(builder.getContext()));

  // Create the body with its sole block argument -- the
  // `!mgp.device_graph.builder` (bound to the `graphBuilder` operand) the
  // body's device-context uses are re-sourced from -- and let the callback
  // populate it (it is expected to emit the dependencies.yield carrying the
  // former region's results). The body otherwise references enclosing SSA
  // directly. createBlock also positions the insertion point.
  Region *bodyRegion = result.addRegion();
  Block *body = builder.createBlock(bodyRegion);
  Value builderArg = body->addArgument(
      DeviceGraphBuilderType::get(builder.getContext()), result.location);
  if (bodyBuilder)
    bodyBuilder(builder, result.location, builderArg);

  // Supply the operandless terminator (no body results) if the callback did not
  // emit one.
  if (!body->mightHaveTerminator())
    ensureTerminator(*bodyRegion, builder, result.location);

  // The remaining results are exactly the dependencies.yield operands.
  auto yieldOp = cast<DeviceGraphDependenciesYieldOp>(body->getTerminator());
  result.addTypes(yieldOp.getOperands().getTypes());
}

namespace {

/// Rebuilds an `mgp.device_graph.dependencies` op with a new set of body
/// results (and matching `dependencies.yield` operands), preserving the
/// synthesized `node` result and the dependency edges. `newBodyResultTypes`
/// describes the surviving body results; `newYieldOperands` are the values the
/// moved yield should carry. `bodyResultReplacement(i)` returns, for each
/// original body result index `i`, the position of its replacement among the
/// new body results.
DeviceGraphDependenciesOp
rebuildWithBodyResults(PatternRewriter &rewriter, DeviceGraphDependenciesOp op,
                       TypeRange newBodyResultTypes,
                       ValueRange newYieldOperands,
                       ArrayRef<unsigned> bodyResultRemap) {
  // The `node` result is synthesized (not yielded); it stays first, followed by
  // the surviving body results.
  SmallVector<Type> newResultTypes;
  newResultTypes.push_back(op.getNode().getType());
  llvm::append_range(newResultTypes, newBodyResultTypes);

  auto newOp = DeviceGraphDependenciesOp::create(
      rewriter, op.getLoc(), TypeRange(newResultTypes), op.getGraphBuilder(),
      op.getDependencies());
  newOp.getBody().takeBody(op.getBody());
  DeviceGraphDependenciesYieldOp newYield = newOp.getYieldOp();
  rewriter.modifyOpInPlace(newYield, [&]() {
    newYield.getOperandsMutable().assign(newYieldOperands);
  });

  // Map results back onto the original positions: the node, then each original
  // body result onto its surviving counterpart (offset by one for the node).
  SmallVector<Value> replacements(op.getNumResults());
  replacements[0] = newOp.getNode();
  for (auto [idx, result] : llvm::enumerate(op.getBodyResults())) {
    if (result.use_empty())
      continue;
    replacements[idx + 1] = newOp.getResult(bodyResultRemap[idx] + 1);
  }
  rewriter.replaceOp(op, replacements);
  return newOp;
}

/// Drops body results of an `mgp.device_graph.dependencies` op that have no
/// uses, along with the matching operands of the body's `dependencies.yield`.
/// The synthesized `node` result is always kept (it carries the ordering edge,
/// not dataflow).
struct RemoveUnusedDependenciesResults
    : public OpRewritePattern<DeviceGraphDependenciesOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(DeviceGraphDependenciesOp op,
                                PatternRewriter &rewriter) const override {
    // Collect the indices/types of the body results that are actually used.
    SmallVector<unsigned> usedResultIndices;
    SmallVector<Type> usedResultTypes;
    for (auto [idx, result] : llvm::enumerate(op.getBodyResults())) {
      if (!result.use_empty()) {
        usedResultIndices.push_back(idx);
        usedResultTypes.push_back(result.getType());
      }
    }

    // Nothing to do if every body result is used.
    if (usedResultIndices.size() == op.getBodyResults().size())
      return failure();

    // Snapshot the surviving yield operands before moving the body.
    DeviceGraphDependenciesYieldOp oldYield = op.getYieldOp();
    auto newYieldOperands =
        llvm::map_to_vector(usedResultIndices, [&](unsigned idx) -> Value {
          return oldYield.getOperand(idx);
        });

    // Each used body result maps onto its slot in the compacted list.
    SmallVector<unsigned> remap(op.getBodyResults().size());
    for (auto [newIdx, oldIdx] : llvm::enumerate(usedResultIndices))
      remap[oldIdx] = newIdx;

    rebuildWithBodyResults(rewriter, op, TypeRange(usedResultTypes),
                           newYieldOperands, remap);
    return success();
  }
};

/// De-duplicates body results of an `mgp.device_graph.dependencies` op whose
/// `dependencies.yield` carries the same value more than once: the yield keeps
/// one copy and every duplicate result is rewritten to the surviving one.
struct DeduplicateDependenciesResults
    : public OpRewritePattern<DeviceGraphDependenciesOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(DeviceGraphDependenciesOp op,
                                PatternRewriter &rewriter) const override {
    DeviceGraphDependenciesYieldOp oldYield = op.getYieldOp();

    unsigned numElements = oldYield.getNumOperands();

    llvm::SmallDenseMap<Value, unsigned> firstOccurrence(numElements);
    SmallVector<unsigned> resultRemap(op.getBodyResults().size());

    for (auto [idx, operand] : llvm::enumerate(oldYield.getOperands())) {
      auto [it, _] =
          firstOccurrence.try_emplace(operand, firstOccurrence.size());
      resultRemap[idx] = it->second;
    }

    // Nothing to do if there are no duplicated yield operands.
    if (firstOccurrence.size() == op.getBodyResults().size())
      return failure();

    auto dedupedOperands = llvm::to_vector(firstOccurrence.keys());
    rebuildWithBodyResults(rewriter, op, TypeRange(dedupedOperands),
                           dedupedOperands, resultRemap);
    return success();
  }
};

} // namespace

void DeviceGraphDependenciesOp::getCanonicalizationPatterns(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.add<RemoveUnusedDependenciesResults, DeduplicateDependenciesResults>(
      ctx);
}

//===----------------------------------------------------------------------===//
// DeviceGraphDependenciesYieldOp
//===----------------------------------------------------------------------===//

void DeviceGraphDependenciesYieldOp::build(OpBuilder &, OperationState &) {
  // No-operand stub for the parent's SingleBlockImplicitTerminator; operands
  // are populated after insertion.
}

//===----------------------------------------------------------------------===//
// DeviceGraphStagingOp
//===----------------------------------------------------------------------===//

DeviceGraphStagingYieldOp DeviceGraphStagingOp::getYieldOp() {
  return cast<DeviceGraphStagingYieldOp>(getBody().front().getTerminator());
}

namespace {
/// `got` is an acceptable boundary type for plain value type `want` when it is
/// either `want` itself or `!mogg.async<want>` -- the staging op wraps its
/// boundary values into async, like `mogg.generic.compiled.region` (which also
/// tolerates the unwrapped form for ABI-lowered slots).
bool isAsyncOrPlain(Type got, Type want) {
  if (got == want)
    return true;
  auto asyncType = llvm::dyn_cast<MOGG::AsyncType>(got);
  return asyncType && asyncType.getElementType() == want;
}

/// Unwrap `!mogg.async<T>` to `T`; leave other types unchanged.
Type unwrapAsync(Type type) {
  if (auto asyncType = llvm::dyn_cast<MOGG::AsyncType>(type))
    return asyncType.getElementType();
  return type;
}
} // namespace

LogicalResult DeviceGraphStagingOp::verify() {
  Block &body = getBody().front();

  // Block args are [builder, async<input_0>, ..., async<input_n-1>]: the
  // builder has no operand; the rest are the `inputs` operands wrapped into
  // `!mogg.async<T>` (the plain `T` form is also accepted, as in
  // `mogg.generic.compiled.region`).
  if (body.getNumArguments() != 1 + getInputs().size())
    return emitOpError("staging body has ")
           << body.getNumArguments() << " block argument(s), expected "
           << (1 + getInputs().size())
           << " (the device-graph builder plus one "
              "per input)";
  if (!isa<DeviceGraphBuilderType>(body.getArgument(0).getType()))
    return emitOpError("staging body argument #0 must be a "
                       "`!mgp.device_graph.builder`, but is ")
           << body.getArgument(0).getType();
  for (auto [idx, inputType] : llvm::enumerate(getInputs().getTypes())) {
    Type argType = body.getArgument(idx + 1).getType();
    if (!isAsyncOrPlain(argType, inputType))
      return emitOpError("staging body argument #")
             << (idx + 1) << " has type " << argType << ", expected "
             << inputType << " or a `!mogg.async` of it";
  }

  // The yield carries the staged results out as `!mogg.async<T>` (or plain `T`)
  // while the op's results are the unwrapped `T`.
  DeviceGraphStagingYieldOp yieldOp = getYieldOp();
  if (yieldOp.getNumOperands() != getNumResults())
    return emitOpError("`mgp.device_graph.staging.yield` has ")
           << yieldOp.getNumOperands() << " operand(s), expected "
           << getNumResults();
  for (auto [idx, resultType] :
       llvm::enumerate(getStagedResults().getTypes())) {
    Type operandType = yieldOp.getOperand(idx).getType();
    if (!isAsyncOrPlain(operandType, resultType))
      return emitOpError("`mgp.device_graph.staging.yield` operand #")
             << idx << " has type " << operandType << ", expected "
             << resultType << " or a `!mogg.async` of it";
  }
  return success();
}

void DeviceGraphStagingOp::build(
    OpBuilder &builder, OperationState &result, ValueRange inputs,
    llvm::function_ref<void(OpBuilder &, Location, Value, ValueRange)>
        bodyBuilder) {
  OpBuilder::InsertionGuard guard(builder);

  result.addOperands(inputs);

  // Create the body with the device-graph-builder block argument followed by
  // one
  // `!mogg.async<T>` argument per input, and let the callback populate it (it
  // is expected to emit the staging.yield carrying the async-wrapped staged
  // values). createBlock also positions the insertion point.
  Region *bodyRegion = result.addRegion();
  Block *body = builder.createBlock(bodyRegion);
  Value graphBuilder = body->addArgument(
      DeviceGraphBuilderType::get(builder.getContext()), result.location);
  SmallVector<Value> asyncInputArgs;
  asyncInputArgs.reserve(inputs.size());
  for (Value input : inputs)
    asyncInputArgs.push_back(body->addArgument(
        MOGG::AsyncType::get(input.getType()), result.location));
  if (bodyBuilder)
    bodyBuilder(builder, result.location, graphBuilder, asyncInputArgs);

  // Supply the operandless terminator (zero staged results) if the callback did
  // not emit one.
  if (!body->mightHaveTerminator())
    ensureTerminator(*bodyRegion, builder, result.location);

  // The op's results are the staging.yield operands, unwrapped from async.
  auto yieldOp = cast<DeviceGraphStagingYieldOp>(body->getTerminator());
  for (Type operandType : yieldOp.getOperands().getTypes())
    result.addTypes(unwrapAsync(operandType));
}

namespace {

/// Drops results of an `mgp.device_graph.staging` op that have no uses, along
/// with the matching operands of the body's `staging.yield`.
struct RemoveUnusedStagingResults
    : public OpRewritePattern<DeviceGraphStagingOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(DeviceGraphStagingOp op,
                                PatternRewriter &rewriter) const override {
    // Collect the indices/types of the results that are actually used.
    SmallVector<unsigned> usedResultIndices;
    SmallVector<Type> usedResultTypes;
    for (auto [idx, result] : llvm::enumerate(op.getResults())) {
      if (!result.use_empty()) {
        usedResultIndices.push_back(idx);
        usedResultTypes.push_back(result.getType());
      }
    }

    // Nothing to do if every result is used.
    if (usedResultIndices.size() == op.getNumResults())
      return failure();

    // Snapshot the surviving yield operands before moving the body.
    DeviceGraphStagingYieldOp oldYield = op.getYieldOp();
    auto newYieldOperands =
        llvm::map_to_vector(usedResultIndices, [&](unsigned idx) -> Value {
          return oldYield.getOperand(idx);
        });

    // Build the slimmer op (keeping the inputs, hence the body block args) and
    // move the body into it, then update the moved yield to carry only the used
    // values.
    auto newOp = DeviceGraphStagingOp::create(
        rewriter, op.getLoc(), TypeRange(usedResultTypes), op.getInputs());
    newOp.getBody().takeBody(op.getBody());
    DeviceGraphStagingYieldOp newYield = newOp.getYieldOp();
    rewriter.modifyOpInPlace(newYield, [&]() {
      newYield.getOperandsMutable().assign(newYieldOperands);
    });

    // Map the surviving results back onto the original result positions.
    SmallVector<Value> replacements(op.getNumResults());
    for (auto [newIdx, oldIdx] : llvm::enumerate(usedResultIndices))
      replacements[oldIdx] = newOp.getResult(newIdx);
    rewriter.replaceOp(op, replacements);
    return success();
  }
};

/// De-duplicates results of an `mgp.device_graph.staging` op whose body
/// `staging.yield` carries the same value more than once: the yield keeps one
/// copy and every duplicate result is rewritten to the surviving one.
struct DeduplicateStagingResults
    : public OpRewritePattern<DeviceGraphStagingOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(DeviceGraphStagingOp op,
                                PatternRewriter &rewriter) const override {
    DeviceGraphStagingYieldOp oldYield = op.getYieldOp();

    // Map each distinct yielded value to the index it will occupy in the
    // de-duplicated result list, and record where each old result lands.
    llvm::SmallDenseMap<Value, unsigned> firstOccurrence;
    SmallVector<Value> dedupedOperands;
    SmallVector<Type> dedupedTypes;
    SmallVector<unsigned> resultRemap(op.getNumResults());
    for (auto [idx, operand] : llvm::enumerate(oldYield.getOperands())) {
      auto [it, inserted] =
          firstOccurrence.try_emplace(operand, dedupedOperands.size());
      if (inserted) {
        dedupedOperands.push_back(operand);
        dedupedTypes.push_back(op.getResult(idx).getType());
      }
      resultRemap[idx] = it->second;
    }

    // Nothing to do if there are no duplicated yield operands.
    if (dedupedOperands.size() == op.getNumResults())
      return failure();

    // Build the de-duplicated op (keeping the inputs, hence the body block
    // args) and move the body into it, then update the moved yield to carry
    // only the surviving values.
    auto newOp = DeviceGraphStagingOp::create(
        rewriter, op.getLoc(), TypeRange(dedupedTypes), op.getInputs());
    newOp.getBody().takeBody(op.getBody());
    DeviceGraphStagingYieldOp newYield = newOp.getYieldOp();
    rewriter.modifyOpInPlace(newYield, [&]() {
      newYield.getOperandsMutable().assign(dedupedOperands);
    });

    // Point each original result (including the duplicates) at its surviving
    // de-duplicated result.
    SmallVector<Value> replacements =
        llvm::map_to_vector(resultRemap, [&](unsigned newIdx) -> Value {
          return newOp.getResult(newIdx);
        });
    rewriter.replaceOp(op, replacements);
    return success();
  }
};

} // namespace

void DeviceGraphStagingOp::getCanonicalizationPatterns(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.add<RemoveUnusedStagingResults, DeduplicateStagingResults>(ctx);
}

//===----------------------------------------------------------------------===//
// DeviceGraphStagingYieldOp
//===----------------------------------------------------------------------===//

void DeviceGraphStagingYieldOp::build(OpBuilder &, OperationState &) {
  // No-operand stub for the parent's SingleBlockImplicitTerminator; operands
  // are populated after insertion.
}
