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

#include "GraphCompiler/Interfaces/EmitMojoState.h"
#include "GraphCompiler/MGPDialect/Dialect/MGPOps.h"
#include "GraphCompiler/MGPDialect/Dialect/MGPTypes.h"
#include "GraphCompiler/MOGGDialect/Analysis/MojoRegistry.h"
#include "GraphCompiler/MOGGDialect/Support/MojoBuilder.h"
#include "GraphCompiler/MOGGDialect/Support/MojoIdentifiers.h"
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

  // Block argument types are [builder, context, inputs...]. The op creates the
  // builder from its `deviceContext` operand and passes it in as the first
  // block arg; it instantiates the recorded builder into `outGraph` itself, so
  // the body neither creates the builder nor yields the graph. The
  // `!mgp.context` block arg (passed through from the `theContext` operand) is
  // build-time only
  // -- it feeds in-body `mgp.buffer.get_cached` reads and is not part of the
  // signature. The graph's outputs are produced inside the body
  // (value-returning), so they have no operand and no block arg.
  //
  // The input arguments carry the `inputs` operand types verbatim (bare `T`, no
  // `!mogg.async` wrapping at the boundary): by the time a build body is
  // assembled the async pack/unpack mediation has been folded away.
  if (body.getNumArguments() != 2 + getInputs().size())
    return emitOpError("build body has ")
           << body.getNumArguments() << " block argument(s), expected "
           << (2 + getInputs().size())
           << " (the builder, the context, and one per input)";

  if (!llvm::isa<DeviceGraphBuilderType>(body.getArgument(0).getType()))
    return emitOpError("build body argument #0 must be a "
                       "`!mgp.device_graph.builder`, but is ")
           << body.getArgument(0).getType();

  if (body.getArgument(1).getType() != getTheContext().getType())
    return emitOpError(
               "build body argument #1 must be the `theContext` operand "
               "type ")
           << getTheContext().getType() << ", but is "
           << body.getArgument(1).getType();

  // `inputs` matched against the body args after the builder and context.
  for (auto [idx, operandType] : llvm::enumerate(getInputs().getTypes())) {
    Type argType = body.getArgument(idx + 2).getType();
    if (argType != operandType)
      return emitOpError("build body argument #")
             << (idx + 2) << " has type " << argType << ", expected "
             << operandType;
  }

  // The result device_graph's signature must agree with the op: the signature
  // inputs match the `inputs` operands, and the signature results are the
  // graph's (value-returning) outputs, which the build body yields -- so they
  // match the `mgp.device_graph.yield` operand types.
  FunctionType sig = getOutGraph().getType().getSignature();
  if (!llvm::equal(sig.getInputs(), getInputs().getTypes()))
    return emitOpError("result `!mgp.device_graph` signature inputs do not "
                       "match the op's `inputs` operand types");

  auto yieldOp = cast<DeviceGraphYieldOp>(body.back());
  if (!llvm::equal(sig.getResults(), yieldOp.getOutputs().getTypes()))
    return emitOpError("result `!mgp.device_graph` signature results do not "
                       "match the build body's yielded value types");

  return success();
}

void DeviceGraphCreateOp::build(
    OpBuilder &builder, OperationState &result, Value theContext,
    Value deviceContext, ValueRange inputs,
    llvm::function_ref<void(OpBuilder &, Location, Value, Value, ValueRange)>
        bodyBuilder) {
  OpBuilder::InsertionGuard guard(builder);

  // Operand order mirrors `arguments`: context, device context, inputs.
  result.addOperands({theContext, deviceContext});
  result.addOperands(inputs);

  // Create the body with the device-graph-builder block argument, then the
  // `!mgp.context` block argument, then one bare-`T` argument per input, and
  // let the callback record nodes into the builder and yield the graph's output
  // values. createBlock also positions the insertion point.
  Region *bodyRegion = result.addRegion();
  Block *body = builder.createBlock(bodyRegion);
  Value graphBuilder = body->addArgument(
      DeviceGraphBuilderType::get(builder.getContext()), result.location);
  Value contextArg = body->addArgument(theContext.getType(), result.location);
  SmallVector<Value> inputArgs;
  inputArgs.reserve(inputs.size());
  for (Value input : inputs)
    inputArgs.push_back(body->addArgument(input.getType(), result.location));
  if (bodyBuilder)
    bodyBuilder(builder, result.location, graphBuilder, contextArg, inputArgs);

  // Supply the operandless `mgp.device_graph.yield` terminator (no graph
  // outputs) if the callback did not emit one.
  if (!body->mightHaveTerminator())
    ensureTerminator(*bodyRegion, builder, result.location);

  // The sole result is the device graph, whose signature is
  // `(type(inputs)) -> (type(yielded outputs))`.
  auto yieldOp = cast<DeviceGraphYieldOp>(body->getTerminator());
  auto signature = FunctionType::get(builder.getContext(), inputs.getTypes(),
                                     yieldOp.getOutputs().getTypes());
  result.addTypes(DeviceGraphType::get(builder.getContext(), signature));
}

// Emits `var graph_N = <deviceContext>.create_device_graph(build_N)`, where
// `build_N` is a recording closure wrapping the build body. `create` is
// IsolatedFromAbove with block args `[builder, context, inputs...]`, but the
// Mojo `create_device_graph(build)` only passes the builder to the closure; the
// context and inputs are captured lexically from the enclosing kernel-stub
// function, so we bind those block args to their operands' Mojo values.
LogicalResult DeviceGraphCreateOp::emitMojo(MOGG::EmitMojoState &state) {
  using namespace MOGG;

  Block &body = *getBodyBuilderBlock();
  Value builderArg = body.getArgument(0);
  Value contextArg = body.getArgument(1);

  // The closure parameter: `mut builder: DeviceGraphBuilder` (origin inferred).
  NamedMojoDef builderParam =
      NamedMojoDef::asVerbatim("builder", MojoValue(kStructDeviceGraphBuilder));
  builderParam.argConvention = "mut";

  bool errOccurred = false;
  // The build closure only reads its captured values (the context and inputs);
  // it records into the builder parameter. Capture convention: `read`.
  MojoValue buildClosure = state.builder.lambda(
      "build", /*parameters=*/{}, /*arguments=*/{builderParam},
      /*returnType=*/std::nullopt, /*isInline=*/false, /*isParameter=*/false,
      /*isAsync=*/false, /*captureConvention=*/"imm",
      [&](const FunctionBody &functionBody) {
        // Builder block arg -> the closure's parameter.
        state.setMojoValueFor(builderArg, functionBody.arguments.front());
        // Context block arg -> the build-time context operand (captured).
        state.setMojoValueFor(contextArg,
                              state.getMojoValueFor(getTheContext()));
        // Input block args -> their operands' values (captured).
        for (auto [arg, operand] :
             llvm::zip_equal(body.getArguments().drop_front(2), getInputs()))
          state.setMojoValueFor(arg, state.getMojoValueFor(operand));

        // Emit the recording body (dependencies ops + hoisted plumbing). The
        // `mgp.device_graph.yield` terminator emits nothing.
        for (Operation &op : body.getOperations())
          if (state.lowerOpToMojo(&op).failed()) {
            errOccurred = true;
            return;
          }
      });
  if (errOccurred)
    return failure();

  // Build the graph via the static `DeviceGraph.create(deviceContext, build)`
  // factory, which creates a builder, runs the build closure, and instantiates
  // the recorded graph.
  MojoValue deviceContext = state.getMojoValueFor(getDeviceContext());
  MojoValue graph =
      state.builder.var(MojoValue(kStructDeviceGraph)
                            .get(kFnDeviceGraphCreate)(
                                /*parameters=*/{},
                                /*operands=*/{deviceContext, buildClosure}));
  state.setMojoValueFor(getOutGraph(), graph);
  return success();
}

bool DeviceGraphCreateOp::isThrows() { return true; }

void DeviceGraphYieldOp::build(OpBuilder &, OperationState &) {
  // No-operand stub for the parent's `SingleBlockImplicitTerminator`; operands
  // are populated after insertion.
}

// Register each yielded value as an output of the device graph. The recording
// wrote into these buffers/tensors; pack each into a fresh `AsyncValue` (via
// the composite's `async_pack`) and hand it to the builder's `add_output` so
// its backing memory outlives the graph that references it.
LogicalResult DeviceGraphYieldOp::emitMojo(MOGG::EmitMojoState &state) {
  using namespace MOGG;

  // The builder is the first block argument of the enclosing `create` body.
  Value builderArg = getOperation()->getBlock()->getArgument(0);
  MojoValue builder = state.getMojoValueFor(builderArg);

  for (Value output : getOutputs()) {
    MojoValue outVal = state.getMojoValueFor(output);
    MojoValue asyncValue =
        outVal.get(kFnAsyncPack)(/*parameters=*/{}, /*operands=*/{});
    state.builder.call(builder.get(kFnDeviceGraphBuilderAddOutput),
                       /*parameters=*/{}, /*operands=*/{asyncValue});
  }
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
    return emitOpError("`outputs` result types do not match the graph's "
                       "signature results");

  return success();
}

//===----------------------------------------------------------------------===//
// DeviceGraphAllocOp
//===----------------------------------------------------------------------===//

// Emits `var buf_N = mgp_device_graph_alloc(byteSize, builder)`. Mirrors
// `BufferAllocOp::emitMojo`, but passes the device-graph builder as the
// allocation handle instead of a raw device context.
LogicalResult DeviceGraphAllocOp::emitMojo(MOGG::EmitMojoState &state) {
  using namespace MOGG;
  // The primitive's `sym_name` (set by ImportPackages) is the Mojo function to
  // call; look it up by the op name it registered under.
  auto primDecl = state.registry.lookupPrimitive(getOperationName());
  MojoValue function(primDecl.getSymNameAttr().strref());
  MojoValue result = state.builder.var(function(
      /*parameters=*/{},
      /*operands=*/{state.getMojoValueFor(getRuntimeSize()),
                    state.getMojoValueFor(getGraphBuilder())}));
  state.setMojoValueFor(getOutBuffer(), result);
  return success();
}

// The `mgp_device_graph_alloc` primitive raises (it allocates a device buffer).
bool DeviceGraphAllocOp::isThrows() { return true; }

//===----------------------------------------------------------------------===//
// DeviceGraphRecordingContextOp
//===----------------------------------------------------------------------===//

// Emits `var ctx_N = builder.recording_context()`: the recording device-context
// view of the builder that execute-only kernels dispatch their `execute`
// against.
LogicalResult
DeviceGraphRecordingContextOp::emitMojo(MOGG::EmitMojoState &state) {
  using namespace MOGG;
  MojoValue builder = state.getMojoValueFor(getGraphBuilder());
  MojoValue result =
      state.builder.var(builder.get(kFnDeviceGraphBuilderRecordingContext)());
  state.setMojoValueFor(getContext(), result);
  return success();
}

// `recording_context()` raises (it asks the driver for a recording context).
bool DeviceGraphRecordingContextOp::isThrows() { return true; }

//===----------------------------------------------------------------------===//
// DeviceGraphDependenciesOp
//===----------------------------------------------------------------------===//

DeviceGraphDependenciesYieldOp DeviceGraphDependenciesOp::getYieldOp() {
  return cast<DeviceGraphDependenciesYieldOp>(
      getBody().front().getTerminator());
}

TypedValue<DeviceGraphBuilderType>
DeviceGraphDependenciesOp::getBuilderBlockArgument() {
  return cast<TypedValue<DeviceGraphBuilderType>>(
      getBody().front().getArgument(0));
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

bool DeviceGraphDependenciesOp::producesNewNodes() {
  return !getBuilderBlockArgument().use_empty();
}

// Emits `var node_N = <builder>.region(work_N, dependencies=[...])`, where
// `work_N` is a recording closure wrapping the body. The op is not
// IsolatedFromAbove, so the body captures enclosing values lexically; only the
// builder block arg maps to the closure parameter. The `dependencies=` keyword
// argument injects the predecessor node edges as ambient predecessors of every
// node the body records (the builder's `_implicit_deps` mechanism).
//
// `DeviceGraphBuilder.region` returns only the join node, so a body result
// produced *inside* the body cannot be returned directly. We surface each such
// result through a captured outer `Optional[T]` var: declare it (None) before
// the closure, write it from inside `work` (captured by `mut`), and unwrap it
// after `region` returns. `Optional` is used because the surfaced type may not
// be default-constructible (e.g. a non-null buffer pointer), which a `mut`
// capture requires:
//
//   var out = Optional[T](None)
//   def work(mut builder) raises {mut}:
//     ...
//     out = <yielded>
//   var node = builder.region(work, dependencies=[...])
//   var resolved = out.take()
//   // uses of the op's body result now refer to `resolved`
LogicalResult DeviceGraphDependenciesOp::emitMojo(MOGG::EmitMojoState &state) {
  using namespace MOGG;

  Value builderArg = getBuilderBlockArgument();
  DeviceGraphDependenciesYieldOp yieldOp = getYieldOp();

  // Partition the used body results into those produced inside the body (which
  // need a captured outer var) and external pass-throughs (bound directly).
  struct SurfacedResult {
    Value result;
    Value yielded;
    MojoValue outVar;
  };

  SmallVector<SurfacedResult> surfaced;
  for (auto [result, yielded] :
       llvm::zip_equal(getBodyResults(), yieldOp.getOperands())) {
    if (result.use_empty())
      continue;

    // The returned value may not be default constructable, so use
    // `Optional[T]` for the result variable.
    MojoValue optionalType = MojoValue(
        kStructOptional)[EmitMojoState::mlirTypeToMojoType(result.getType())];
    MojoValue outVar =
        state.builder.var(optionalType(/*parameters=*/{},
                                       /*operands=*/{MojoValue(kComptimeNone)}),
                          /*cache=*/false);
    surfaced.emplace_back(result, yielded, outVar);
  }

  // The closure parameter: `mut builder: DeviceGraphBuilder` (origin inferred).
  NamedMojoDef builderParam =
      NamedMojoDef::asVerbatim("builder", MojoValue(kStructDeviceGraphBuilder));
  builderParam.argConvention = "mut";

  // A closure that surfaces in-body results writes its captured `Optional`
  // var(s), so it must capture by `mut`; otherwise it only reads its captures,
  // so `read`. (A surfacing closure that also *reads* register-passable
  // captures would need a finer per-capture convention; that does not arise
  // for the current single-dispatch bodies.)
  llvm::StringRef workConvention =
      surfaced.empty() ? llvm::StringRef("imm") : llvm::StringRef("mut");

  bool errOccurred = false;
  MojoValue workClosure = state.builder.lambda(
      "work", /*parameters=*/{}, /*arguments=*/{builderParam},
      /*returnType=*/std::nullopt, /*isInline=*/false, /*isParameter=*/false,
      /*isAsync=*/false, /*captureConvention=*/workConvention,
      [&](const FunctionBody &functionBody) {
        state.setMojoValueFor(builderArg, functionBody.arguments.front());
        for (Operation &op : getBodyBlock().getOperations())
          if (state.lowerOpToMojo(&op).failed()) {
            errOccurred = true;
            return;
          }
        // Write each in-body result into its captured `Optional` var (the
        // value implicitly wraps into the Optional).
        for (const SurfacedResult &s : surfaced)
          state.builder.assign(s.outVar,
                               state.getMojoValueFor(s.yielded).transfer());
      });
  if (errOccurred)
    return failure();

  // dependencies=[...] from the predecessor node operands.
  SmallVector<MojoValue> depValues = state.getMojoValuesFor(getDependencies());
  MojoValue dependenciesList = state.builder.listLiteral(depValues);

  MojoValue builder = state.getMojoValueFor(getGraphBuilder());
  MojoValue node = state.builder.var(builder.get(kFnDeviceGraphRegion)(
      /*parameters=*/{},
      /*operands=*/{
          workClosure,
          MojoValue(kParameterDependencies).equal(dependenciesList)}));
  state.setMojoValueFor(getNode(), node);

  // Bind the op's body results: unwrap each surfaced `Optional` via `.take()`.
  for (const SurfacedResult &s : surfaced)
    state.setMojoValueFor(s.result, state.builder.var(s.outVar.get("take")(
                                        /*parameters=*/{}, /*operands=*/{})));

  return success();
}

Block &DeviceGraphDependenciesOp::getBodyBlock() { return getBody().front(); }

bool DeviceGraphDependenciesOp::isThrows() { return true; }

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

    llvm::SmallMapVector<Value, unsigned, 8> firstOccurrence;
    firstOccurrence.reserve(numElements);

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

/// Forwards `mgp.device_graph.dependencies` body results whose
/// `dependencies.yield` operand is produced *outside* the body region.
/// Rewriting those uses leaves the result dead for
/// `RemoveUnusedDependenciesResults` to drop.
struct ForwardExternalDependenciesResults
    : public OpRewritePattern<DeviceGraphDependenciesOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(DeviceGraphDependenciesOp op,
                                PatternRewriter &rewriter) const override {
    Region &body = op.getBody();
    DeviceGraphDependenciesYieldOp yieldOp = op.getYieldOp();

    bool changed = false;
    for (auto [result, yielded] :
         llvm::zip(op.getBodyResults(), yieldOp.getOperands())) {
      // Already-dead results have nothing to forward; skipping them keeps the
      // pattern idempotent (otherwise it reports a no-op rewrite as `success`
      // forever and the greedy driver never converges).
      if (result.use_empty())
        continue;

      // Keep results whose value originates inside the body (incl. the body's
      // own block arguments); only forward genuine pass-throughs of outer SSA.
      if (body.isAncestor(yielded.getParentRegion()))
        continue;

      rewriter.replaceAllUsesWith(result, yielded);
      changed = true;
    }
    return success(changed);
  }
};

/// Hoists body-invariant ops out of an `mgp.device_graph.dependencies` op. Any
/// non-terminator body op whose operands are all defined *outside* the body
/// (i.e. it depends on neither the builder block arg nor another in-body value)
/// is moved in front of the op, so the emitted `region` `work` closure captures
/// it once instead of recomputing it on every graph build. The lift is
/// transitive: once an invariant op moves out, an in-body consumer of its
/// result becomes hoistable on the same sweep. Builder-dependent ops (the
/// recording dispatch) stay in the body.
struct HoistOperations final
    : public OpRewritePattern<DeviceGraphDependenciesOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(DeviceGraphDependenciesOp dependenciesOp,
                                PatternRewriter &rewriter) const override {
    Region &body = dependenciesOp.getBody();

    auto isRegionCapture = [&body](Value operand) {
      return body.isAncestor(operand.getParentRegion());
    };

    bool changed = false;
    for (Operation &op : llvm::make_early_inc_range(body.getOps())) {
      if (op.hasTrait<OpTrait::IsTerminator>())
        continue;

      WalkResult result = op.walk([&](Operation *inner) -> WalkResult {
        bool canHoist = llvm::none_of(inner->getOperands(), isRegionCapture);

        if (!canHoist)
          return WalkResult::interrupt();
        return WalkResult::advance();
      });

      if (!result.wasInterrupted()) {
        rewriter.moveOpBefore(&op, dependenciesOp);
        changed = true;
      }
    }

    return success(changed);
  }
};

/// Bypasses empty (node-less) `mgp.device_graph.dependencies` producers: a
/// dependency edge pointing at a producer that records no nodes is replaced by
/// that producer's own dependency edges, preserving the transitive ordering the
/// empty op stood in for. Applied to a fixpoint this splices past chains of
/// empty producers; the bypassed ops are left for DCE once their `node` result
/// is unused, and any duplicate edges introduced are folded away by the op's
/// edge-dedup folder.
struct BypassEmptyDependencies final
    : public OpRewritePattern<DeviceGraphDependenciesOp> {
  using Base::Base;

  /// Get the forwarded transitive dependencies for a dependency node which
  /// corresponds to an empty region.
  static std::optional<ValueRange> getTransitiveDependencies(Value dependency) {
    if (auto producer = dependency.getDefiningOp<DeviceGraphDependenciesOp>())
      if (!producer.producesNewNodes())
        return producer.getDependencies();

    return {};
  }

  LogicalResult matchAndRewrite(DeviceGraphDependenciesOp dependenciesOp,
                                PatternRewriter &rewriter) const override {
    llvm::SmallVector<Value> newDependencies;

    bool changed = false;
    for (auto dependency : dependenciesOp.getDependencies()) {
      if (auto transitiveDeps = getTransitiveDependencies(dependency)) {
        llvm::append_range(newDependencies, *transitiveDeps);
        changed = true;
      } else {
        newDependencies.push_back(dependency);
      }
    }

    if (!changed)
      return rewriter.notifyMatchFailure(dependenciesOp,
                                         "no dependencies to remove");

    rewriter.modifyOpInPlace(dependenciesOp, [&] {
      dependenciesOp.getDependenciesMutable().assign(newDependencies);
    });

    return success();
  }
};

} // namespace

void DeviceGraphDependenciesOp::getCanonicalizationPatterns(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  // clang-format off
  patterns.add<
    BypassEmptyDependencies,
    DeduplicateDependenciesResults,
    ForwardExternalDependenciesResults,
    HoistOperations,
    RemoveUnusedDependenciesResults
  >(ctx);
  // clang-format on
}

//===----------------------------------------------------------------------===//
// DeviceGraphDependenciesYieldOp
//===----------------------------------------------------------------------===//

void DeviceGraphDependenciesYieldOp::build(OpBuilder &, OperationState &) {
  // No-operand stub for the parent's SingleBlockImplicitTerminator; operands
  // are populated after insertion.
}

// The body results are surfaced by the parent op's `emitMojo` (region join node
// plus external pass-throughs), so the yield itself emits nothing.
LogicalResult DeviceGraphDependenciesYieldOp::emitMojo(MOGG::EmitMojoState &) {
  return success();
}

//===----------------------------------------------------------------------===//
// DeviceGraphStagingOp
//===----------------------------------------------------------------------===//

DeviceGraphStagingYieldOp DeviceGraphStagingOp::getYieldOp() {
  return cast<DeviceGraphStagingYieldOp>(getBody().front().getTerminator());
}

LogicalResult DeviceGraphStagingOp::verify() {
  Block &body = getBody().front();

  // Block args are [builder, input_0, ..., input_n-1]: the builder has no
  // operand; the rest carry the `inputs` operand types verbatim (bare `T`, no
  // `!mogg.async` wrapping at the staging boundary).
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
    if (argType != inputType)
      return emitOpError("staging body argument #")
             << (idx + 1) << " has type " << argType << ", expected "
             << inputType;
  }

  // The yield carries the staged results out as bare `T`, matching the op's
  // results exactly.
  DeviceGraphStagingYieldOp yieldOp = getYieldOp();
  if (yieldOp.getNumOperands() != getNumResults())
    return emitOpError("`mgp.device_graph.staging.yield` has ")
           << yieldOp.getNumOperands() << " operand(s), expected "
           << getNumResults();
  for (auto [idx, resultType] :
       llvm::enumerate(getStagedResults().getTypes())) {
    Type operandType = yieldOp.getOperand(idx).getType();
    if (operandType != resultType)
      return emitOpError("`mgp.device_graph.staging.yield` operand #")
             << idx << " has type " << operandType << ", expected "
             << resultType;
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
  // one bare-`T` argument per input, and let the callback populate it (it is
  // expected to emit the staging.yield carrying the staged values).
  // createBlock also positions the insertion point.
  Region *bodyRegion = result.addRegion();
  Block *body = builder.createBlock(bodyRegion);
  Value graphBuilder = body->addArgument(
      DeviceGraphBuilderType::get(builder.getContext()), result.location);
  SmallVector<Value> inputArgs;
  inputArgs.reserve(inputs.size());
  for (Value input : inputs)
    inputArgs.push_back(body->addArgument(input.getType(), result.location));
  if (bodyBuilder)
    bodyBuilder(builder, result.location, graphBuilder, inputArgs);

  // Supply the operandless terminator (zero staged results) if the callback did
  // not emit one.
  if (!body->mightHaveTerminator())
    ensureTerminator(*bodyRegion, builder, result.location);

  // The op's results are exactly the staging.yield operand types (bare `T`).
  auto yieldOp = cast<DeviceGraphStagingYieldOp>(body->getTerminator());
  result.addTypes(yieldOp.getOperandTypes());
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
        rewriter, op.getLoc(), TypeRange(usedResultTypes), op.getInputs(),
        op.getConstantResources());

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
        rewriter, op.getLoc(), TypeRange(dedupedTypes), op.getInputs(),
        op.getConstantResources());

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
