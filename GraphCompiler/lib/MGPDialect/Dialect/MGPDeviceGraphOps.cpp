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
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace M;
using namespace MGP;

/// The device a tensor/buffer graph input lives on. Both types default their
/// `#M.device_ref` parameter to the host, so an omitted device reads as CPU.
static DeviceRefAttr graphInputDevice(Value v) {
  return llvm::TypeSwitch<Type, DeviceRefAttr>(v.getType())
      .Case<MGP::DeviceTensorType, MGP::DeviceBufferType>(
          [](auto type) { return type.getDevice(); })
      .Default(nullptr);
}

/// Is `v` one of the graph's device-resident inputs, as opposed to a context
/// operand? Device-graph finalization strips only chains from `inputs`, so
/// `!mgp.context` and `!mgp.device_context` values ride along in there too.
///
/// This is the single definition of "graph input" for both the stable locations
/// `emitMojo` requests and the cache key it derives. The runtime counterpart is
/// `graphInputData` in MGPPrimitives.cpp, which selects the same operands by
/// their runtime type; the two must agree or inputs pair with the wrong stable
/// buffers at replay. Mutable inputs preserve that pairing by registering an
/// in-place marker (a null entry) instead of a stable buffer, so every device
/// input still occupies exactly one position in `graph->getInputs()`.
static bool isDeviceGraphInput(Value v) {
  return graphInputDevice(v) != nullptr;
}

/// For now, the device graph workflow assumes that host tensor inputs are used
/// purely to encode launch parameters for device computation. As such their
/// contents are integrated into the cache key. A large host input will likely
/// result in a large device graph cache, so warn uses until we have a knob to
/// control this behavior.
static constexpr uint64_t kHostInputWarnBytes = 1024;

/// Verifies the residency rules for the graph's tensor/buffer inputs.
///
/// - Device-resident tensors are copied into device-pool stable locations (or
///   recorded in place when `mut`).
///
/// - Host-placed *immutable* tensors are allowed: they get a host-pool stable
///   location populated before recording, and their contents join the cache
///   key, because recorded ops read host values at enqueue time.
///
/// - Host-placed `mut` tensors are rejected -- in-place writes to
///   a value-keyed input would invalidate its own cache entry -- as are
///   host-placed raw buffers, whose lowering to a mut tensor should have
///   happened earlier.
static LogicalResult verifyInputResidency(Operation *op, OperandRange inputs) {
  for (auto [idx, input] : llvm::enumerate(inputs)) {
    DeviceRefAttr device = graphInputDevice(input);
    if (!device || !M::isHostPlaced(device))
      continue;

    auto tensorType = dyn_cast<MGP::DeviceTensorType>(input.getType());
    if (!tensorType)
      return op->emitOpError("input #")
             << idx << " is a buffer on " << device.toString()
             << ", but host-placed device graph inputs must be immutable "
                "tensors";
    if (tensorType.getMut())
      return op->emitOpError("input #")
             << idx << " is a mutable tensor on " << device.toString()
             << "; in-place writes contradict the value-keyed replay of a "
                "host-placed input";

    std::optional<uint64_t> byteSize =
        tensorType.getTensorSpec().getSizeInBytes();
    if (!byteSize)
      continue; // Dynamic shape: the runtime keys whatever size shows up.
    if (*byteSize > kHostInputWarnBytes)
      op->emitWarning("input #")
          << idx << " is a host tensor of " << *byteSize
          << " bytes; its contents are baked into the graph cache key, so "
             "every distinct value records a new graph";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// DeviceGraphCreateOp
//===----------------------------------------------------------------------===//

LogicalResult DeviceGraphCreateOp::verify() {
  Block &body = getBodyBuilderBlock();

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

  return verifyInputResidency(*this, getInputs());
}

void DeviceGraphCreateOp::build(
    OpBuilder &builder, OperationState &result, Value theContext,
    Value deviceContext, Value cache, ValueRange inputs,
    llvm::function_ref<void(OpBuilder &, Location, Value, Value, ValueRange)>
        bodyBuilder) {
  OpBuilder::InsertionGuard guard(builder);

  // The body is assembled in a detached region because the generated builder
  // below needs the result type up front, and the graph signature's results are
  // whatever the body ends up yielding.
  Region bodyRegion;

  // The body takes the device-graph-builder block argument, then the
  // `!mgp.context` block argument, then one bare-`T` argument per input, and
  // the callback records nodes into the builder and yields the graph's output
  // values. createBlock also positions the insertion point.
  Block *body = builder.createBlock(&bodyRegion);
  Value graphBuilder = body->addArgument(
      DeviceGraphBuilderType::get(builder.getContext()), result.location);

  Value contextArg = body->addArgument(theContext.getType(), result.location);

  SmallVector<Location> locations(inputs.size(), result.location);
  auto inputArgs = body->addArguments(TypeRange(inputs), locations);

  if (bodyBuilder)
    bodyBuilder(builder, result.location, graphBuilder, contextArg,
                ValueRange(inputArgs));

  // Supply the operandless `mgp.device_graph.yield` terminator (no graph
  // outputs) if the callback did not emit one.
  if (!body->mightHaveTerminator())
    ensureTerminator(bodyRegion, builder, result.location);

  // The sole result is the device graph, whose signature is
  // `(type(inputs)) -> (type(yielded outputs))`.
  auto yieldOp = cast<DeviceGraphYieldOp>(body->getTerminator());
  auto signature = FunctionType::get(builder.getContext(), inputs.getTypes(),
                                     yieldOp.getOutputs().getTypes());

  // Operands, operand segment sizes and the result type are all the generated
  // builder's job; it also adds the (empty) build body region that the
  // assembled body moves into.
  build(builder, result, DeviceGraphType::get(builder.getContext(), signature),
        theContext, deviceContext, cache, inputs);
  result.regions.back()->takeBody(bodyRegion);
}

// Emits `var graph_N = <deviceContext>.create_device_graph(build_N)`, where
// `build_N` is a recording closure wrapping the build body. `create` is
// IsolatedFromAbove with block args `[builder, context, inputs...]`, but the
// Mojo `create_device_graph(build)` only passes the builder to the closure; the
// context and inputs are captured lexically from the enclosing kernel-stub
// function, so we bind those block args to their operands' Mojo values.
LogicalResult DeviceGraphCreateOp::emitMojo(MOGG::EmitMojoState &state) {
  using namespace MOGG;

  Block &body = getBodyBuilderBlock();
  Value builderArg = getBodyBuilder();
  Value contextArg = getBodyContext();
  MutableArrayRef<BlockArgument> bodyInputs = getBodyInputs();

  // The closure parameter: `mut builder: DeviceGraphBuilder` (origin inferred).
  NamedMojoDef builderParam =
      NamedMojoDef::asVerbatim("builder", MojoValue(kStructDeviceGraphBuilder));
  builderParam.argConvention = "mut";

  bool errOccurred = false;
  // The build closure only reads its captured values (the context and inputs);
  // it records into the builder parameter. Capture convention: `read`.
  MojoValue buildClosure = state.builder.lambda(
      "build", /*parameters=*/{}, /*arguments=*/{builderParam},
      /*returnType=*/std::nullopt,
      // Never inline: the build body sequences every recorded kernel of the
      // model, and inlining its callees back in produces one enormous
      // function whose Mem2Reg dominates model compile time.
      MojoBuilder::LambdaInline::kNever, /*isParameter=*/false,
      /*isAsync=*/false, /*raises=*/true, /*captureConvention=*/"var",
      [&](const FunctionBody &functionBody) {
        // Builder block arg -> the closure's parameter.
        state.setMojoValueFor(builderArg, functionBody.arguments.front());
        // Context block arg -> the build-time context operand (captured).
        state.setMojoValueFor(contextArg,
                              state.getMojoValueFor(getTheContext()));
        // Input block args -> what the graph records against. Every device
        // input routes through `builder.add_input` uniformly; the input's
        // Mojo type decides what that means. An immutable tensor gets a
        // stable twin from the graph's pool and `mgp.device_graph.execute`
        // copies the live input in before replaying. A mutable tensor
        // (`OwnedTensor[.., mut=True]`, the lowered form of `!mo.buffer` --
        // a KV cache) must NOT be given a private location because the graph
        // writes it in place.
        DeviceGraphBuilderRef builderValue(functionBody.arguments.front());
        for (auto [arg, operand] : llvm::zip_equal(bodyInputs, getInputs())) {
          if (!isDeviceGraphInput(operand)) {
            state.setMojoValueFor(arg, state.getMojoValueFor(operand));
            continue;
          }

          MojoValue stable = state.builder.var(
              builderValue.addInput(state.getMojoValueFor(operand)));
          state.setMojoValueFor(arg, stable);
        }

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

  SmallVector<MojoValue> inputs = {deviceContext, buildClosure};

  // The inputs exist only to key the cache, so they go along with it: the
  // uncached `create` overload takes none. With a cache, `create` returns a
  // previously built graph when the key hits and the build closure is never
  // called.
  if (Value cache = getCache()) {
    for (Value input : getInputs())
      if (isDeviceGraphInput(input))
        inputs.push_back(state.getMojoValueFor(input));

    inputs.push_back(DeviceGraphRef::cacheArg(state.getMojoValueFor(cache)));
  }

  MojoValue graph = state.builder.var(DeviceGraphRef::create(inputs));
  state.setMojoValueFor(getOutGraph(), graph);
  return success();
}

bool DeviceGraphCreateOp::isThrows() { return true; }

// Emits `var cache_N = Pointer(to=<ctx>.get_device_graph_cache())`. The Mojo
// method hands back a reference to the cache the state context owns, and
// `DeviceGraph.create` takes it by pointer; `DeviceGraphCache` is not copyable,
// so the reference cannot be bound to a `var` directly.
LogicalResult DeviceGraphCacheOp::emitMojo(MOGG::EmitMojoState &state) {
  using namespace MOGG;

  MojoValue theContext = state.getMojoValueFor(getTheContext());
  MojoValue cache = state.builder.var(MojoValue(kStructPointer)(
      /*parameters=*/{},
      /*operands=*/{
          MojoValue(kParameterTo)
              .equal(StateContextRef(theContext).getDeviceGraphCache())}));
  state.setMojoValueFor(getOutCache(), cache);
  return success();
}

bool DeviceGraphCacheOp::isThrows() { return true; }

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
  DeviceGraphBuilderRef builder(state.getMojoValueFor(builderArg));

  for (Value output : getOutputs()) {
    MojoValue outVal = state.getMojoValueFor(output);
    MojoValue asyncValue =
        outVal.get(kFnAsyncPack)(/*parameters=*/{}, /*operands=*/{});
    state.builder.call(builder.addOutput(),
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

  return verifyInputResidency(*this, getInputs());
}

//===----------------------------------------------------------------------===//
// DeviceGraphAllocOp
//===----------------------------------------------------------------------===//

LogicalResult DeviceGraphAllocOp::emitMojo(MOGG::EmitMojoState &state) {
  using namespace MOGG;
  // The primitive's `sym_name` (set by ImportPackages) is the Mojo function to
  // call; look it up by the op name it registered under.
  auto primDecl = state.registry.lookupPrimitive(getOperationName());
  MojoValue function(primDecl.getSymNameAttr().strref());

  DeviceRefAttr hostDevice = M::getHostDevice(getContext());
  // Pass the host parameter to differentiate device from host allocations.
  MojoValue isHostParam(getOutBuffer().getType().getDevice() == hostDevice);
  MojoValue result = state.builder.var(function(
      /*parameters=*/{isHostParam},
      /*operands=*/{state.getMojoValueFor(getRuntimeSize()),
                    state.getMojoValueFor(getGraphBuilder())}));
  state.setMojoValueFor(getOutBuffer(), result);
  return success();
}

// The `mgp_device_graph_alloc` primitive raises (it allocates a buffer).
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
  DeviceGraphBuilderRef builder(state.getMojoValueFor(getGraphBuilder()));
  MojoValue result = state.builder.var(builder.recordingContext());
  state.setMojoValueFor(getContext(), result);
  return success();
}

// `recording_context()` raises (it asks the driver for a recording context).
bool DeviceGraphRecordingContextOp::isThrows() { return true; }

//===----------------------------------------------------------------------===//
// DeviceGraphDeviceContextOp
//===----------------------------------------------------------------------===//

// Emits `var ctx_N = builder.context()`: the backing device context the builder
// was created from. `context()` is a non-raising field read, so `isThrows`
// keeps the interface default of false.
LogicalResult DeviceGraphDeviceContextOp::emitMojo(MOGG::EmitMojoState &state) {
  using namespace MOGG;
  DeviceGraphBuilderRef builder(state.getMojoValueFor(getGraphBuilder()));
  MojoValue result = state.builder.var(builder.context());
  state.setMojoValueFor(getDeviceContext(), result);
  return success();
}

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
// argument injects the predecessor node edges as predecessors of the first node
// the body records (the builder's `_implicit_deps` mechanism); `region` chains
// the rest after it, so body order is the recording order.
//
// `DeviceGraphBuilder.region` returns only the body's last node, so a body
// result produced *inside* the body cannot be returned directly. We surface
// each such result through a captured outer `Optional[T]` var: declare it
// (None) before the closure, write it from inside `work` (captured by `mut`),
// and unwrap it after `region` returns. `Optional` is used because the surfaced
// type may not be default-constructible (e.g. a non-null buffer pointer), which
// a `mut` capture requires:
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
      /*returnType=*/std::nullopt,
      // Never inline: single-use per-kernel bodies otherwise merge into the
      // build closure (see the build lambda in DeviceGraphCreateOp).
      MojoBuilder::LambdaInline::kNever, /*isParameter=*/false,
      /*isAsync=*/false, /*raises=*/true, /*captureConvention=*/workConvention,
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

  DeviceGraphBuilderRef builder(state.getMojoValueFor(getGraphBuilder()));
  MojoValue node =
      state.builder.var(builder.region(workClosure, dependenciesList));
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

/// Absorbs the body of an `mgp.device_graph.dependencies` op into its sole
/// successor, when the two form a link in a linear chain.
struct MergeChainedDependencies final
    : public OpRewritePattern<DeviceGraphDependenciesOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(DeviceGraphDependenciesOp op,
                                PatternRewriter &rewriter) const override {
    // Exactly one edge, so absorbing its producer adds no ordering: the
    // producer's edges become this op's, and nothing else changes. Absorbing a
    // body into an op that has other edges would order that body after them
    // too, collapsing genuine fan-in.
    if (op.getDependencies().size() != 1)
      return failure();

    auto producer =
        op.getDependencies().front().getDefiningOp<DeviceGraphDependenciesOp>();
    if (!producer)
      return failure();

    // The producer has a single consumer, which must be the input operation.
    if (!producer.getNode().hasOneUse())
      return failure();

    // Dataflow out of the producer complicates the canonicalization. For now,
    // bail out as this case is uncommon.
    if (!producer.getBodyResults().empty())
      return failure();

    // Producer and consumer must share the same builder and parent block.
    if (producer.getGraphBuilder() != op.getGraphBuilder() ||
        producer->getBlock() != op->getBlock())
      return failure();

    // Erase the producer yield operation. The merged block will be terminated
    // by the terminator from the consumer operation.
    rewriter.eraseOp(producer.getYieldOp());

    // Inline the producer's body block before the consumer, remapping the
    // builder argument.
    rewriter.inlineBlockBefore(&producer.getBodyBlock(), &op.getBodyBlock(),
                               op.getBodyBlock().begin(),
                               {op.getBuilderBlockArgument()});

    rewriter.modifyOpInPlace(op, [&] {
      op.getDependenciesMutable().assign(producer.getDependencies());
    });
    rewriter.eraseOp(producer);

    return success();
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
    MergeChainedDependencies,
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

//===----------------------------------------------------------------------===//
// DeviceGraphStableCreateOp
//===----------------------------------------------------------------------===//

LogicalResult DeviceGraphStableCreateOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, Adaptor adaptor,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  auto tensorType = cast<DeviceTensorType>(adaptor.getTensor().getType());

  auto resultType = DeviceTensorType::get(
      tensorType.getDevice(), tensorType.getTensorSpec(), tensorType.getMut(),
      /*stable=*/true);

  inferredReturnTypes.push_back(resultType);
  return success();
}

LogicalResult DeviceGraphStableCreateOp::verify() {
  if (getTensor().getType().getStable())
    return emitOpError("input tensor is already marked `stable`");
  return success();
}

//===----------------------------------------------------------------------===//
// DeviceGraphStableGetOp
//===----------------------------------------------------------------------===//

bool DeviceGraphStableGetOp::isThrows() { return true; }

LogicalResult DeviceGraphStableGetOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, Adaptor adaptor,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  auto slotType = cast<StableAddressType>(adaptor.getAddr().getType());
  auto elementType = slotType.getElementType();

  auto resultType = DeviceTensorType::get(
      elementType.getDevice(), elementType.getTensorSpec(),
      elementType.getMut(), /*stable=*/true);

  inferredReturnTypes.push_back(resultType);
  return success();
}

LogicalResult DeviceGraphStableGetOp::emitMojo(MOGG::EmitMojoState &) {
  return emitError(
      "`mgp.device_graph.stable.get` Mojo emission is not yet implemented");
}

//===----------------------------------------------------------------------===//
// DeviceGraphStablePlaceOp
//===----------------------------------------------------------------------===//

LogicalResult DeviceGraphStablePlaceOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, Adaptor adaptor,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  auto slotType = cast<StableAddressType>(adaptor.getAddr().getType());
  auto elementType = slotType.getElementType();

  auto resultType =
      DeviceTensorType::get(elementType.getDevice(),
                            elementType.getTensorSpec(), elementType.getMut(),
                            /*stable=*/true);

  inferredReturnTypes.push_back(resultType);
  return success();
}

bool DeviceGraphStablePlaceOp::isThrows() { return true; }

LogicalResult DeviceGraphStablePlaceOp::emitMojo(MOGG::EmitMojoState &) {
  return emitError(
      "`mgp.device_graph.stable.place` Mojo emission is not yet implemented");
}

//===----------------------------------------------------------------------===//
// DeviceGraphCollectiveOp
//===----------------------------------------------------------------------===//

// Verifies the structural invariants implied by the design:
//   1. `$devices.size() == #regions`.
//   2. Every region has the same block-arg count (the per-device collective
//      bodies see the same shape of the stable-address map).
//   3. Block argument #0 of every region is a `!mgp.device_graph.builder`.
//   4. Remaining block arguments are `!mgp.stable_addr<...>` slots and match
//      across regions.
//   5. Every region terminates with `DeviceGraphCollectiveYieldOp`.
//   6. The total yielded operand count across regions equals `$outputs.size()`
//      (yields append in region order).
LogicalResult DeviceGraphCollectiveOp::verify() {
  if (getGraphs().size() != getDevices().size())
    return emitOpError("`devices` attribute has ")
           << getDevices().size() << " entries, but the op has "
           << getGraphs().size() << " regions";

  if (getDevices().empty())
    return success();

  const unsigned firstNumArgs = getRegionBlock(0).getNumArguments();
  TypeRange firstArgTypes = getRegionBlock(0).getArgumentTypes();
  unsigned outputIdx = 0;
  for (auto [idx, region] : llvm::enumerate(getGraphs())) {
    Block &block = region.front();

    if (block.getNumArguments() != firstNumArgs)
      return emitOpError("region #")
             << idx << " has " << block.getNumArguments()
             << " block argument(s), but region 0 has " << firstNumArgs
             << "; all regions must agree on the block argument shape";

    if (block.getNumArguments() == 0)
      return emitOpError("region #")
             << idx
             << " has no block arguments; expected at least a "
                "`!mgp.device_graph.builder`";

    if (!isa<DeviceGraphBuilderType>(block.getArgument(0).getType()))
      return emitOpError("region #")
             << idx << " block argument #0 has type "
             << block.getArgument(0).getType()
             << "; expected `!mgp.device_graph.builder`";

    for (auto [argIdx, blockArg] : llvm::enumerate(block.getArguments())) {
      if (argIdx == 0)
        continue;

      if (!isa<StableAddressType>(blockArg.getType()))
        return emitOpError("region #")
               << idx << " block argument #" << argIdx << " has type "
               << blockArg.getType() << "; expected `!mgp.stable_addr<...>`";

      if (blockArg.getType() != firstArgTypes[argIdx])
        return emitOpError("region #")
               << idx << " block argument #" << argIdx << " has type "
               << blockArg.getType() << "; expected the same type as region 0 ("
               << firstArgTypes[argIdx] << ")";
    }

    if (!block.mightHaveTerminator() ||
        !isa<DeviceGraphCollectiveYieldOp>(block.getTerminator()))
      return emitOpError("region #")
             << idx
             << " must terminate with `mgp.device_graph.collective.yield`";

    auto yield = cast<DeviceGraphCollectiveYieldOp>(block.getTerminator());
    for (auto operand : yield.getOperands()) {
      if (outputIdx >= getOutputs().size())
        return emitOpError("region #")
               << idx << " yields more operands than declared outputs ("
               << getOutputs().size() << ")";
      if (operand.getType() != getOutputs()[outputIdx].getType())
        return emitOpError("region #")
               << idx << " yield operand type " << operand.getType()
               << " does not match output #" << outputIdx << " type "
               << getOutputs()[outputIdx].getType();
      ++outputIdx;
    }
  }

  if (outputIdx != getOutputs().size())
    return emitOpError("total `mgp.device_graph.collective.yield` operand "
                       "count across regions is ")
           << outputIdx << ", expected " << getOutputs().size()
           << " (one per op result)";

  return success();
}

ParseResult DeviceGraphCollectiveOp::parse(OpAsmParser &parser,
                                           OperationState &result) {
  // Parse the shared `!mgp.device_graph.builder` block argument. The same
  // prototype is threaded into every per-device region. The printed form
  // omits the builder type, so when it is missing we infer it.
  if (parser.parseKeyword("builder"))
    return failure();

  if (parser.parseLParen())
    return failure();

  OpAsmParser::Argument builderArg;
  if (parser.parseArgument(builderArg, /*allowType=*/false))
    return failure();

  // The builder type is optional in the printed form and defaults to
  // `!mgp.device_graph.builder`.
  if (succeeded(parser.parseOptionalColon())) {
    if (parser.parseType(builderArg.type))
      return failure();
  } else {
    builderArg.type = DeviceGraphBuilderType::get(parser.getContext());
  }

  if (parser.parseRParen())
    return failure();

  // Parse the shared stable-address block arguments. The same prototype is
  // threaded into every per-device region.
  if (parser.parseKeyword("stable"))
    return failure();

  SmallVector<OpAsmParser::Argument> stableArgs;
  if (parser.parseArgumentList(stableArgs, OpAsmParser::Delimiter::Paren,
                               /*allowType=*/true))
    return failure();

  // The first region's entry block takes the builder, then the stable slots.
  SmallVector<OpAsmParser::Argument> regionArgs;
  regionArgs.push_back(builderArg);
  llvm::append_range(regionArgs, stableArgs);

  // Parse each device region: a `DeviceRefAttr` label followed by the region
  // body.  The label order defines the `$devices` attribute.
  SmallVector<DeviceRefAttr> deviceRefs;

  auto parseDeviceRegionPair = [&]() -> ParseResult {
    DeviceRefAttr deviceRef;
    if (parser.parseCustomAttributeWithFallback(deviceRef))
      return failure();
    deviceRefs.push_back(deviceRef);

    Region *region = result.addRegion();
    if (parser.parseRegion(*region, regionArgs,
                           /*enableNameShadowing=*/false))
      return failure();

    return success();
  };

  if (parser.parseCommaSeparatedList(OpAsmParser::Delimiter::None,
                                     parseDeviceRegionPair))
    return failure();

  // Parse the optional attribute dictionary (e.g. constant_resources).
  // `devices` is represented by the per-region labels, so drop it if the
  // textual attr-dict happens to contain it.
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  result.attributes.erase(getDevicesAttrName(result.name));

  auto devices = DeviceRefsAttr::get(parser.getContext(), deviceRefs);
  result.addAttribute(getDevicesAttrName(result.name), devices);

  // Parse optional trailing result types.
  SmallVector<Type> resultTypes;
  if (parser.parseOptionalColonTypeList(resultTypes))
    return failure();
  result.types.append(resultTypes);

  return success();
}

void DeviceGraphCollectiveOp::print(OpAsmPrinter &p) {
  // Print the shared entry-block arguments once, using the first region as the
  // canonical prototype. The first argument is the `!mgp.device_graph.builder`
  // printed in a `builder(...)` clause; the rest are stable-address slots
  // printed in a `stable(...)` clause. The printer assigns unique names and
  // reuses them when the region bodies are printed without their entry-block
  // arguments.
  p << " builder(";
  if (!getGraphs().empty()) {
    // The builder block argument is always a `!mgp.device_graph.builder`,
    // so the type is omitted from the printed form.
    p.printRegionArgument(getRegionBlock(0).getArgument(0),
                          /*argAttrs=*/{}, /*omitType=*/true);
  }
  p << ") stable(";
  if (!getGraphs().empty()) {
    Block &firstBlock = getRegionBlock(0);
    auto stableArgs = firstBlock.getArguments().drop_front(1);
    llvm::interleaveComma(stableArgs, p, [&](BlockArgument arg) {
      p.printRegionArgument(arg, /*argAttrs=*/{},
                            /*omitType=*/false);
    });
  }
  p << ')';

  if (!getDevices().empty())
    p << " ";

  llvm::interleaveComma(llvm::enumerate(getDevices()), p, [&](auto idxAndAttr) {
    auto [idx, attr] = idxAndAttr;
    p.printStrippedAttrOrType(attr);
    p << " ";
    p.printRegion(getGraphs()[idx], /*printEntryBlockArgs=*/false);
  });

  // The `devices` attribute is represented by the per-region labels above. An
  // empty `constantResources` is omitted to keep the printed form compact and
  // consistent with `mgp.device_graph.staging`.
  SmallVector<StringRef> elide = {getDevicesAttrName()};
  if (getConstantResources().empty())
    elide.push_back(getConstantResourcesAttrName());
  p.printOptionalAttrDict(getOperation()->getAttrs(), elide);

  // Print trailing result types when present.
  if (!getOutputs().empty()) {
    p << " : ";
    llvm::interleaveComma(getOutputs(), p,
                          [&](Value v) { p.printType(v.getType()); });
  }
}

void DeviceGraphCollectiveOp::getAsmBlockArgumentNames(
    Region &region, llvm::function_ref<void(Value, StringRef)> setNameFn) {
  // Set argument names for the entry block arguments: the builder is `b`, and
  // the stable-address slots are `s0`, `s1`, ...
  for (auto [idx, arg] : llvm::enumerate(region.getArguments())) {
    if (idx == 0)
      setNameFn(arg, "b");
    else
      setNameFn(arg, "s" + std::to_string(idx - 1));
  }
}

void DeviceGraphCollectiveOp::build(
    OpBuilder &builder, OperationState &state,
    ArrayRef<M::DeviceRefAttr> devices, ArrayRef<StableAddressType> stableTypes,
    function_ref<void(OpBuilder &, M::DeviceRefAttr, BlockArgument,
                      ArrayRef<BlockArgument>)>
        bodyBuilder) {
  OpBuilder::InsertionGuard guard(builder);

  MLIRContext *ctx = builder.getContext();

  SmallVector<Location> argLocs(stableTypes.size(), state.location);
  auto argTypes = llvm::to_vector_of<Type>(stableTypes);

  // Build one region per device.  Each region gets the shared builder block
  // argument followed by one block argument per stable address slot.
  SmallVector<Type> resultTypes;
  for (auto dev : devices) {
    Region *region = state.addRegion();
    Block *block = builder.createBlock(region);

    auto builderArg =
        block->addArgument(DeviceGraphBuilderType::get(ctx), state.location);
    auto args = block->addArguments(argTypes, argLocs);

    SmallVector<BlockArgument> stableArgs = llvm::to_vector(args);

    builder.setInsertionPointToEnd(block);
    bodyBuilder(builder, dev, builderArg, stableArgs);
    ensureTerminator(*region, builder, state.location);

    // Infer output types from the terminator the callback produced.
    llvm::append_range(resultTypes, block->getTerminator()->getOperandTypes());
  }

  // Call into the tablegen builder with `graphsCount=0`. We've already added
  // the child regions to `state`, so we do not want to add any additional
  // regions as the region count will not match the number of devices.
  DeviceGraphCollectiveOp::build(builder, state, resultTypes, devices,
                                 /*constantResources=*/ArrayAttr(),
                                 /*graphsCount=*/0);
}

//===----------------------------------------------------------------------===//
// DeviceGraphCollectiveYieldOp
//===----------------------------------------------------------------------===//

void DeviceGraphCollectiveYieldOp::build(OpBuilder &, OperationState &) {
  // No-operand stub for the parent's `SingleBlockImplicitTerminator`; operands
  // are populated after insertion.
}

//===----------------------------------------------------------------------===//
// DeviceGraphCollectiveCreateOp
//===----------------------------------------------------------------------===//

ParseResult DeviceGraphCollectiveCreateOp::parse(OpAsmParser &parser,
                                                 OperationState &result) {
  MLIRContext *ctx = parser.getContext();
  Type contextType = MGP::ContextType::get(ctx);

  // The shared `!mgp.context` operand.  It is also threaded into each region as
  // the second block argument.
  OpAsmParser::UnresolvedOperand theContext;
  if (parser.parseOperand(theContext))
    return failure();

  // Optional cache(%cache : !mgp.device_graph.cache).
  bool hasCache = false;
  OpAsmParser::UnresolvedOperand cacheVal;
  Type cacheType;
  if (succeeded(parser.parseOptionalKeyword("cache"))) {
    hasCache = true;
    if (parser.parseLParen() || parser.parseOperand(cacheVal) ||
        parser.parseColon() || parser.parseType(cacheType) ||
        parser.parseRParen())
      return failure();
  }

  SmallVector<OpAsmParser::UnresolvedOperand> deviceContextVals;
  SmallVector<Type> deviceContextTypes;
  SmallVector<OpAsmParser::UnresolvedOperand> inputVals;
  SmallVector<Type> inputTypes;

  // Each per-device graph is introduced by its device context, builder block
  // arg, and input block args.  The body itself is printed without its
  // entry-block arguments because they are declared in the prefix clauses.
  auto parseGraph = [&]() -> ParseResult {
    // graph(<deviceContext> : <type>)
    if (parser.parseKeyword("graph") || parser.parseLParen())
      return failure();

    OpAsmParser::UnresolvedOperand deviceContext;
    Type deviceContextType;
    if (parser.parseOperand(deviceContext) || parser.parseColon() ||
        parser.parseType(deviceContextType) || parser.parseRParen())
      return failure();
    deviceContextVals.push_back(deviceContext);
    deviceContextTypes.push_back(deviceContextType);

    // builder(<name>)
    if (parser.parseKeyword("builder") || parser.parseLParen())
      return failure();
    OpAsmParser::Argument builderArg;
    if (parser.parseArgument(builderArg, /*allowType=*/false))
      return failure();
    builderArg.type = DeviceGraphBuilderType::get(ctx);
    if (parser.parseRParen())
      return failure();

    // inputs(<name> : <type>, ...)
    if (parser.parseKeyword("inputs") || parser.parseLParen())
      return failure();
    SmallVector<OpAsmParser::Argument> inputArgs;
    if (succeeded(parser.parseOptionalRParen())) {
      // empty input list
    } else {
      do {
        OpAsmParser::UnresolvedOperand input;
        Type inputType;
        if (parser.parseOperand(input) || parser.parseColon() ||
            parser.parseType(inputType))
          return failure();
        inputVals.push_back(input);
        inputTypes.push_back(inputType);

        OpAsmParser::Argument inputArg;
        inputArg.ssaName = input;
        inputArg.type = inputType;
        inputArgs.push_back(inputArg);
      } while (succeeded(parser.parseOptionalComma()));
      if (parser.parseRParen())
        return failure();
    }

    // The shared context becomes the second block argument; use the same name
    // as the op-level context operand so the body can refer to it naturally.
    OpAsmParser::Argument contextArg;
    contextArg.ssaName = theContext;
    contextArg.type = contextType;

    SmallVector<OpAsmParser::Argument> regionArgs;
    regionArgs.push_back(builderArg);
    regionArgs.push_back(contextArg);
    llvm::append_range(regionArgs, inputArgs);

    Region *region = result.addRegion();
    if (parser.parseRegion(*region, regionArgs,
                           /*enableNameShadowing=*/true))
      return failure();

    return success();
  };

  if (parser.parseCommaSeparatedList(parseGraph))
    return failure();

  // Resolve operands in the order expected by the ODS-generated accessors.
  if (parser.resolveOperand(theContext, contextType, result.operands) ||
      parser.resolveOperands(deviceContextVals, deviceContextTypes, {},
                             result.operands) ||
      (hasCache &&
       parser.resolveOperand(cacheVal, cacheType, result.operands)) ||
      parser.resolveOperands(inputVals, inputTypes, {}, result.operands))
    return failure();

  // The `operandSegmentSizes` attribute is required by
  // `AttrSizedOperandSegments`.
  result.addAttribute(
      getOperandSegmentSizesAttrName(result.name),
      parser.getBuilder().getDenseI32ArrayAttr(
          {1, static_cast<int32_t>(deviceContextVals.size()), hasCache ? 1 : 0,
           static_cast<int32_t>(inputVals.size())}));

  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  SmallVector<Type> resultTypes;
  if (parser.parseOptionalColonTypeList(resultTypes))
    return failure();
  result.types.append(resultTypes);

  return success();
}

void DeviceGraphCollectiveCreateOp::print(OpAsmPrinter &p) {
  p << ' ' << getTheContext();

  if (Value cache = getCache()) {
    p << " cache(" << cache << " : " << cache.getType() << ")";
  }

  size_t inputOffset = 0;
  for (auto [idx, region] : llvm::enumerate(getGraphs())) {
    if (idx != 0)
      p << ",";

    Block &body = region.front();
    Value deviceContext = getDeviceContexts()[idx];
    BlockArgument builderArg = body.getArgument(0);
    size_t numRegionInputs = body.getNumArguments() - 2;
    auto regionInputs =
        getInputs().drop_front(inputOffset).take_front(numRegionInputs);

    p << " graph(" << deviceContext << " : " << deviceContext.getType()
      << ") builder(";
    p.printRegionArgument(builderArg, /*argAttrs=*/{}, /*omitType=*/true);

    p << ") inputs(";
    llvm::interleaveComma(regionInputs, p, [&](Value input) {
      p << input << " : " << input.getType();
    });
    p << ") ";

    SmallVector<Value> shadowNames;
    shadowNames.reserve(1 + 1 + numRegionInputs);
    shadowNames.push_back(builderArg);
    shadowNames.push_back(getTheContext());
    llvm::append_range(shadowNames, regionInputs);
    p.shadowRegionArgs(region, shadowNames);

    p.printRegion(region, /*printEntryBlockArgs=*/false);
    inputOffset += numRegionInputs;
  }

  SmallVector<StringRef> elide = {getOperandSegmentSizesAttrName()};
  p.printOptionalAttrDict(getOperation()->getAttrs(), elide);

  if (!getOutGraphs().empty()) {
    p << " : ";
    llvm::interleaveComma(getOutGraphs(), p,
                          [&](Value v) { p.printType(v.getType()); });
  }
}

void DeviceGraphCollectiveCreateOp::getAsmBlockArgumentNames(
    Region &region, llvm::function_ref<void(Value, StringRef)> setNameFn) {
  unsigned idx = region.getRegionNumber();
  setNameFn(region.front().getArgument(0), "b" + Twine(idx).str());
}

LogicalResult DeviceGraphCollectiveCreateOp::verify() {
  if (getGraphs().size() != getDeviceContexts().size())
    return emitOpError("`deviceContexts` has ")
           << getDeviceContexts().size() << " operands, but the op has "
           << getGraphs().size() << " regions";

  if (getOutGraphs().size() != getGraphs().size())
    return emitOpError() << "number of result graphs (" << getOutGraphs().size()
                         << ") does not match the number of body regions ("
                         << getGraphs().size() << ")";

  if (getGraphs().empty())
    return emitOpError("must have at least one device graph region");

  Type theContextType = getTheContext().getType();
  SmallVector<Type> inputTypes;
  llvm::append_range(inputTypes, getInputs().getTypes());
  size_t inputIdx = 0;

  for (auto [regionIdx, region] : llvm::enumerate(getGraphs())) {
    Block &block = region.front();

    if (block.empty() || !llvm::isa<DeviceGraphYieldOp>(block.back()))
      return emitOpError("region #")
             << regionIdx << " must terminate with `mgp.device_graph.yield`";

    if (block.getNumArguments() < 2)
      return emitOpError("region #")
             << regionIdx << " has " << block.getNumArguments()
             << " block argument(s), expected at least 2 (the builder and the "
                "context)";

    if (!llvm::isa<DeviceGraphBuilderType>(block.getArgument(0).getType()))
      return emitOpError("region #")
             << regionIdx << " block argument #0 has type "
             << block.getArgument(0).getType()
             << "; expected `!mgp.device_graph.builder`";

    if (block.getArgument(1).getType() != theContextType)
      return emitOpError("region #")
             << regionIdx
             << " block argument #1 must be the `theContext` operand type "
             << theContextType << ", but is " << block.getArgument(1).getType();

    size_t numRegionInputs = block.getNumArguments() - 2;
    if (inputIdx + numRegionInputs > getInputs().size())
      return emitOpError("region #")
             << regionIdx << " has " << numRegionInputs
             << " input block argument(s), but only "
             << (getInputs().size() - inputIdx) << " input operand(s) remain";

    for (size_t i = 0; i < numRegionInputs; ++i) {
      Type argType = block.getArgument(2 + i).getType();
      Type inputType = getInputs()[inputIdx + i].getType();
      if (argType != inputType)
        return emitOpError("region #")
               << regionIdx << " block argument #" << (2 + i) << " has type "
               << argType << ", expected " << inputType;
    }

    auto yieldOp = cast<DeviceGraphYieldOp>(block.back());
    auto graphType = cast<DeviceGraphType>(getOutGraphs()[regionIdx].getType());
    FunctionType sig = graphType.getSignature();

    ArrayRef<Type> regionInputTypes(
        ArrayRef<Type>(inputTypes).slice(inputIdx, numRegionInputs));
    if (!llvm::equal(sig.getInputs(), regionInputTypes))
      return emitOpError("result graph #")
             << regionIdx
             << " signature inputs do not match this region's input operands";

    if (!llvm::equal(sig.getResults(), yieldOp.getOutputs().getTypes()))
      return emitOpError("result graph #")
             << regionIdx
             << " signature results do not match the region's yielded value "
                "types";

    inputIdx += numRegionInputs;
  }

  if (inputIdx != getInputs().size())
    return emitOpError("the op has ")
           << getInputs().size()
           << " input operand(s), but the regions consume only " << inputIdx;

  return verifyInputResidency(*this, getInputs());
}

void DeviceGraphCollectiveCreateOp::build(
    OpBuilder &builder, OperationState &result, Value theContext,
    llvm::ArrayRef<Value> deviceContexts, Value cache,
    llvm::ArrayRef<Value> inputs, llvm::ArrayRef<unsigned> inputCounts,
    llvm::function_ref<void(OpBuilder &, Location, unsigned, Value, Value,
                            ValueRange)>
        bodyBuilder) {
  OpBuilder::InsertionGuard guard(builder);

  unsigned numRegions = inputCounts.size();
  unsigned inputTotal = 0;
  for (unsigned count : inputCounts)
    inputTotal += count;
  assert(inputTotal == inputs.size() &&
         "inputCounts must sum to inputs.size()");

  MLIRContext *ctx = builder.getContext();
  SmallVector<Type> graphTypes;
  graphTypes.reserve(numRegions);

  size_t inputOffset = 0;

  for (unsigned i : llvm::seq(numRegions)) {
    Region *region = result.addRegion();
    Block *block = builder.createBlock(region);

    Value graphBuilder =
        block->addArgument(DeviceGraphBuilderType::get(ctx), result.location);
    Value contextArg =
        block->addArgument(theContext.getType(), result.location);

    unsigned count = inputCounts[i];
    ValueRange argSlice =
        llvm::ArrayRef<Value>(inputs).slice(inputOffset, count);
    SmallVector<Location> locs(count, result.location);
    auto inputArgs = block->addArguments(argSlice.getTypes(), locs);

    if (bodyBuilder)
      bodyBuilder(builder, result.location, i, graphBuilder, contextArg,
                  ValueRange(inputArgs));

    if (!block->mightHaveTerminator())
      ensureTerminator(*region, builder, result.location);

    auto yieldOp = cast<DeviceGraphYieldOp>(block->getTerminator());
    FunctionType signature = FunctionType::get(ctx, argSlice.getTypes(),
                                               yieldOp.getOutputs().getTypes());
    graphTypes.push_back(DeviceGraphType::get(ctx, signature));

    inputOffset += count;
  }

  build(builder, result, graphTypes, theContext, deviceContexts, cache, inputs,
        /*graphsCount=*/0);
}

LogicalResult DeviceGraphCollectiveCreateOp::emitMojo(MOGG::EmitMojoState &) {
  return (*this)->emitError(
      "`mgp.device_graph.collective.create` Mojo emission is not yet "
      "implemented");
}

bool DeviceGraphCollectiveCreateOp::isThrows() { return true; }

//===----------------------------------------------------------------------===//
// DeviceGraphCollectiveExecuteOp
//===----------------------------------------------------------------------===//

LogicalResult DeviceGraphCollectiveExecuteOp::verify() {
  size_t expectedInputCount = 0;
  size_t expectedOutputCount = 0;

  for (auto graph : getGraphs()) {
    auto sig = cast<DeviceGraphType>(graph.getType()).getSignature();
    expectedInputCount += sig.getInputs().size();
    expectedOutputCount += sig.getResults().size();
  }

  if (getInputs().size() != expectedInputCount)
    return emitOpError("`inputs` has ")
           << getInputs().size()
           << " operands, but the graph signatures consume "
           << expectedInputCount;

  if (getOutputs().size() != expectedOutputCount)
    return emitOpError("`outputs` has ")
           << getOutputs().size()
           << " results, but the graph signatures produce "
           << expectedOutputCount;

  SmallVector<Type> inputTypes;
  llvm::append_range(inputTypes, getInputs().getTypes());
  SmallVector<Type> outputTypes;
  llvm::append_range(outputTypes, getOutputs().getTypes());

  size_t inputIdx = 0;
  size_t outputIdx = 0;
  for (auto [graphIdx, graph] : llvm::enumerate(getGraphs())) {
    auto sig = cast<DeviceGraphType>(graph.getType()).getSignature();

    if (!llvm::equal(
            sig.getInputs(),
            ArrayRef<Type>(inputTypes).slice(inputIdx, sig.getInputs().size())))
      return emitOpError("graph #")
             << graphIdx
             << " signature inputs do not match the corresponding `inputs` "
                "operands";
    inputIdx += sig.getInputs().size();

    if (!llvm::equal(sig.getResults(),
                     ArrayRef<Type>(outputTypes)
                         .slice(outputIdx, sig.getResults().size())))
      return emitOpError("graph #")
             << graphIdx
             << " signature results do not match the corresponding `outputs` "
                "results";
    outputIdx += sig.getResults().size();
  }

  return verifyInputResidency(*this, getInputs());
}
