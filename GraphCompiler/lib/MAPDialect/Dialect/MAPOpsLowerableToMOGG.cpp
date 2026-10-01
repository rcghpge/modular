//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// `LowerableToMOGGInterface` implementations for the `map` ops MAPToMOGG
// lowers. See `GraphCompiler/lib/Conversion/MAPToMOGG/MAPToMOGG.cpp` for the
// pass driver.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/Conversion/ParameterInference.h"
#include "GraphCompiler/Interfaces/MAPToMOGGState.h"
#include "GraphCompiler/MAPDialect/Dialect/MAPOps.h"
#include "GraphCompiler/MODialect/MODevices.h"
#include "GraphCompiler/MODialect/MOTypes.h"
#include "GraphCompiler/MOGGDialect/Analysis/KernelDeclAdaptor.h"
#include "GraphCompiler/MOGGDialect/Analysis/TraitContractDerivation.h"
#include "GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GraphCompiler/MOGGDialect/Support/FusionUtils.h"
#include "GraphCompiler/MOGGDialect/Support/MojoIdentifiers.h"
#include "GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "Mojo/KGENDialect/KGENAttrs.h"
#include "Support/AssertStream.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "llvm/ADT/SetVector.h"

using namespace mlir;

namespace M::MAP {

namespace {

// A dimension is dynamic unless it is a concrete `KGEN::SIMDAttr`.
bool isDynamicDim(Attribute dim) { return !isa<KGEN::SIMDAttr>(dim); }

// Registers a walk that collects every dynamic dim of any `MOSH::ShapeAttr`
// nested in what it is handed -- a tensor type's shape, a view op's spec, a
// `map.iter.foreach.body`'s `$shape` -- deduping into `state.dimArgs`. One
// generic `ShapeAttr` walk finds them wherever they nest, so there is no need
// to special-case tensor types versus attributes.
mlir::AttrTypeWalker dynamicDimWalker(MAPToMOGGState &state) {
  mlir::AttrTypeWalker walker;
  walker.addWalk([&state](MOSH::ShapeAttr shape) {
    for (Attribute dim : shape.getValues())
      if (isDynamicDim(dim))
        state.dimArgs.insert({dim, BlockArgument()});
  });
  return walker;
}

// The dynamic dims of `types`, deduped in first-seen order.
void collectDynamicDims(TypeRange types, MAPToMOGGState &state) {
  mlir::AttrTypeWalker walker = dynamicDimWalker(state);
  for (Type ty : types)
    walker.walk(ty);
}

// Every dynamic dim `root` and its nested ops reference -- from any
// `MOSH::ShapeAttr` in an op's attributes (a view op's spec, a
// `map.iter.foreach.body`'s `$shape`) or its operand/result types (a tensor
// type's shape). Recurses over the whole op tree, so one call on a kernel's
// top op collects every dim it references, wherever it lives.
void collectDynamicDims(Operation *root, MAPToMOGGState &state) {
  mlir::AttrTypeWalker walker = dynamicDimWalker(state);
  root->walk([&](Operation *op) {
    // `AttrTypeWalker` recurses on its own: one walk of the attribute
    // dictionary reaches every nested `ShapeAttr` (view specs, a foreach
    // body's `$shape`). Tensor shapes live in the operand/result TYPES rather
    // than the attribute dictionary, so those are walked too -- a plain kernel
    // with no view specs has its dims only there.
    walker.walk(op->getAttrDictionary());
    for (Type ty : op->getOperandTypes())
      walker.walk(ty);
    for (Type ty : op->getResultTypes())
      walker.walk(ty);
  });
}

// Materializes a `!mogg.shape` for `shapeAttr` via `mogg.shape.create`.
// Canonicalizes every dynamic dim in the shape ATTR to one uniform `dyn`
// placeholder so shapes carrying different symbols (e.g. `[M, K]` vs `[N,
// K]`) dedup to `[dyn, dyn]` / `[dyn, K]`. The `!mogg.dim_` OPERANDS stay the
// real per-dim handles (they carry runtime values); only the attr is folded.
Value buildShape(Location loc, MOSH::ShapeAttr shapeAttr,
                 MAPToMOGGState &state) {
  MLIRContext *ctx = state.builder.getContext();
  TypedAttr canonicalDyn = KGEN::ParamDeclRefAttr::get(
      StringAttr::get(ctx, "dyn"), MOSH::ShapeAttr::getDimType(ctx));
  SmallVector<Value> dims;
  SmallVector<TypedAttr> canonicalDims;
  for (TypedAttr dim : shapeAttr.getValues()) {
    if (isDynamicDim(dim)) {
      dims.push_back(state.dimArgs.lookup(dim));
      canonicalDims.push_back(canonicalDyn);
    } else {
      canonicalDims.push_back(dim);
    }
  }
  auto canonicalShapeAttr =
      MOSH::ShapeAttr::get(canonicalDims, shapeAttr.getType());
  return MOGG::ShapeCreateOp::create(
      state.builder, loc, MOGG::ShapeType::get(ctx), dims, canonicalShapeAttr);
}

// Reconstructs an `!mogg.tensor_` for `tensorType` (a tensor, or a buffer --
// e.g. one `FuseMutableLoads` fused a load's buffer into in place of the
// tensor operand it used to read -- read via `MO::castFromTensorLike`) from
// the `!mogg.pointer_` block argument `ptr`, via `mogg.shape.create` +
// `mogg.tensor.create`. `io`/`isMut` distinguish immutable inputs from
// mutable outputs.
Value buildTensor(Location loc, Value ptr, Type tensorType, MO::IOKind io,
                  bool isMut, MAPToMOGGState &state) {
  MO::TensorType tensorTy = MO::castFromTensorLike(tensorType);
  MLIRContext *ctx = state.builder.getContext();
  MOSH::ShapeAttr shapeAttr = tensorTy.getShape();
  Value shape = buildShape(loc, shapeAttr, state);
  // A `!mo.buffer` operand (mutable memory the kernel writes through, e.g. a
  // collective's signal buffers) reconstructs as a `!mogg.buffer_` handle
  // (`Buffer` kind); a `!mo.tensor` -- even a mutable output -- reconstructs as
  // a `!mogg.tensor_` (`Tensor` kind). Buffer-ness (the kind) is thus a
  // property of the operand type, while `$mut` is the store-side mutability
  // (true for buffers and for output tensors alike); `mogg.tensor.create`'s
  // verifier only requires a `!mogg.buffer_` to be mutable.
  bool isBuffer = isa<MO::BufferType>(tensorType);
  Type resultType = isBuffer ? Type(MOGG::BufferType::get(ctx))
                             : Type(MOGG::TensorType::get(ctx));
  return MOGG::TensorCreateOp::create(
      state.builder, loc, resultType, ptr, shape, shapeAttr,
      state.builder.getStringAttr(state.device.getLabel()),
      TypeAttr::get(tensorTy.getElementType()),
      state.builder.getBoolAttr(isMut), MO::IOKindAttr::get(ctx, io));
}

} // namespace

// mo.invoke_shape_func -> mogg.kernel wrapping a mogg.call.shape. The one
// kernel here that is not destination-passing: the shape leaves by value, and
// the dims it defines ride in `outputParamDecls`.
LogicalResult lowerInvokeShapeFuncToMOGG(MO::InvokeShapeFuncOp shapeFuncOp,
                                         MAPToMOGGState &state) {
  Operation *op = shapeFuncOp.getOperation();
  MLIRContext *ctx = op->getContext();
  Location loc = op->getLoc();
  OpBuilder &builder = state.builder;
  builder.setInsertionPoint(op);

  MOGG::ShapeFunctionDeclOp shapeDecl =
      state.registry.lookupShapeFunction(shapeFuncOp.getShapeFuncName());
  if (!shapeDecl)
    return op->emitOpError("MAPToMOGG: '")
           << shapeFuncOp.getShapeFuncName()
           << "' requires a registered shape function; add "
              "'@extensibility.register_shape_function(\""
           << shapeFuncOp.getShapeFuncName() << "\")' to the kernel";

  ValueRange inputs = shapeFuncOp.getInputs();

  // The op's declared output rank, never derived from an argument.
  const uint64_t rank = shapeFuncOp.getOutputParamDecls().size();
  Type shapeResultType = MOGG::ShapeFuncResultType::get(ctx, rank);

  // Host-only.
  state.device = MO::getDeviceOrDefaultAny(op);

  auto kernel = MOGG::KernelOp::create(
      builder, loc, /*resultTypes=*/TypeRange{shapeResultType}, inputs,
      /*chain=*/Value(),
      MOGG::FusionInfoAttr::get(ctx, MOGG::createFusionLeaf(op)),
      shapeFuncOp.getOutputParamDeclsAttr());
  kernel.setDeviceRef(state.device);

  // Inputs only: the sole result is the shape.
  collectDynamicDims(inputs.getTypes(), state);

  // A pointer per input, then the dims, then the context.
  Block *body = builder.createBlock(&kernel.getBody());
  SmallVector<BlockArgument> inputPtrs;
  for (size_t i = 0, e = inputs.size(); i < e; ++i)
    inputPtrs.push_back(body->addArgument(MOGG::PointerType::get(ctx), loc));
  for (auto &[dim, blockArg] : state.dimArgs)
    blockArg = body->addArgument(
        MOGG::DimType::get(ctx, dyn_cast<KGEN::ParamDeclRefAttr>(dim)), loc);
  // A shape-function kernel runs on host and takes a single context.
  state.contextArgs.assign(
      {body->addArgument(MOGG::ContextType::get(ctx), loc)});

  builder.setInsertionPointToEnd(body);
  SmallVector<Value> callOperands;
  for (auto [i, in] : llvm::enumerate(inputs))
    callOperands.push_back(buildTensor(loc, inputPtrs[i], in.getType(),
                                       MO::IOKind::Input, /*isMut=*/false,
                                       state));

  // `kgenParams` are parameters of the shape function too, so forwarding the
  // rank alone leaves them uninferable.
  SmallVector<NamedAttribute> shapeParams{
      builder.getNamedAttr(MOGG::kMoggOutputRankParameterName,
                           builder.getIndexAttr(static_cast<int64_t>(rank)))};
  if (DictionaryAttr kgenParams = shapeFuncOp.getKgenParamsAttr())
    for (NamedAttribute param : kgenParams)
      shapeParams.push_back(builder.getNamedAttr(
          param.getName(), normalizeKernelParameter(param.getValue())));

  bool isThrows = shapeDecl.getFunction().getSignature().isThrows();
  auto call = MOGG::CallShapeOp::create(
      builder, loc, shapeResultType,
      FlatSymbolRefAttr::get(shapeDecl.getSymNameAttr()),
      isThrows ? builder.getUnitAttr() : nullptr, callOperands,
      builder.getDictionaryAttr(shapeParams));

  // The kernel's only output.
  MOGG::OutputOp::create(builder, loc, call.getResult());

  op->erase();
  return success();
}

// map.constant -> mogg.buffer.constant + mogg.tensor.create. The constant has
// no operand to arrive on, so its bytes become a constant buffer and the tensor
// is rebuilt from that pointer like any input. `MOGGToMGP` turns the buffer
// into an `mgp.buffer.constant` at the call site, where the pipeline's minimum
// CPU alignment is known.
LogicalResult ConstantOp::lowerToMOGG(MAPToMOGGState &state) {
  MLIRContext *ctx = state.builder.getContext();
  Value ptr = MOGG::BufferConstantOp::create(
      state.builder, getLoc(), MOGG::PointerType::get(ctx), getValueAttr());
  state.mapping.map(getResult(),
                    buildTensor(getLoc(), ptr, getType(), MO::IOKind::Input,
                                /*isMut=*/false, state));
  return success();
}

// map.constant.scalar -> mogg.constant.scalar. Unlike `map.constant` above,
// there is no buffer to reconstruct: the value is a literal at both ends.
LogicalResult ConstantScalarOp::lowerToMOGG(MAPToMOGGState &state) {
  auto constant = MOGG::ConstantScalarOp::create(
      state.builder, getLoc(), getResult().getType(), getValue());
  state.mapping.map(getResult(), constant.getResult());
  return success();
}

// map.load -> mogg.tensor.load.
LogicalResult LoadOp::lowerToMOGG(MAPToMOGGState &state) {
  Value result = MOGG::TensorLoadOp::create(
      state.builder, getLoc(), getResult().getType(),
      state.mapping.lookup(getTensor()), state.mapping.lookup(getIndex()));
  state.mapping.map(getResult(), result);
  return success();
}

// map.store -> mogg.tensor.store.
LogicalResult StoreOp::lowerToMOGG(MAPToMOGGState &state) {
  MOGG::TensorStoreOp::create(
      state.builder, getLoc(), state.mapping.lookup(getTensor()),
      state.mapping.lookup(getIndex()), state.mapping.lookup(getValue()));
  return success();
}

// map.call.elementwise -> mogg.call.elementwise. `$moOpName` isn't carried
// on the MAP op, so it's read from the kernel decl `getCallee()` names.
LogicalResult CallElementwiseOp::lowerToMOGG(MAPToMOGGState &state) {
  StringRef kernelSymName = getCallee();
  // `verifySymbolUses` already guarantees `kernelSymName` resolves to a
  // kernel decl with an `elementwise` function.
  MOGG::KernelDeclOp decl = state.registry.lookupKernelBySymName(kernelSymName);
  MOGG::KernelDeclAdaptor adaptor(decl);
  SmallVector<Value> inputs;
  for (Value arg : getArgs())
    inputs.push_back(state.mapping.lookup(arg));
  // `$index` is set only if this kernel's `elementwise` function declares a
  // trailing index-list parameter for it.
  Value index;
  const MOGG::KernelDeclFunctionAdaptor &fn = *adaptor.elementwiseFunction;
  if (!fn.inputArguments.empty() &&
      fn.inputArguments.back().isOpaqueWithTypeName(
          MOGG::kMojoInternalDpsIndexListTypeQualifiedName))
    index = state.mapping.lookup(getIndex());
  auto moggCall = MOGG::CallElementwiseOp::create(
      state.builder, getLoc(), getResult().getType(), kernelSymName,
      adaptor.registeredOpName.strref(), inputs, index, getParamsAttr());
  state.mapping.map(getResult(), moggCall.getOutputs()[0]);
  return success();
}

// Which fusion-lambda kind the kernel consuming `epilogueOp` wants for the
// output it feeds. Load-bearing detail: this is resolved here, at MAP level,
// because the epilogue -> `map.call.opaque` edge is still explicit in the IR
// and the epilogue lowers before the call does. Emit therefore never has to
// walk anywhere to recover it.
static ErrorOr<MOGG::KernelOutputFusionLambdaType>
classifyEpilogueLambdaKind(IterEpilogueOp epilogueOp, MAPToMOGGState &state) {
  using Kind = MOGG::KernelOutputFusionLambdaType;

  // The epilogue result feeds exactly one `map.call.opaque` output operand in
  // every shape `EpilogueFuser` builds. Anything else (an unfused result, a
  // second user) gets a store lambda, matching what the MOGG-ABI classifier
  // does for an unrecognized user.
  Value result = epilogueOp.getResult();
  if (!result.hasOneUse())
    return Kind::Output;
  auto callOp = dyn_cast<CallOpaqueOp>(*result.getUsers().begin());
  if (!callOp)
    return Kind::Output;
  std::optional<unsigned> outputIdx;
  for (auto [idx, out] : llvm::enumerate(callOp.getOutputs()))
    if (out == result)
      outputIdx = idx;
  if (!outputIdx)
    return Kind::Output;

  MOGG::KernelDeclAdaptor adaptor(
      state.registry.lookupKernelBySymName(callOp.getCallee()));
  if (!adaptor.executeFunction)
    return Kind::Output;
  const auto &outputArgs = adaptor.executeFunction->outputArguments;
  if (*outputIdx >= outputArgs.size())
    return Kind::Output;
  const auto *tensorArg = std::get_if<MOGG::KernelDeclTensorAdaptor>(
      &outputArgs[*outputIdx].underlyingType);
  if (!tensorArg)
    return Kind::Output;

  // `EpilogueFuser` only ever builds the SIMD body shape, so a tile-based
  // fused-output argument has nothing to lower into.
  if (tensorArg->ioSpec == MOGG::IOSpec::FusedComputeOutputTileTensor ||
      tensorArg->ioSpec == MOGG::IOSpec::FusedOutputTileTensor)
    return Error("MAP-driven epilogue fusion into a tile fused-output kernel "
                 "argument is not yet supported");

  if (tensorArg->ioSpec != MOGG::IOSpec::FusedComputeOutputTensor)
    return Kind::Output;

  // A store-index transform (a reshape fused into the epilogue) stores each
  // element at a DIFFERENT position than the kernel handed it. Only a store
  // lambda, which owns its store, can express that -- a compute lambda returns
  // a value the kernel then stores at ITS OWN index, silently dropping the
  // transform. Detected structurally: the store's index is not the epilogue's
  // own (untransformed) index block arg.
  if (auto storeOp = dyn_cast<StoreOp>(epilogueOp.getBody().front().back()))
    if (storeOp.getIndex() != epilogueOp.getBody().getArgument(1))
      return Kind::OutputLegacyForCompute;

  // The kernel can take a compute lambda. Two things still force the classic
  // store lambda: the opt-out, and a dtype-changing epilogue -- `compute()`
  // returns the dtype it was handed, so a body whose value leaves at a
  // different dtype than it arrived cannot round-trip through it.
  if (!state.enableComputeLambdas)
    return Kind::OutputLegacyForCompute;
  Type inType = epilogueOp.getBody().getArgument(0).getType();
  Type outType = cast<MO::TensorType>(result.getType()).getElementType();
  if (inType != outType)
    return Kind::OutputLegacyForCompute;

  return Kind::Compute;
}

// The trait contract the consuming `map.call.opaque` declares for the fused
// operand `mapResult` feeds. Present only for a trait-ABI fused argument
// (`Fused & ... & Access`), whose functor emit wraps in a conforming trait
// tensor; null for the legacy `FusedInput/OutputTensor` (MTS-API) lane, which
// binds the functor onto a `ManagedTensorSlice`. `isOutput` picks the call's
// output vs input operand list. Resolved here, at MAP level, for the same
// reason `classifyEpilogueLambdaKind` is: the fused-tensor -> call edge is
// still explicit, so emit never walks to recover it.
static MOGG::TensorArgContractAttr
consumingFusedContract(Value mapResult, bool isOutput, MAPToMOGGState &state) {
  if (!mapResult.hasOneUse())
    return {};
  auto callOp = dyn_cast<CallOpaqueOp>(*mapResult.getUsers().begin());
  if (!callOp)
    return {};
  ValueRange operands = isOutput ? callOp.getOutputs() : callOp.getInputs();
  std::optional<unsigned> operandIdx;
  for (auto [i, v] : llvm::enumerate(operands))
    if (v == mapResult) {
      operandIdx = i;
      break;
    }
  if (!operandIdx)
    return {};
  MOGG::KernelDeclAdaptor adaptor(
      state.registry.lookupKernelBySymName(callOp.getCallee()));
  if (!adaptor.executeFunction)
    return {};
  std::optional<MOGG::TensorArgContract> contract =
      MOGG::declaredExecuteOperandContract(*adaptor.executeFunction,
                                           *operandIdx, isOutput);
  if (!contract)
    return {};
  return MOGG::tensorArgContractToAttr(state.builder.getContext(), *contract);
}

// map.iter.epilogue -> mogg.tensor.create.epilogue. Its body is lowered via
// `state.lowerRegionBody`, then the resulting tensor is mapped into
// `state.mapping` under the epilogue's own result, so `CallOpaqueOp` can
// resolve it exactly like a plain kernel output.
LogicalResult IterEpilogueOp::lowerToMOGG(MAPToMOGGState &state) {
  MLIRContext *ctx = getContext();
  Location loc = getLoc();
  Block &epiBody = getBody().front();
  Block::BlockArgListType epiArgs = epiBody.getArguments();
  size_t numCaptures = getCaptures().size();
  if (epiArgs.size() < numCaptures)
    return emitOpError(
        "MAPToMOGG: epilogue has fewer block args than captures");
  // Leading block args are the callback params (value, index); the rest
  // mirror the captured operands.
  size_t numParams = epiArgs.size() - numCaptures;

  // Every capture must already be lowered by this point: either one of the
  // kernel's own outputs (mapped by `IterOpaqueOp::lowerToMOGG`) or a hoisted
  // view's own result (mapped by e.g. `SliceStrideOp::lowerToMOGG`, which
  // runs first since `LoopInvariantViewMotion` always inserts it before the
  // closure that captures it) -- so a plain `BlockArgument` check is too
  // strict now that captures can be either.
  SmallVector<Value> moggCaptures;
  for (Value cap : getCaptures()) {
    if (!state.mapping.contains(cap))
      return emitOpError("MAPToMOGG: epilogue capture is not yet lowered");
    moggCaptures.push_back(state.mapping.lookup(cap));
  }

  // A dynamic store-index reshape cannot evaluate its transform from
  // compile-time shape alone, so its runtime from/to `!mogg.shape`s are
  // threaded into the functor as shape captures. Build them in kernel scope now
  // (where the dim args are live) and remember, per reshape op, where its two
  // shape captures will land so `ReshapeIndexOp::lowerToMOGG` can wire them as
  // its `$dynamicShapes`. The block args are added below, after the create.
  SmallVector<Value> shapeCaptures;
  SmallVector<std::pair<ReshapeIndexOp, unsigned>> reshapeShapeCaptureStart;
  for (ReshapeIndexOp reshape : epiBody.getOps<ReshapeIndexOp>()) {
    MAP::ViewSignatureAttr sig = reshape.getSpec().getSignature();
    if (sig.getOut().isStatic() && sig.getIn().isStatic())
      continue;
    reshapeShapeCaptureStart.emplace_back(reshape, shapeCaptures.size());
    shapeCaptures.push_back(buildShape(loc, sig.getOut(), state));
    shapeCaptures.push_back(buildShape(loc, sig.getIn(), state));
  }

  // Build the output shape / dtype for the tensor.
  auto resultTy = cast<MO::TensorType>(getResult().getType());
  Value outShape = buildShape(loc, resultTy.getShape(), state);
  Type outElemType = resultTy.getElementType();

  ErrorOr<MOGG::KernelOutputFusionLambdaType> kindOr =
      classifyEpilogueLambdaKind(*this, state);
  if (kindOr.isError())
    return emitOpError() << kindOr.getError();
  MOGG::KernelOutputFusionLambdaType kind = *kindOr;
  bool isCompute = kind == MOGG::KernelOutputFusionLambdaType::Compute;

  // A trait-ABI fused output's contract (null for the legacy MTS API). A
  // transform-access output runs the fused body as a value->value functor (the
  // wrapper stores the result through the real output buffer), so its body must
  // YIELD the value like a compute lambda rather than store it -- even for an
  // identity transform with no consumer fused.
  MOGG::TensorArgContractAttr contractAttr =
      consumingFusedContract(getResult(), /*isOutput=*/true, state);
  bool isTransform = contractAttr && MOGG::bitEnumContainsAny(
                                         contractAttr.getAccess(),
                                         MOGG::TensorArgAccessKind::transform);
  bool yieldBody = isCompute || isTransform;
  // A yield-terminated body is a compute-output lambda; a transform's wrapper
  // stores the yielded value, so it satisfies that invariant too. The flag is
  // unused by the transform wrap emit itself.
  bool computeOutputArg =
      kind != MOGG::KernelOutputFusionLambdaType::Output || isTransform;

  // The tensor whose storage the kernel-facing slice is built over.
  // `AllocateAccumulatingBuffers` always sets `$intermediateTensor` (the store
  // target, or a pre-cast accumulation buffer) before this pass runs, so emit
  // gets a real pointer, never a dangling one.
  Value intermediateTensor = getIntermediateTensor();
  if (!intermediateTensor)
    return emitOpError("MAPToMOGG: epilogue has no source tensor");
  Value outputPtr = MOGG::TensorGetPtrOp::create(
      state.builder, loc, MOGG::PointerType::get(ctx),
      state.mapping.lookup(intermediateTensor));

  auto epilogueOp = MOGG::TensorCreateEpilogueOp::create(
      state.builder, loc, MOGG::TensorType::get(ctx), outShape, moggCaptures,
      /*shapeCaptures=*/shapeCaptures,
      /*outputPtr=*/outputPtr,
      state.builder.getStringAttr(state.device.getLabel()),
      TypeAttr::get(outElemType), state.builder.getBoolAttr(true),
      MO::IOKindAttr::get(ctx, MO::IOKind::Output),
      state.builder.getBoolAttr(computeOutputArg),
      /*contract=*/contractAttr);

  // New block args mirror the old ones 1:1, except captures are typed as
  // the reconstructed `!mogg.tensor_` instead of the original `!mo.tensor`.
  // Shape captures, if any, follow the tensor captures as trailing block args.
  Block *body = state.builder.createBlock(&epilogueOp.getBody());
  for (size_t i = 0; i < numParams; ++i) {
    BlockArgument p =
        body->addArgument(epiArgs[i].getType(), epiArgs[i].getLoc());
    state.mapping.map(epiArgs[i], p);
  }
  for (size_t j = 0; j < numCaptures; ++j) {
    BlockArgument capArg = body->addArgument(moggCaptures[j].getType(),
                                             epiArgs[numParams + j].getLoc());
    state.mapping.map(epiArgs[numParams + j], capArg);
  }
  SmallVector<BlockArgument> shapeCaptureArgs;
  for (Value shapeCap : shapeCaptures)
    shapeCaptureArgs.push_back(body->addArgument(shapeCap.getType(), loc));
  for (auto [reshape, start] : reshapeShapeCaptureStart)
    state.reshapeShapeArgs[reshape] = {shapeCaptureArgs[start],
                                       shapeCaptureArgs[start + 1]};

  state.builder.setInsertionPointToEnd(body);
  // A compute lambda returns the value rather than writing it: the kernel owns
  // the one resulting store. Skip lowering that store and yield its value
  // instead. The store is found by the tensor it writes, not by position, so a
  // body that grows a trailing op cannot silently have a different store
  // elided. The store-target capture stays a capture -- emit still binds the
  // functor onto its slice -- it just has no use in the body.
  Value yielded;
  Value storeTarget = yieldBody ? epiBody.getArgument(numParams) : Value();
  if (failed(state.lowerRegionBody(epiBody, [&](Operation *op) {
        auto storeOp = dyn_cast<StoreOp>(op);
        if (!storeOp || storeOp.getTensor() != storeTarget)
          return true;
        ASSERT_STREAM(!yielded, "a MAP epilogue body must store to its "
                                "store-target capture exactly once");
        // Ops lower in order, so the store's value operand is already mapped
        // by the time the walk reaches the store itself.
        yielded = state.mapping.lookup(storeOp.getValue());
        return false;
      })))
    return failure();
  if (yieldBody) {
    ASSERT_STREAM(yielded, "expected a MAP epilogue body to store to its "
                           "store-target capture");
    MOGG::YieldOp::create(state.builder, loc, yielded);
  }

  state.builder.setInsertionPointAfter(epilogueOp);
  state.mapping.map(getResult(), epilogueOp.getTensor());
  return success();
}

// map.iter.prologue -> mogg.tensor.create.prologue. Load-side mirror of
// `IterEpilogueOp::lowerToMOGG`: the body is lowered via
// `state.lowerRegionBody` and the resulting tensor is mapped into
// `state.mapping` under the prologue's own result, so `CallOpaqueOp` resolves
// it exactly like a plain kernel input.
LogicalResult IterPrologueOp::lowerToMOGG(MAPToMOGGState &state) {
  MLIRContext *ctx = getContext();
  Location loc = getLoc();
  Block &proBody = getBody().front();
  Block::BlockArgListType proArgs = proBody.getArguments();
  size_t numCaptures = getCaptures().size();
  // The index, then one arg per capture.
  if (proArgs.size() != 1 + numCaptures)
    return emitOpError()
           << "MAPToMOGG: prologue body takes the index plus one arg per "
              "capture, got "
           << proArgs.size() << " args for " << numCaptures << " captures";

  SmallVector<Value> moggCaptures;
  for (Value cap : getCaptures()) {
    // Every capture must already be lowered by this point: either one of
    // the kernel's own inputs (mapped by `IterOpaqueOp::lowerToMOGG`) or a
    // hoisted view's own result (mapped by e.g. `SliceStrideOp::lowerToMOGG`,
    // which runs first since `LoopInvariantViewMotion` always inserts it
    // before the closure that captures it) -- so a plain `BlockArgument`
    // check is too strict now that captures can be either.
    if (!state.mapping.contains(cap))
      return emitOpError("MAPToMOGG: prologue capture is not yet lowered");
    moggCaptures.push_back(state.mapping.lookup(cap));
  }

  // A dynamic index reshape in the body cannot read its from/to shapes from
  // the static signature, so build them here and thread them in as shape
  // captures -- the load-side mirror of `IterEpilogueOp::lowerToMOGG`.
  SmallVector<Value> shapeCaptures;
  SmallVector<std::pair<ReshapeIndexOp, unsigned>> reshapeShapeCaptureStart;
  for (ReshapeIndexOp reshape : proBody.getOps<ReshapeIndexOp>()) {
    MAP::ViewSignatureAttr sig = reshape.getSpec().getSignature();
    if (sig.getOut().isStatic() && sig.getIn().isStatic())
      continue;
    reshapeShapeCaptureStart.emplace_back(reshape, shapeCaptures.size());
    shapeCaptures.push_back(buildShape(loc, sig.getOut(), state));
    shapeCaptures.push_back(buildShape(loc, sig.getIn(), state));
  }

  // Shape / element type come from the prologue's own RESULT -- the tensor the
  // kernel believes it is reading -- not from a capture: after fusion the
  // captures are the producer's operands, which need not be the value the
  // consumer sees.
  auto resultTy = cast<MO::TensorType>(getResult().getType());
  auto prologueOp = MOGG::TensorCreatePrologueOp::create(
      state.builder, loc, /*tensor=*/MOGG::TensorType::get(ctx),
      /*shape=*/buildShape(loc, resultTy.getShape(), state),
      /*captures=*/moggCaptures, /*shapeCaptures=*/shapeCaptures,
      /*device=*/state.builder.getStringAttr(state.device.getLabel()),
      /*elementType=*/TypeAttr::get(resultTy.getElementType()),
      /*mut=*/state.builder.getBoolAttr(false),
      /*io=*/MO::IOKindAttr::get(ctx, MO::IOKind::Input),
      /*contract=*/
      consumingFusedContract(getResult(), /*isOutput=*/false, state));

  // New block args mirror the old ones 1:1, with the shape captures following
  // as trailing args (as in the epilogue).
  Block *body = state.builder.createBlock(&prologueOp.getBody());
  for (auto [i, arg] : llvm::enumerate(proArgs)) {
    // Arg 0 is the index and carries over; each capture is retyped to the
    // rebuilt `!mogg.tensor_`.
    Type argTy = i == 0 ? arg.getType() : moggCaptures[i - 1].getType();
    state.mapping.map(arg, body->addArgument(argTy, arg.getLoc()));
  }
  SmallVector<BlockArgument> shapeCaptureArgs;
  for (Value shapeCap : shapeCaptures)
    shapeCaptureArgs.push_back(body->addArgument(shapeCap.getType(), loc));
  for (auto [reshape, start] : reshapeShapeCaptureStart)
    state.reshapeShapeArgs[reshape] = {shapeCaptureArgs[start],
                                       shapeCaptureArgs[start + 1]};

  // Lower the region body.
  state.builder.setInsertionPointToEnd(body);
  for (Operation &op : proBody.without_terminator())
    if (failed(state.lowerOp(&op)))
      return failure();

  // Replace the final yield with MOGG::YieldOp.
  auto yieldOp = cast<YieldOp>(proBody.getTerminator());
  SmallVector<Value> yielded;
  for (Value v : yieldOp.getOperands())
    yielded.push_back(state.mapping.lookup(v));
  MOGG::YieldOp::create(state.builder, yieldOp.getLoc(), yielded);

  // Remap the op output.
  state.builder.setInsertionPointAfter(prologueOp);
  state.mapping.map(getResult(), prologueOp.getTensor());
  return success();
}

/// Whether an epilogue recomputes the value the kernel hands it, which is what
/// `has_epilogue_fusion` asks. Its presence answers nothing: `MOToMAP` gives
/// every output a store epilogue, so an unfused one has a body that stores the
/// value untouched. An index transform cannot show up in these uses, and only
/// `ReshapeEpilogueFuser` remaps the store index anyway -- never alone.
static bool epilogueRecomputesValue(MOGG::TensorCreateEpilogueOp epilogue) {
  mlir::Block &body = epilogue.getBody().front();
  if (body.getNumArguments() == 0)
    return false;
  // Stored or yielded as it arrived is a pass-through; every other use derives
  // something from it.
  return !llvm::all_of(body.getArgument(0).getUsers(),
                       [](mlir::Operation *user) {
                         return isa<MOGG::TensorStoreOp, MOGG::YieldOp>(user);
                       });
}

/// Whether a prologue computes the value the kernel reads, rather than merely
/// loading it. Same trap as `epilogueRecomputesValue`: every fusible input gets
/// a load prologue whether or not anything fuses into it. The value is produced
/// here rather than handed in, so the question is what gets yielded.
static bool prologueRecomputesValue(MOGG::TensorCreatePrologueOp prologue) {
  mlir::Block &body = prologue.getBody().front();
  // Block arguments are the callback index, then one per capture. An unfused
  // input yields a plain load of its capture at that index.
  if (body.getNumArguments() < 2)
    return true;
  auto yield = dyn_cast<MOGG::YieldOp>(body.getTerminator());
  if (!yield || yield.getOperands().size() != 1)
    return true;
  auto load = yield.getOperands().front().getDefiningOp<MOGG::TensorLoadOp>();
  return !load || load.getIndex() != body.getArgument(0) ||
         load.getTensor() != body.getArgument(1);
}

// map.call.opaque -> mogg.call.opaque. Each call output -- a plain opaque
// output block arg or a `map.iter.epilogue` result -- is already in
// `state.mapping`, lowered earlier by `IterOpaqueOp::lowerToMOGG`'s
// whole-block walk.
LogicalResult CallOpaqueOp::lowerToMOGG(MAPToMOGGState &state) {
  // `map.call.opaque`'s inputs/outputs are the codegen tensor handles of the
  // enclosing `map.iter.opaque`'s non-chain operands; the chain never reaches
  // it (it is pure MGP sequencing on the kernel's own chain channel), so there
  // is nothing to filter here.
  SmallVector<Value> callInputs;
  for (Value in : getInputs())
    callInputs.push_back(state.mapping.lookup(in));

  // `$outputs` is destination-passing on both sides, so it maps across
  // directly; a by-value opaque is already this op's own result.
  SmallVector<Value> callOutputs;
  for (Value callOut : getOutputs())
    callOutputs.push_back(state.mapping.lookup(callOut));
  Value opaqueOut = getOpaqueResult();
  Type opaqueResultTy = opaqueOut ? opaqueOut.getType() : Type();

  // Bake whether an output is a fused epilogue tensor, and an input a fused
  // prologue tensor, into the op now, so its own emit never has to inspect an
  // operand's producer for this.
  bool hasEpilogueFusion = llvm::any_of(callOutputs, [](Value out) {
    auto epilogue = out.getDefiningOp<MOGG::TensorCreateEpilogueOp>();
    return epilogue && epilogueRecomputesValue(epilogue);
  });
  bool hasPrologueFusion = llvm::any_of(callInputs, [](Value in) {
    auto prologue = in.getDefiningOp<MOGG::TensorCreatePrologueOp>();
    return prologue && prologueRecomputesValue(prologue);
  });

  // Emit appends the contexts verbatim. A host-only kernel's execute takes
  // none; a device kernel takes one, or a variadic `DeviceContextArray` when
  // distributed. Pass the contexts the kernel body created only when execute
  // actually declares one.
  MOGG::KernelDeclOp decl = state.registry.lookupKernelBySymName(getCallee());
  MOGG::KernelDeclAdaptor adaptor(decl);
  ValueRange contexts;
  if (const auto &fn = adaptor.executeFunction;
      fn && !fn->inputArguments.empty() &&
      (fn->inputArguments.back().isContextType() ||
       fn->inputArguments.back().isVariadicContextType()))
    contexts = state.contextArgs;

  auto call = MOGG::CallOpaqueOp::create(
      state.builder, getLoc(),
      opaqueResultTy ? TypeRange{opaqueResultTy} : TypeRange{}, getCalleeAttr(),
      callInputs, callOutputs, contexts, getParams(),
      state.builder.getBoolAttr(hasEpilogueFusion),
      state.builder.getBoolAttr(hasPrologueFusion),
      state.builder.getBoolAttr(getHasSameVariadicOperandSize()));
  if (opaqueOut)
    state.mapping.map(opaqueOut, call.getOpaqueResult());
  return success();
}

// A hoisted view whose input is an in-place kernel's fused buffer re-addresses
// mutable storage, so `mut` has to say so; the result stays a `!mogg.tensor_`
// either way, since `LoopInvariantViewMotion` only hoists views that are
// loaded from.
static bool viewsMutableStorage(Value input) {
  return isa<MO::BufferType>(input.getType());
}

// map.index.slice.stride -> mogg.tensor.create.slice. Reads its own
// input/starts/stops/steps out of `state.mapping`, same as any other kernel
// input.
LogicalResult SliceStrideOp::lowerToMOGG(MAPToMOGGState &state) {
  MLIRContext *ctx = getContext();
  MAP::ViewSignatureAttr signature = getSpec().getSignature();
  Type elementType =
      MO::castFromTensorLike(getInput().getType()).getElementType();

  // Only a negative start on a dynamic dim is unresolvable: MOToMAP resolves
  // Python-style negatives against the dim's extent, so `[-window:]` off a
  // dynamic `seq_len` stays a raw literal whose real offset `dim + start` is
  // unknown until runtime. Mark just those `?` so the slice primitive reads
  // them from the runtime `starts` operand; a non-negative start is already an
  // absolute offset and stays a comptime hint even on a dynamic dim.
  SmallVector<int64_t> inDims = signature.getIn().getIntDims();
  SmallVector<int64_t> staticStart =
      llvm::to_vector(getSpec().getStarts().asArrayRef());
  for (size_t i = 0; i < staticStart.size(); ++i)
    if (staticStart[i] < 0 && ShapedType::isDynamic(inDims[i]))
      staticStart[i] = ShapedType::kDynamic;

  auto createSlice = MOGG::TensorCreateSliceOp::create(
      state.builder, getLoc(), MOGG::TensorType::get(ctx), signature.getOut(),
      DenseI64ArrayAttr::get(ctx, staticStart), getSpec().getSteps(),
      TypeAttr::get(elementType),
      state.builder.getBoolAttr(viewsMutableStorage(getInput())),
      state.mapping.lookup(getInput()), state.mapping.lookup(getStarts()),
      state.mapping.lookup(getStops()), state.mapping.lookup(getSteps()));
  state.mapping.map(getResult(), createSlice.getResult());
  return success();
}

// map.index.broadcast.stride -> mogg.tensor.create.broadcast. Reads its own
// input out of `state.mapping`, same as any other kernel input.
LogicalResult BroadcastStrideOp::lowerToMOGG(MAPToMOGGState &state) {
  MLIRContext *ctx = getContext();
  MAP::ViewSignatureAttr signature = getSpec().getSignature();
  Type elementType =
      MO::castFromTensorLike(getInput().getType()).getElementType();
  auto createBroadcast = MOGG::TensorCreateBroadcastOp::create(
      state.builder, getLoc(), MOGG::TensorType::get(ctx), signature.getOut(),
      TypeAttr::get(elementType),
      state.builder.getBoolAttr(viewsMutableStorage(getInput())),
      state.mapping.lookup(getInput()));
  state.mapping.map(getResult(), createBroadcast.getResult());
  return success();
}

// map.index.reshape.stride -> mogg.tensor.create.reshape. Reads its own
// input out of `state.mapping`, same as any other kernel input.
//
// Unlike broadcast/transpose, reshape's own runtime stride is computed
// directly from its output shape's values (row-major de-linearization has
// no input stride to fall back on), so a target shape with more than one
// dim genuinely unknown at compile time -- e.g. reshaping into another
// tensor's own dynamic shape -- needs a real runtime `!mogg.shape`, built via
// the same `buildShape` helper `mogg.tensor.create` uses -- letting CSE fold
// it with an identical shape built elsewhere in the same kernel.
LogicalResult ReshapeStrideOp::lowerToMOGG(MAPToMOGGState &state) {
  MLIRContext *ctx = getContext();
  MAP::ViewSignatureAttr signature = getSpec().getSignature();
  Type elementType =
      MO::castFromTensorLike(getInput().getType()).getElementType();
  Value shape = buildShape(getLoc(), signature.getOut(), state);
  auto createReshape = MOGG::TensorCreateReshapeOp::create(
      state.builder, getLoc(), MOGG::TensorType::get(ctx), signature.getOut(),
      TypeAttr::get(elementType),
      state.builder.getBoolAttr(viewsMutableStorage(getInput())),
      state.mapping.lookup(getInput()), shape);
  state.mapping.map(getResult(), createReshape.getResult());
  return success();
}

// map.index.reshape -> mogg.index.reshape. Unlike the `.stride` view ops,
// this per-index transform is NOT hoisted out of the loop (it depends on the
// loop index and feeds a store, e.g. a reshape fused into an epilogue), so it
// lowers in place. The op maps an index in the spec's OUT shape to one in the
// IN shape; the MOGG op names those `fromShape` (the index's space) and
// `toShape` (the result's space).
LogicalResult ReshapeIndexOp::lowerToMOGG(MAPToMOGGState &state) {
  MAP::ViewSignatureAttr signature = getSpec().getSignature();
  // A dynamic reshape has its runtime from/to shapes threaded in as shape
  // captures by whichever functor owns it -- `IterEpilogueOp` on the store
  // side, `IterPrologueOp` on the load side. Wire those block args here as
  // `$dynamicShapes` so emit can evaluate the transform at runtime; order
  // matches the build there: from-shape, then to-shape.
  ValueRange dynamicShapes;
  if (!signature.getOut().isStatic() || !signature.getIn().isStatic()) {
    auto it = state.reshapeShapeArgs.find(getOperation());
    // Emit cannot tell "no runtime shapes needed" from "they went missing" --
    // both leave `$dynamicShapes` empty and silently de-linearize against the
    // static defaults -- so a miss has to fail here.
    if (it == state.reshapeShapeArgs.end())
      return emitOpError("MAPToMOGG: a dynamic index reshape has no shape "
                         "captures; its parent functor did not thread them in");
    dynamicShapes = ValueRange(it->second);
  }
  auto indexReshape = MOGG::IndexReshapeOp::create(
      state.builder, getLoc(), getResult().getType(),
      state.mapping.lookup(getIndex()), signature.getOut(), signature.getIn(),
      dynamicShapes);
  state.mapping.map(getResult(), indexReshape.getResult());
  return success();
}

// map.index.transpose.stride -> mogg.tensor.create.transpose. Reads its own
// input out of `state.mapping`, same as any other kernel input.
LogicalResult TransposeStrideOp::lowerToMOGG(MAPToMOGGState &state) {
  MLIRContext *ctx = getContext();
  MAP::ViewSignatureAttr signature = getSpec().getSignature();
  Type elementType =
      MO::castFromTensorLike(getInput().getType()).getElementType();
  auto createTranspose = MOGG::TensorCreateTransposeOp::create(
      state.builder, getLoc(), MOGG::TensorType::get(ctx), signature.getOut(),
      getSpec().getPerm(), TypeAttr::get(elementType),
      state.builder.getBoolAttr(viewsMutableStorage(getInput())),
      state.mapping.lookup(getInput()),
      state.mapping.lookup(getPermutations()));
  state.mapping.map(getResult(), createTranspose.getResult());
  return success();
}

// map.iter.foreach.body -> mogg.call.foreach. Builds the loop shape and the
// captured tensors, then lowers the body into the new region via
// `state.lowerRegionBody`.
LogicalResult IterForeachBodyOp::lowerToMOGG(MAPToMOGGState &state) {
  MLIRContext *ctx = getContext();
  Location loc = getLoc();
  // The loop's own iteration shape comes from `map.iter.foreach.body`'s
  // `$shape` attribute, not from any I/O tensor's shape (they happen to
  // coincide for a plain, unfused elementwise op like `mo.add`, but not in
  // general -- a broadcast could differ).
  Value loopShape = buildShape(loc, getShape(), state);

  // The foreach driver is one generic primitive, not a per-kernel lookup.
  MOGG::PrimitiveDeclOp foreachPrim =
      state.registry.lookupPrimitive(MOGG::CallForeachOp::getOperationName());

  // `$captures` already covers the kernel's inputs/output and any hoisted
  // view uniformly (`map.iter.foreach.body` is isolated from above, so every
  // outer value the body needs -- there is nothing else it could reference --
  // is one of these, in declaration order). Each is already lowered (and
  // registered in `state.mapping`) by the time this op is reached, since
  // block order visits its definition first.
  Block &body = getBody().front();
  Value oldIndexArg = body.getArgument(0);
  ValueRange captures = getCaptures();
  Block::BlockArgListType captureBlockArgs =
      body.getArguments().take_back(captures.size());

  SmallVector<Value> capturing;
  for (Value v : captures)
    capturing.push_back(state.mapping.lookup(v));

  // A foreach kernel is elementwise and single-device, so it always has the one
  // context arg `IterOpaqueOp::lowerToMOGG` created.
  auto callForeach = MOGG::CallForeachOp::create(
      state.builder, loc, FlatSymbolRefAttr::get(ctx, foreachPrim.getSymName()),
      loopShape, state.contextArgs.front(), capturing,
      state.builder.getStringAttr(state.device.getLabel()));

  // New block: one arg per capture (same order), then the index.
  Block *foreachBlock = state.builder.createBlock(&callForeach.getBody());
  for (auto [i, v] : llvm::enumerate(captureBlockArgs))
    state.mapping.map(v,
                      foreachBlock->addArgument(capturing[i].getType(), loc));
  state.mapping.map(oldIndexArg,
                    foreachBlock->addArgument(oldIndexArg.getType(), loc));

  state.builder.setInsertionPointToStart(foreachBlock);
  if (failed(state.lowerRegionBody(body)))
    return failure();

  state.builder.setInsertionPointAfter(callForeach);
  return success();
}

// map.iter.opaque -> mogg.kernel. Builds the kernel region, device and
// fusion-info attrs, dynamic-dim block args, and the reconstructed
// input/output tensors, then lowers the body (a `map.call.opaque` or a
// `map.iter.foreach.body`) via `state.lowerRegionBody`.
LogicalResult IterOpaqueOp::lowerToMOGG(MAPToMOGGState &state) {
  MLIRContext *ctx = getContext();
  Location loc = getLoc();
  ValueRange inputs = getInputs();
  TypeRange resultTypes = getResultTypes();

  // `map.iter.opaque` threads its chain through a dedicated `$inputChain`
  // operand and a trailing output-chain result, mirroring `mogg.kernel` --
  // but (unlike `mogg.kernel`'s own body) still shadows it as a body block
  // argument on both sides, since `map.call.opaque` has no return channel of
  // its own for a wrapped call that consumes/produces a chain directly.
  Value inputChain = getInputChain();
  bool hasOutputChain = getOutputChain() != nullptr;

  // The execution device is stated outright on `map.iter.opaque`
  // (`MO_ExplicitDevice`), stamped by whoever built the op from the tensor
  // types back when they were still annotated. Only the label is recorded on
  // each reconstructed tensor (a string -- codegen picks host vs GPU from it);
  // the full ref is kept on the `mogg.kernel` for device-context resolution.
  state.device = MO::getDeviceOrDefault(*this);

  // Forward the fusion-tree provenance `MOToMAP` recorded on `map.iter.opaque`
  // (typed `AnyAttr` there so the `map` dialect need not depend on `mogg`);
  // fall back to a fresh leaf only if it's absent (e.g. hand-written test IR).
  Attribute rawFusionInfo = getFusionInfoAttr();
  MOGG::FusionInfoAttr fusionInfo =
      rawFusionInfo
          ? cast<MOGG::FusionInfoAttr>(rawFusionInfo)
          : MOGG::FusionInfoAttr::get(ctx, MOGG::createFusionLeaf(*this));

  OpBuilder &builder = state.builder;
  auto kernel = MOGG::KernelOp::create(builder, loc, resultTypes, inputs,
                                       inputChain, fusionInfo);
  // The `MOExplicitDevice` verifier requires a device; record the same
  // device used for the reconstructed tensors so outlining and `MOGGToMGP`
  // resolve the real execution device instead of the host fallback.
  kernel.setDeviceRef(state.device);
  // Carry the effects the MAP op holds. A kernel whose only observable effect
  // is a mutable store has no used result, so without them it reports none and
  // is dead -- the store is dropped and the buffer never written.
  kernel.setHasReadEffect(getHasReadEffect());
  kernel.setHasWriteEffect(getHasWriteEffect());
  kernel.setIsAsync(getIsAsync());

  // Unique dynamic dims the kernel references anywhere -- not just its I/O
  // signature. A dim can live only on an intermediate value or a view's spec:
  // e.g. the reduced axis of a dynamic-perm transpose fused into a reduce is a
  // fresh runtime symbol on the transposed view, absent from both the kernel's
  // inputs and its result, yet `buildShape` still needs a `!mogg.dim_` handle
  // for it. `collectDynamicDims` walks for any `MOSH::ShapeAttr`, so the
  // opaque's own I/O plus every op in the body covers all of them.
  collectDynamicDims(getOperation(), state);
  getBody().walk([&](Operation *op) { collectDynamicDims(op, state); });

  // Body block args: one per non-chain input, then per destination-passing
  // output, then the dims, then the context. A tensor-like entry gets a
  // pointer to rebuild from; an opaque input crosses as itself (no shape to
  // rebuild), and an opaque result gets none at all -- it returns by value.
  Block *body = builder.createBlock(&kernel.getBody());
  auto addBodyArg = [&](Type ty) {
    return body->addArgument(
        isa<MO::OpaqueType>(ty) ? ty : Type(MOGG::PointerType::get(ctx)), loc);
  };
  SmallVector<BlockArgument> inputArgs;
  // Indexed by result number; null where a result takes no block argument (a
  // by-value opaque, or the trailing chain).
  SmallVector<BlockArgument> outputArgs(resultTypes.size(), BlockArgument());
  for (Value in : inputs)
    inputArgs.push_back(addBodyArg(in.getType()));
  // Neither the trailing chain nor a by-value opaque gets a block arg, so both
  // are skipped -- matching `mogg.kernel`'s verifier.
  for (size_t i = 0, e = resultTypes.size(); i < e; ++i) {
    if (hasOutputChain && i + 1 == e)
      break;
    if (isa<MO::OpaqueType>(resultTypes[i]))
      continue;
    outputArgs[i] = addBodyArg(resultTypes[i]);
  }
  // One `!mogg.dim_` per unique dynamic dim, each carrying that dim's
  // parameter name (e.g. `!mogg.dim_<batch>`); shapes never hold dim
  // expressions, so the dynamic dim is always a `ParamDeclRefAttr`.
  for (auto &[dim, blockArg] : state.dimArgs)
    blockArg = body->addArgument(
        MOGG::DimType::get(ctx, dyn_cast<KGEN::ParamDeclRefAttr>(dim)), loc);
  // Context block args. A distributed kernel whose `execute` takes a
  // `DeviceContextArray` gets one `!mogg.context` per unique device (in
  // `getUniqueDevicesFromKernel` order, which `lowerToCompiledRegion`
  // re-derives to materialize each), and `ConformCallOperands` packs
  // them into a tuple-of-contexts. Every other kernel gets exactly one: its
  // `execute` takes a single `DeviceContext`, and the unique-device COUNT is
  // not a proxy for the context count -- an ordinary single-compute kernel can
  // still name several devices (e.g. a GPU op with CPU scalar operands like
  // `band_part`'s bounds or `pad`'s paddings), which would otherwise pass
  // contexts `execute` never declared. So gate the per-device fan-out on the
  // callee actually taking a variadic context.
  //
  // TODO(GEX-3701): label each context with its `DeviceRefAttr` instead of
  // relying on creation order matching the order `lowerToCompiledRegion`
  // re-derives -- any reordering of `getUniqueDevicesFromKernel` would silently
  // misassign devices.
  bool variadicContext = false;
  getBody().walk([&](CallOpaqueOp call) {
    MOGG::KernelDeclOp decl =
        state.registry.lookupKernelBySymName(call.getCallee());
    if (!decl)
      return WalkResult::advance();
    MOGG::KernelDeclAdaptor adaptor(decl);
    if (!adaptor.executeFunction)
      return WalkResult::advance();
    for (const auto &arg : adaptor.executeFunction->inputArguments)
      if (arg.isVariadicContextType()) {
        // The flag is monotonic, so stop at the first variadic-context arg.
        variadicContext = true;
        return WalkResult::interrupt();
      }
    return WalkResult::advance();
  });
  size_t numContexts =
      variadicContext
          ? std::max<size_t>(1, MO::getUniqueDevicesFromKernel(*this).size())
          : 1;
  state.contextArgs.clear();
  for (size_t i = 0; i < numContexts; ++i)
    state.contextArgs.push_back(
        body->addArgument(MOGG::ContextType::get(ctx), loc));

  builder.setInsertionPointToEnd(body);

  // `map.iter.opaque` is isolated from above: nested ops reference its
  // shadowed input block args, not `inputs`/`$inputChain` (its outer
  // operands) directly, so the mapping must be keyed on the block args.
  // The chain has no block argument on either side: `$inputChain` is already
  // the kernel's own chain operand (passed to `KernelOp::create` above)
  // and the trailing chain result is the kernel's own chain result (wired by
  // `replaceAllUsesWith` below). Nothing in the body references it, so the
  // block args here are exactly the non-chain inputs then the non-chain
  // outputs.
  Block::BlockArgListType inputBlockArgs = getInputBlockArgs();
  for (auto [i, in] : llvm::enumerate(inputs)) {
    // An opaque body arg already *is* the value the call wants; there is no
    // pointer to reconstruct from.
    if (isa<MO::OpaqueType>(in.getType())) {
      state.mapping.map(inputBlockArgs[i], inputArgs[i]);
      continue;
    }
    // A `!mo.buffer` input (e.g. a collective's signal buffers) is mutable
    // memory the kernel writes through, so reconstruct it as a `!mogg.buffer_`
    // (mut) rather than a `!mogg.tensor_`; the kernel decl expects a mutable
    // operand there.
    bool isMut = isa<MO::BufferType>(in.getType());
    state.mapping.map(inputBlockArgs[i],
                      buildTensor(loc, inputArgs[i], in.getType(),
                                  MO::IOKind::Input, isMut, state));
  }

  // Every output block arg is destination-passing: a by-value opaque has none.
  Block &opaqueBody = getBody().front();
  for (auto [i, outputBlockArg] : llvm::enumerate(getOutputBlockArgs()))
    state.mapping.map(outputBlockArg,
                      buildTensor(loc, outputArgs[i], resultTypes[i],
                                  MO::IOKind::Output, /*isMut=*/true, state));

  // `map.yield` has no MOGG op of its own: it becomes the `mogg.output` that
  // forwards the by-value result, once the call producing it is lowered.
  auto yieldOp = dyn_cast<YieldOp>(opaqueBody.back());
  if (failed(state.lowerRegionBody(
          opaqueBody, [&](Operation *op) { return op != yieldOp; })))
    return failure();

  // Forward only a genuine by-value result. A `map.yield` may also restate a
  // DPS output block arg, which is redundant and must not become an output.
  if (getOpaqueResult() && yieldOp) {
    Block::BlockArgListType outputArgs = getOutputBlockArgs();
    for (Value operand : yieldOp.getOperands())
      if (!llvm::is_contained(outputArgs, operand)) {
        MOGG::OutputOp::create(builder, loc, state.mapping.lookup(operand));
        break;
      }
  }

  replaceAllUsesWith(kernel.getResults());
  erase();
  return success();
}

} // namespace M::MAP
