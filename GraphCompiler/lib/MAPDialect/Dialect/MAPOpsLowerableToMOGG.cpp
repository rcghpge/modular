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

#include "GraphCompiler/Interfaces/MAPToMOGGState.h"
#include "GraphCompiler/MAPDialect/Dialect/MAPOps.h"
#include "GraphCompiler/MODialect/MODevices.h"
#include "GraphCompiler/MODialect/MOTypes.h"
#include "GraphCompiler/MOGGDialect/Analysis/KernelDeclAdaptor.h"
#include "GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GraphCompiler/MOGGDialect/Support/FusionUtils.h"
#include "GraphCompiler/MOGGDialect/Support/MojoIdentifiers.h"
#include "GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "Mojo/KGENDialect/KGENAttrs.h"
#include "llvm/ADT/SetVector.h"

using namespace mlir;

namespace M::MAP {

namespace {

// A dimension is dynamic unless it is a concrete `KGEN::SIMDAttr`.
bool isDynamicDim(Attribute dim) { return !isa<KGEN::SIMDAttr>(dim); }

// Materializes a `!mogg.shape` for `shapeAttr` via `mogg._shape.create`.
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

// Reconstructs an `!mogg.tensor_` for `tensorType` from the `!mogg.pointer_`
// block argument `ptr`, via `mogg._shape.create` + `mogg._tensor.create`.
// `io`/`isMut` distinguish immutable inputs from mutable outputs.
Value buildTensor(Location loc, Value ptr, Type tensorType, MO::IOKind io,
                  bool isMut, MAPToMOGGState &state) {
  auto tensorTy = cast<MO::TensorType>(tensorType);
  MLIRContext *ctx = state.builder.getContext();
  MOSH::ShapeAttr shapeAttr = tensorTy.getShape();
  Value shape = buildShape(loc, shapeAttr, state);
  return MOGG::TensorCreateOp::create(
      state.builder, loc, MOGG::TensorType::get(ctx), ptr, shape, shapeAttr,
      state.device, TypeAttr::get(tensorTy.getElementType()),
      state.builder.getBoolAttr(isMut), MO::IOKindAttr::get(ctx, io));
}

} // namespace

// map.load -> mogg._tensor.load.
LogicalResult LoadOp::lowerToMOGG(MAPToMOGGState &state) {
  Value result = MOGG::TensorLoadOp::create(
      state.builder, getLoc(), getResult().getType(),
      state.mapping.lookup(getTensor()), state.mapping.lookup(getIndex()));
  state.mapping.map(getResult(), result);
  return success();
}

// map.store -> mogg._tensor.store.
LogicalResult StoreOp::lowerToMOGG(MAPToMOGGState &state) {
  MOGG::TensorStoreOp::create(
      state.builder, getLoc(), state.mapping.lookup(getTensor()),
      state.mapping.lookup(getIndex()), state.mapping.lookup(getValue()));
  return success();
}

// map.call.elementwise -> mogg._call.elementwise. `$moOpName` isn't carried
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
  auto moggCall = MOGG::CallElementwiseV2Op::create(
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
//
// The MOGG-ABI sibling of this is `MOGG::getKernelOutputFusionLambdaType`,
// which applies the same IOSpec and tile rules to a `mogg.tensor.fused`. The
// two deliberately live apart: no MOGG header depends on the MAP dialect, and
// a MAP-typed entry point there would invert that. Keep the rules in sync.
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

// map.iter.epilogue -> mogg._tensor.create.epilogue. Its body is lowered via
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
  bool computeOutputArg = kind != MOGG::KernelOutputFusionLambdaType::Output;

  auto epilogueOp = MOGG::TensorCreateEpilogueOp::create(
      state.builder, loc, MOGG::TensorType::get(ctx), outShape, moggCaptures,
      /*args=*/ValueRange{}, state.device, TypeAttr::get(outElemType),
      state.builder.getBoolAttr(true),
      MO::IOKindAttr::get(ctx, MO::IOKind::Output),
      state.builder.getBoolAttr(computeOutputArg));

  // New block args mirror the old ones 1:1, except captures are typed as
  // the reconstructed `!mogg.tensor_` instead of the original `!mo.tensor`.
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

  state.builder.setInsertionPointToEnd(body);
  // A compute lambda returns the value rather than writing it: the kernel owns
  // the one resulting store. Skip lowering that store and yield its value
  // instead. The store is found by the tensor it writes, not by position, so a
  // body that grows a trailing op cannot silently have a different store
  // elided. The store-target capture stays a capture -- emit still binds the
  // functor onto its slice -- it just has no use in the body.
  Value yielded;
  Value storeTarget = isCompute ? epiBody.getArgument(numParams) : Value();
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
  if (isCompute) {
    ASSERT_STREAM(yielded, "expected a MAP epilogue body to store to its "
                           "store-target capture");
    MOGG::YieldOp::create(state.builder, loc, yielded);
  }

  state.builder.setInsertionPointAfter(epilogueOp);
  state.mapping.map(getResult(), epilogueOp.getTensor());
  return success();
}

// map.iter.prologue -> mogg._tensor.create.prologue. Load-side mirror of
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

  // Shape / element type come from the prologue's own RESULT -- the tensor the
  // kernel believes it is reading -- not from a capture: after fusion the
  // captures are the producer's operands, which need not be the value the
  // consumer sees.
  auto resultTy = cast<MO::TensorType>(getResult().getType());
  auto prologueOp = MOGG::TensorCreatePrologueOp::create(
      state.builder, loc, /*tensor=*/MOGG::TensorType::get(ctx),
      /*shape=*/buildShape(loc, resultTy.getShape(), state),
      /*captures=*/moggCaptures, /*device=*/state.device,
      /*elementType=*/TypeAttr::get(resultTy.getElementType()),
      /*mut=*/state.builder.getBoolAttr(false),
      /*io=*/MO::IOKindAttr::get(ctx, MO::IOKind::Input));

  // New block args mirror the old ones 1:1.
  Block *body = state.builder.createBlock(&prologueOp.getBody());
  for (auto [i, arg] : llvm::enumerate(proArgs)) {
    // Arg 0 is the index and carries over; each capture is retyped to the
    // rebuilt `!mogg.tensor_`.
    Type argTy = i == 0 ? arg.getType() : moggCaptures[i - 1].getType();
    state.mapping.map(arg, body->addArgument(argTy, arg.getLoc()));
  }

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

// map.call.opaque -> mogg._call.opaque. Each call output -- a plain opaque
// output block arg or a `map.iter.epilogue` result -- is already in
// `state.mapping`, lowered earlier by `IterOpaqueOp::lowerToMOGG`'s
// whole-block walk.
LogicalResult CallOpaqueOp::lowerToMOGG(MAPToMOGGState &state) {
  SmallVector<Value> callInputs;
  for (Value in : getInputs())
    callInputs.push_back(state.mapping.lookup(in));

  SmallVector<Value> callOutputs;
  for (Value callOut : getOutputs())
    callOutputs.push_back(state.mapping.lookup(callOut));

  // Bake whether an output is a fused epilogue tensor, and an input a fused
  // prologue tensor, into the op now, so its own emit never has to inspect an
  // operand's producer for this.
  bool hasEpilogueFusion = llvm::any_of(callOutputs, [](Value out) {
    return out.getDefiningOp<MOGG::TensorCreateEpilogueOp>() != nullptr;
  });
  bool hasPrologueFusion = llvm::any_of(callInputs, [](Value in) {
    return in.getDefiningOp<MOGG::TensorCreatePrologueOp>() != nullptr;
  });

  MOGG::CallOpaqueOp::create(state.builder, getLoc(), getCalleeAttr(),
                             callInputs, callOutputs, state.contextArg,
                             getParams(),
                             state.builder.getBoolAttr(hasEpilogueFusion),
                             state.builder.getBoolAttr(hasPrologueFusion));
  return success();
}

// map.index.slice.stride -> mogg._tensor.create.slice. Reads its own
// input/starts/stops/steps out of `state.mapping`, same as any other kernel
// input.
LogicalResult SliceStrideOp::lowerToMOGG(MAPToMOGGState &state) {
  MLIRContext *ctx = getContext();
  MAP::ViewSignatureAttr signature = getSpec().getSignature();
  Type elementType =
      cast<MO::TensorType>(getInput().getType()).getElementType();
  auto createSlice = MOGG::TensorCreateSliceOp::create(
      state.builder, getLoc(), MOGG::TensorType::get(ctx), signature.getOut(),
      getSpec().getStarts(), getSpec().getSteps(), TypeAttr::get(elementType),
      state.builder.getBoolAttr(false), state.mapping.lookup(getInput()),
      state.mapping.lookup(getStarts()), state.mapping.lookup(getStops()),
      state.mapping.lookup(getSteps()));
  state.mapping.map(getResult(), createSlice.getResult());
  return success();
}

// map.index.broadcast.stride -> mogg._tensor.create.broadcast. Reads its own
// input out of `state.mapping`, same as any other kernel input.
LogicalResult BroadcastStrideOp::lowerToMOGG(MAPToMOGGState &state) {
  MLIRContext *ctx = getContext();
  MAP::ViewSignatureAttr signature = getSpec().getSignature();
  Type elementType =
      cast<MO::TensorType>(getInput().getType()).getElementType();
  auto createBroadcast = MOGG::TensorCreateBroadcastOp::create(
      state.builder, getLoc(), MOGG::TensorType::get(ctx), signature.getOut(),
      TypeAttr::get(elementType), state.builder.getBoolAttr(false),
      state.mapping.lookup(getInput()));
  state.mapping.map(getResult(), createBroadcast.getResult());
  return success();
}

// map.index.reshape.stride -> mogg._tensor.create.reshape. Reads its own
// input out of `state.mapping`, same as any other kernel input.
LogicalResult ReshapeStrideOp::lowerToMOGG(MAPToMOGGState &state) {
  MLIRContext *ctx = getContext();
  MAP::ViewSignatureAttr signature = getSpec().getSignature();
  Type elementType =
      cast<MO::TensorType>(getInput().getType()).getElementType();
  auto createReshape = MOGG::TensorCreateReshapeOp::create(
      state.builder, getLoc(), MOGG::TensorType::get(ctx), signature.getOut(),
      TypeAttr::get(elementType), state.builder.getBoolAttr(false),
      state.mapping.lookup(getInput()));
  state.mapping.map(getResult(), createReshape.getResult());
  return success();
}

// map.index.transpose.stride -> mogg._tensor.create.transpose. Reads its own
// input out of `state.mapping`, same as any other kernel input.
LogicalResult TransposeStrideOp::lowerToMOGG(MAPToMOGGState &state) {
  MLIRContext *ctx = getContext();
  MAP::ViewSignatureAttr signature = getSpec().getSignature();
  Type elementType =
      cast<MO::TensorType>(getInput().getType()).getElementType();
  auto createTranspose = MOGG::TensorCreateTransposeOp::create(
      state.builder, getLoc(), MOGG::TensorType::get(ctx), signature.getOut(),
      getSpec().getPerm(), TypeAttr::get(elementType),
      state.builder.getBoolAttr(false), state.mapping.lookup(getInput()));
  state.mapping.map(getResult(), createTranspose.getResult());
  return success();
}

// map.iter.foreach.body -> mogg._call.foreach. Builds the loop shape and the
// captured tensors, then lowers the body into the new region via
// `state.lowerRegionBody`.
LogicalResult IterForeachBodyOp::lowerToMOGG(MAPToMOGGState &state) {
  MLIRContext *ctx = getContext();
  Location loc = getLoc();
  auto iterOpaqueKernel = (*this)->getParentOfType<IterOpaqueOp>();
  if (iterOpaqueKernel.getResults().size() != 1)
    return emitOpError(
        "MAPToMOGG: elementwise lowering supports exactly one output");

  // The loop's own iteration shape comes from `map.iter.foreach.body`'s
  // `$shape` attribute, not from any I/O tensor's shape (they happen to
  // coincide for a plain, unfused elementwise op like `mo.add`, but not in
  // general -- a broadcast could differ).
  Value loopShape = buildShape(loc, getShape(), state);

  // The foreach driver is one generic primitive, not a per-kernel lookup.
  MOGG::PrimitiveDeclOp foreachPrim = state.registry.lookupPrimitive(
      MOGG::CallForeachRegionOp::getOperationName());

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

  auto callForeach = MOGG::CallForeachRegionOp::create(
      state.builder, loc, FlatSymbolRefAttr::get(ctx, foreachPrim.getSymName()),
      loopShape, state.contextArg, capturing, state.device);

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

// map.iter.opaque -> mogg._kernel. Builds the kernel region, device and
// fusion-info attrs, dynamic-dim block args, and the reconstructed
// input/output tensors, then lowers the body (a `map.call.opaque` or a
// `map.iter.foreach.body`) via `state.lowerRegionBody`.
LogicalResult IterOpaqueOp::lowerToMOGG(MAPToMOGGState &state) {
  MLIRContext *ctx = getContext();
  Location loc = getLoc();
  ValueRange inputs = getInputs();
  TypeRange resultTypes = getResultTypes();

  // The execution device is stated outright on `map.iter.opaque`
  // (`MO_ExplicitDevice`), stamped by whoever built the op from the tensor
  // types back when they were still annotated. This device is recorded on each
  // reconstructed tensor; its ID is later zeroed by
  // `stripDeviceIdsAndDynamicShapes` for dedup, matching the stub's device.
  state.device = MO::getDeviceOrDefault(*this);

  OpBuilder &builder = state.builder;
  auto kernel = MOGG::KernelRegionOp::create(builder, loc, resultTypes, inputs);
  // The `MOExplicitDevice` verifier requires a device; record the same
  // device used for the reconstructed tensors so outlining and `MOGGToMGP`
  // resolve the real execution device instead of the host fallback.
  kernel.setDeviceRef(state.device);

  // Forward the fusion-tree provenance `MOToMAP` recorded on `map.iter.opaque`
  // (typed `AnyAttr` there so the `map` dialect need not depend on `mogg`);
  // fall back to a fresh leaf only if it's absent (e.g. hand-written test IR).
  if (Attribute fusionInfo = getFusionInfoAttr())
    kernel.setFusionInfoAttr(cast<MOGG::FusionInfoAttr>(fusionInfo));
  else
    kernel.setFusionInfoAttr(
        MOGG::FusionInfoAttr::get(ctx, MOGG::createFusionLeaf(*this)));

  // Unique dynamic dims across this kernel's I/O tensor (or buffer) types,
  // plus any nested `map.iter.foreach.body`'s own `$shape` (which can differ
  // from any I/O tensor's shape for a broadcast).
  auto collectDims = [&](TypeRange types) {
    for (Type ty : types)
      for (Attribute dim : MO::castFromTensorLike(ty).getShape().getValues())
        if (isDynamicDim(dim))
          state.dimArgs.insert({dim, BlockArgument()});
  };
  collectDims(inputs.getTypes());
  collectDims(resultTypes);
  getBody().walk([&](IterForeachBodyOp nested) {
    for (Attribute dim : nested.getShape().getValues())
      if (isDynamicDim(dim))
        state.dimArgs.insert({dim, BlockArgument()});
  });

  // Body block args: one !mogg.pointer_ per input, then per output, then one
  // !mogg.dim_ per unique dynamic dim, then the context.
  Block *body = builder.createBlock(&kernel.getBody());
  SmallVector<BlockArgument> inputPtrs, outputPtrs;
  for (size_t i = 0, e = inputs.size(); i < e; ++i)
    inputPtrs.push_back(body->addArgument(MOGG::PointerType::get(ctx), loc));
  for (size_t i = 0, e = resultTypes.size(); i < e; ++i)
    outputPtrs.push_back(body->addArgument(MOGG::PointerType::get(ctx), loc));
  // One `!mogg.dim_` per unique dynamic dim, each carrying that dim's
  // parameter name (e.g. `!mogg.dim_<batch>`); shapes never hold dim
  // expressions, so the dynamic dim is always a `ParamDeclRefAttr`.
  for (auto &[dim, blockArg] : state.dimArgs)
    blockArg = body->addArgument(
        MOGG::DimType::get(ctx, dyn_cast<KGEN::ParamDeclRefAttr>(dim)), loc);
  state.contextArg = body->addArgument(MOGG::ContextType::get(ctx), loc);

  builder.setInsertionPointToEnd(body);

  // `map.iter.opaque` is isolated from above: nested ops reference its
  // shadowed input block args, not `inputs` (its outer operands) directly,
  // so the mapping must be keyed on the block args.
  Block::BlockArgListType inputBlockArgs = getInputBlockArgs();
  for (auto [i, in] : llvm::enumerate(inputs))
    state.mapping.map(inputBlockArgs[i],
                      buildTensor(loc, inputPtrs[i], in.getType(),
                                  MO::IOKind::Input, /*isMut=*/false, state));

  Block &opaqueBody = getBody().front();
  Block::BlockArgListType outputBlockArgs = getOutputBlockArgs();
  for (auto [i, ty] : llvm::enumerate(resultTypes)) {
    Value outputBlockArg = outputBlockArgs[i];
    Value tensor = buildTensor(loc, outputPtrs[i], ty, MO::IOKind::Output,
                               /*isMut=*/true, state);
    state.mapping.map(outputBlockArg, tensor);
  }

  if (failed(state.lowerRegionBody(opaqueBody)))
    return failure();

  replaceAllUsesWith(kernel.getResults());
  erase();
  return success();
}

} // namespace M::MAP
