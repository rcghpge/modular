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
#include "KGEN/KGENDialect/KGENAttrs.h"

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
  Value shape = buildShape(loc, tensorTy.getTypedShapeAttr(), state);
  return MOGG::TensorCreateOp::create(
      state.builder, loc, MOGG::TensorType::get(ctx), ptr, shape, state.device,
      TypeAttr::get(tensorTy.getElementType()),
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

  // Every capture must be one of the kernel's own outputs, already mapped
  // by `IterOpaqueOp::lowerToMOGG`.
  SmallVector<Value> moggCaptures;
  Type outElemType;
  Value outShape;
  for (Value cap : getCaptures()) {
    if (!isa<BlockArgument>(cap) || !state.mapping.contains(cap))
      return emitOpError("MAPToMOGG: epilogue capture is not a kernel output");
    moggCaptures.push_back(state.mapping.lookup(cap));
    auto capTy = cast<MO::TensorType>(cap.getType());
    outShape = buildShape(loc, capTy.getTypedShapeAttr(), state);
    outElemType = capTy.getElementType();
  }

  auto epilogueOp = MOGG::TensorCreateEpilogueOp::create(
      state.builder, loc, MOGG::TensorType::get(ctx), outShape, moggCaptures,
      /*args=*/ValueRange{}, state.device, TypeAttr::get(outElemType),
      state.builder.getBoolAttr(true),
      MO::IOKindAttr::get(ctx, MO::IOKind::Output));

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
  if (failed(state.lowerRegionBody(epiBody)))
    return failure();

  state.builder.setInsertionPointAfter(epilogueOp);
  state.mapping.map(getResult(), epilogueOp.getTensor());
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

  MOGG::CallOpaqueOp::create(state.builder, getLoc(), getCalleeAttr(),
                             callInputs, callOutputs, state.contextArg,
                             getParams());
  return success();
}

// map.iter.foreach.body -> mogg._call.foreach. Builds the loop shape and the
// captured tensors, then lowers the body into the new region via
// `state.lowerRegionBody`.
LogicalResult IterForeachBodyOp::lowerToMOGG(MAPToMOGGState &state) {
  MLIRContext *ctx = getContext();
  Location loc = getLoc();
  auto iterOpaqueKernel = (*this)->getParentOfType<IterOpaqueOp>();
  // `map.iter.opaque` is isolated from above, so nested ops (e.g. this
  // body's own `map.load`s) reference its shadowed input block args, not
  // its outer operands -- the same values `IterOpaqueOp::lowerToMOGG` keyed
  // its mapping on.
  Block::BlockArgListType inputs = iterOpaqueKernel.getInputBlockArgs();
  if (iterOpaqueKernel.getResults().size() != 1)
    return emitOpError(
        "MAPToMOGG: elementwise lowering supports exactly one output");

  Value outputBlockArg = iterOpaqueKernel.getOutputBlockArgs()[0];
  Value outputTensor = state.mapping.lookup(outputBlockArg);

  // The loop's own iteration shape comes from `map.iter.foreach.body`'s
  // `$shape` attribute, not from any I/O tensor's shape (they happen to
  // coincide for a plain, unfused elementwise op like `mo.add`, but not in
  // general -- a broadcast could differ).
  Value loopShape = buildShape(loc, getShape(), state);

  // The foreach driver is one generic primitive, not a per-kernel lookup.
  MOGG::PrimitiveDeclOp foreachPrim = state.registry.lookupPrimitive(
      MOGG::CallForeachRegionOp::getOperationName());
  SmallVector<Value> capturing;
  for (Value in : inputs)
    capturing.push_back(state.mapping.lookup(in));
  capturing.push_back(outputTensor);

  auto callForeach = MOGG::CallForeachRegionOp::create(
      state.builder, loc, FlatSymbolRefAttr::get(ctx, foreachPrim.getSymName()),
      loopShape, state.contextArg, capturing);

  // New block: one arg per capture (same order), then the index. Maps the
  // kernel's inputs, its output block arg, and the old index directly to
  // these new args.
  Value oldIndexArg = getBody().front().getArgument(0);
  Block *foreachBlock = state.builder.createBlock(&callForeach.getBody());
  for (auto [i, in] : llvm::enumerate(inputs))
    state.mapping.map(in,
                      foreachBlock->addArgument(capturing[i].getType(), loc));
  state.mapping.map(outputBlockArg,
                    foreachBlock->addArgument(outputTensor.getType(), loc));
  state.mapping.map(oldIndexArg,
                    foreachBlock->addArgument(oldIndexArg.getType(), loc));

  state.builder.setInsertionPointToStart(foreachBlock);
  if (failed(state.lowerRegionBody(getBody().front())))
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

  // The execution device is where the kernel writes its output, read from the
  // output tensor TYPE (`gpu:0`/`cpu:0`). The `map.iter.opaque` op itself
  // carries no device, so reading it would wrongly fall back to host for GPU
  // ops. This device is recorded on each reconstructed tensor; its ID is
  // later zeroed by `stripDeviceIdsAndDynamicShapes` for dedup, matching the
  // stub's device. Falls back to an input's device for a kernel with no
  // results (e.g. an in-place/mutate-only kernel).
  state.device =
      resultTypes.empty()
          ? MO::getDeviceOrDefaultAny(cast<MO::TensorType>(inputs[0].getType()))
          : MO::getDeviceOrDefaultAny(cast<MO::TensorType>(resultTypes[0]));

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
      for (Attribute dim :
           MO::castFromTensorLike(ty).getTypedShapeAttr().getValues())
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
