//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Lowers MO ops into `map` kernels. Each lowerable MO op with a registered
// Mojo kernel becomes either a `map.iter.opaque` wrapping a `map.call.opaque`
// to that kernel (with a trivial store epilogue per result), or, for
// elementwise kernels, a `map.iter.foreach` expressing the loop body
// explicitly as scalar `map.load` / `map.call.elementwise` / `map.store` ops.
// Kernel selection
// reuses the same `mogg.kernel.decl` machinery as MOToMOGG (MojoRegistry /
// getMojoKernelAdaptor). Contexts, variadics and scalars are intentionally
// NOT materialized here; that is deferred until after fusion.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/Conversion/MOToMAP/MOToMAP.h"
#include "GraphCompiler/Conversion/OperandInference.h"
#include "GraphCompiler/Conversion/ParameterInference.h"
#include "GraphCompiler/MAPDialect/Dialect/MAPOps.h"
#include "GraphCompiler/MODialect/MODialect.h"
#include "GraphCompiler/MODialect/MOInterfaces.h"
#include "GraphCompiler/MODialect/MOOps.h"
#include "GraphCompiler/MODialect/MOUtils.h"
#include "GraphCompiler/MOGGDialect/Analysis/KernelDeclAdaptor.h"
#include "GraphCompiler/MOGGDialect/Analysis/MojoRegistry.h"
#include "GraphCompiler/MOGGDialect/Analysis/Support.h"
#include "GraphCompiler/MOGGDialect/MOGGAttrs.h"
#include "GraphCompiler/MOGGDialect/Support/FusionUtils.h"
#include "GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "mlir/IR/Builders.h"

using namespace M;
using namespace mlir;

namespace M::MO {

#define GEN_PASS_DEF_MOTOMAP
#include "GraphCompiler/Conversion/MOToMAP/MOToMAP.h.inc"

namespace {

bool shouldLower(Operation *op) {
  if (isa<MO::PreservedDuringKernelLowering>(op))
    return false;
  // No MAP form: returns by value. Left for the MOGG conversion.
  if (isa<MO::InvokeShapeFuncOp>(op))
    return false;
  Dialect *dialect = op->getDialect();
  return dialect && isa<MO::MODialect>(dialect);
}

MAP::IterEpilogueOp buildStoreEpilogue(OpBuilder &builder, Location loc,
                                       BlockArgument out) {
  auto outType = cast<MO::TensorType>(out.getType());
  auto indexType = MAP::IndexType::get(builder.getContext(), outType.getRank());

  // `$intermediateTensor` must be passed (as null): dropping it binds the call
  // to the generic `(TypeRange, ValueRange operands)` builder, which does not
  // set `operandSegmentSizes`. `AllocateAccumulatingBuffers` fills it in after
  // fusion.
  auto epilogue =
      MAP::IterEpilogueOp::create(builder, loc, out.getType(), ValueRange{out},
                                  /*intermediateTensor=*/Value());

  OpBuilder::InsertionGuard guard(builder);
  Block *body = builder.createBlock(&epilogue.getBody());
  BlockArgument val = body->addArgument(outType.getElementType(), loc);
  BlockArgument idx = body->addArgument(indexType, loc);
  BlockArgument cap = body->addArgument(out.getType(), loc);
  builder.setInsertionPointToStart(body);
  MAP::StoreOp::create(builder, loc, /*tensor=*/cap, /*index=*/idx,
                       /*value=*/val);
  return epilogue;
}

MAP::IterPrologueOp buildLoadPrologue(OpBuilder &builder, Location loc,
                                      Value in) {
  auto inType = cast<MO::TensorType>(in.getType());
  auto indexType = MAP::IndexType::get(builder.getContext(), inType.getRank());

  auto prologue =
      MAP::IterPrologueOp::create(builder, loc, in.getType(), ValueRange{in});

  OpBuilder::InsertionGuard guard(builder);
  Block *body = builder.createBlock(&prologue.getBody());
  BlockArgument idx = body->addArgument(indexType, loc);
  BlockArgument cap = body->addArgument(in.getType(), loc);
  builder.setInsertionPointToStart(body);
  Value loaded = MAP::LoadOp::create(builder, loc, inType.getElementType(),
                                     /*tensor=*/cap, /*index=*/idx);
  MAP::YieldOp::create(builder, loc, loaded);
  return prologue;
}

// Reads the memory effects of an op about to be replaced by a kernel, so the
// kernel can carry them forward and downstream passes see accurate effects
// instead of the conservative "unknown" default.
std::pair<bool, bool> getReadWriteEffects(Operation *op) {
  auto memEffectOp = dyn_cast<MemoryEffectOpInterface>(op);
  if (!memEffectOp)
    return {false, false};
  return {memEffectOp.hasEffect<MemoryEffects::Read>(),
          memEffectOp.hasEffect<MemoryEffects::Write>()};
}

// Stamp the async launch policy the old MO->MOGG path derives from `srcOp`.
template <typename OpT>
void stampAsyncLaunch(OpT iterOp, Operation *srcOp) {
  iterOp.setIsAsync(!MO::kernelRequiresSyncLaunch(srcOp));
}

LogicalResult lowerToOpaque(Operation *op,
                            const MOGG::KernelDeclAdaptor &adaptor,
                            const MOGG::KernelDeclOperandMapping &mapping) {
  llvm::DenseMap<uint64_t, bool> inputFusible;
  for (const auto &[idx, arg] : mapping.operandMapping)
    inputFusible[idx] = arg.isFusedTensorType();

  llvm::DenseMap<uint64_t, bool> outputFusible;
  for (const auto &[idx, arg] : mapping.resultMapping)
    outputFusible[idx] = arg.isFusedTensorType();

  OpBuilder builder(op);
  Location loc = op->getLoc();

  // A mutating op's chain, when present, is always its own last operand and
  // last result (the codebase-wide convention -- see e.g. mo.mutable.store).
  // `map.iter.opaque` threads it through a dedicated `$inputChain` operand and
  // a trailing result instead of an ordinary, ambiguous `inputs`/`results`
  // entry -- and, like `map.iter.foreach`/`mogg._kernel`, keeps it out of the
  // body entirely (no block argument, never a `map.call.opaque` operand): it
  // is pure sequencing plumbing no opaque kernel's `execute` reads.
  ValueRange operands = op->getOperands();
  bool hasInputChain =
      !operands.empty() && isa<MO::ChainType>(operands.back().getType());
  Value inputChain = hasInputChain ? operands.back() : Value();
  ValueRange nonChainOperands = hasInputChain ? operands.drop_back() : operands;

  TypeRange resultTypes = op->getResultTypes();

  auto opaque = MAP::IterOpaqueOp::create(builder, loc, resultTypes,
                                          nonChainOperands, inputChain,
                                          /*fusionInfo=*/Attribute());
  // A collective executes on every device it reduces across, so the singular
  // getExecutionDevice() behind getDeviceOrDefaultAny asserts on one. Take the
  // first, as MOToMOGGLowering::getDevice() does -- the label only picks host
  // or GPU codegen. TODO: carry all of them, see GEX-2278.
  DeviceRefAttr device;
  if (auto hasDevice = dyn_cast<MO::MOHasDeviceInterface>(op)) {
    SmallVector<DeviceRefAttr> devices = hasDevice.getExecutionDevices();
    if (!devices.empty())
      device = devices.front();
  }
  opaque.setDeviceRef(device ? device : MO::getDeviceOrDefaultAny(op));
  opaque.setFusionInfoAttr(
      MOGG::FusionInfoAttr::get(op->getContext(), MOGG::createFusionLeaf(op)));
  auto [hasReadEffect, hasWriteEffect] = getReadWriteEffects(op);
  opaque.setHasReadEffect(hasReadEffect);
  opaque.setHasWriteEffect(hasWriteEffect);
  stampAsyncLaunch(opaque, op);

  // Body block args shadow the non-chain inputs then the non-chain outputs.
  // The chain (in and out) is pure MGP sequencing threaded through the op's
  // dedicated `$inputChain` operand and trailing chain result; it is never a
  // body block arg and never reaches the wrapped `map.call.opaque` -- no opaque
  // kernel's `execute` takes a chain, so the body has nothing to do with it.
  Block *body = builder.createBlock(&opaque.getBody());
  SmallVector<BlockArgument> ins;
  for (Value operand : nonChainOperands)
    ins.push_back(body->addArgument(operand.getType(), loc));
  SmallVector<BlockArgument> outs;
  Type opaqueResultType;
  for (Type resultType : resultTypes) {
    if (isa<MO::ChainType>(resultType))
      continue;
    // A kernel allocates its `!mo.opaque` result internally and returns it, so
    // there is no destination to fill and no block argument for it: the body's
    // `map.yield` carries it out instead.
    if (isa<MO::OpaqueType>(resultType)) {
      opaqueResultType = resultType;
      continue;
    }
    outs.push_back(body->addArgument(resultType, loc));
  }

  builder.setInsertionPointToStart(body);
  SmallVector<Value> callInputs;
  for (auto [idx, operand] : llvm::enumerate(ins)) {
    if (!inputFusible.lookup(idx)) {
      callInputs.push_back(operand);
      continue;
    }
    if (!isa<MO::TensorType>(operand.getType()))
      return op->emitOpError(
          "MOToMAP: input fusion is only supported for tensor operands");
    callInputs.push_back(buildLoadPrologue(builder, loc, operand).getResult());
  }

  SmallVector<Value> callOutputs;
  for (auto [idx, out] : llvm::enumerate(outs)) {
    if (!outputFusible.lookup(idx)) {
      callOutputs.push_back(out);
      continue;
    }
    if (!isa<MO::TensorType>(out.getType()))
      return op->emitOpError(
          "MOToMAP: output fusion is only supported for tensor results");
    callOutputs.push_back(buildStoreEpilogue(builder, loc, out).getResult());
  }

  // A kernel parameterized only through an opaque operand or result (e.g.
  // `MakeSimdPair.execute[P0, P1]() -> SIMDPair[P0, P1]`) names those
  // parameters nowhere in the op's own dictionary, so resolve them from the
  // opaque types the way MO->MOGG does.
  DictionaryAttr params = inferParameters(op, builder);
  if (adaptor.executeFunction && adaptor.executeFunction->attr) {
    // `inferOpaqueParameters` appends to `withOpaque`, which starts as a copy
    // of what is already inferred.
    SmallVector<NamedAttribute> withOpaque(params.begin(), params.end());
    MOGG::inferOpaqueParameters(withOpaque, op, *adaptor.executeFunction,
                                adaptor.executeFunction->attr.getParams());
    params = builder.getDictionaryAttr(withOpaque);
  }

  builder.setInsertionPointToEnd(body);
  auto callOp = MAP::CallOpaqueOp::create(
      builder, loc,
      opaqueResultType ? TypeRange{opaqueResultType} : TypeRange{},
      FlatSymbolRefAttr::get(op->getContext(),
                             adaptor.fullyResolvedName.strref()),
      adaptor.registeredOpName, params,
      /*inputs=*/callInputs,
      /*outputs=*/callOutputs);
  // Carry the source op's same-variadic-operand-size property forward: it is
  // needed later to split the call's flat operands back into equal-length
  // variadic groups (e.g. a bundled collective's N peers and N signal buffers).
  callOp.setHasSameVariadicOperandSize(
      isa<MO::SameVariadicOperandSizeInterface>(op));

  // A DPS output needs no terminator -- the output block args themselves are
  // the op's results. A by-value opaque does: `map.yield` is what connects the
  // call's return value to the enclosing op's result.
  if (opaqueResultType)
    MAP::YieldOp::create(builder, loc, callOp.getOpaqueResult());

  op->replaceAllUsesWith(opaque.getResults());
  op->erase();
  return success();
}

// Lowers an elementwise MO op to a `map.iter.foreach` expressing its loop
// body explicitly: one `map.load` per input, a `map.call.elementwise` call
// to the kernel's scalar function, and a `map.store` into the output, all
// indexed by the op's `map.index<N>` block argument.
LogicalResult
lowerElementwiseToForeach(Operation *op,
                          const MOGG::KernelDeclAdaptor &adaptor) {
  // Multi-output elementwise kernels are not supported yet.
  if (op->getNumResults() != 1)
    return op->emitOpError(
        "MOToMAP: elementwise op must have exactly one result");

  auto resultTensorTy = dyn_cast<MO::TensorType>(op->getResult(0).getType());
  if (!resultTensorTy)
    return op->emitOpError("MOToMAP: elementwise op result must be !mo.tensor");

  MLIRContext *ctx = op->getContext();
  Location loc = op->getLoc();
  OpBuilder builder(op);

  MOSH::ShapeAttr shapeAttr = resultTensorTy.getShape();
  unsigned rank = static_cast<unsigned>(shapeAttr.getRank());
  Type elemTy = resultTensorTy.getElementType();

  auto foreachOp = MAP::IterForeachOp::create(
      builder, loc,
      /*resultTypes=*/SmallVector<Type>{resultTensorTy},
      /*shape=*/shapeAttr,
      /*inputs=*/op->getOperands(),
      /*chain=*/Value(),
      /*fusionInfo=*/
      MOGG::FusionInfoAttr::get(ctx, MOGG::createFusionLeaf(op)));
  stampAsyncLaunch(foreachOp, op);

  // Body block args: [input tensors..., output tensor, map.index<rank>].
  Block *body = &foreachOp.getBody().emplaceBlock();
  for (Value operand : op->getOperands())
    body->addArgument(operand.getType(), loc);
  body->addArgument(resultTensorTy, loc);
  body->addArgument(MAP::IndexType::get(ctx, rank), loc);

  OpBuilder bodyBuilder = OpBuilder::atBlockEnd(body);
  BlockArgument idxArg = body->getArguments().back();

  // map.load for each input.
  SmallVector<Value> scalarLoads;
  for (auto [i, operand] : llvm::enumerate(op->getOperands())) {
    BlockArgument inputArg = body->getArgument(i);
    auto inputTensorTy = cast<MO::TensorType>(operand.getType());
    Value loaded = MAP::LoadOp::create(
        bodyBuilder, loc, inputTensorTy.getElementType(), inputArg, idxArg);
    scalarLoads.push_back(loaded);
  }

  // map.call.elementwise — kernel symbol from the fully resolved Mojo name.
  auto kernelSymbol =
      FlatSymbolRefAttr::get(ctx, adaptor.fullyResolvedName.strref());
  DictionaryAttr params = inferParameters(op, builder);
  if (params.empty())
    params = nullptr;
  Value scalar = MAP::CallElementwiseOp::create(
      bodyBuilder, loc, elemTy, kernelSymbol, params, idxArg, scalarLoads);

  // map.store into the output block argument.
  BlockArgument outputArg =
      body->getArgument(static_cast<unsigned>(op->getNumOperands()));
  MAP::StoreOp::create(bodyBuilder, loc, outputArg, idxArg, scalar);

  op->replaceAllUsesWith(foreachOp.getResults());
  op->erase();
  return success();
}

// Lowers mo.mutable.store to a map.iter.foreach that writes directly into
// its buffer operand (folded into `inputs`, per map.iter.foreach's "operands
// shadowed as block args" convention) and threads a chain in and out. Its
// elementwise function is a scalar identity (see gather_scatter.mojo's
// MutableStore); the actual in-place write happens via `map.store` into the
// buffer instead of a materialized tensor output.
//
// mo.mutable.store is the one op MOGG itself special-cases by identity for
// this (MOToMOGGLowering.cpp's `isa<MO::MutableStoreOp>(moOp)`, TODO
// GEX-2360 acknowledges there's no generic mechanism to generalize) --
// mirrored here rather than generalized, for the same reason.
LogicalResult
lowerMutableStoreToForeach(MO::MutableStoreOp op,
                           const MOGG::KernelDeclAdaptor &adaptor) {
  MLIRContext *ctx = op->getContext();
  Location loc = op->getLoc();
  OpBuilder builder(op);

  MO::TensorType bufferAsTensorTy = op.getInBuffer().getType().toTensorType();
  MOSH::ShapeAttr shapeAttr = bufferAsTensorTy.getShape();
  unsigned rank = static_cast<unsigned>(shapeAttr.getRank());
  Type elemTy = bufferAsTensorTy.getElementType();

  auto [hasReadEffect, hasWriteEffect] = getReadWriteEffects(op);

  auto foreachOp = MAP::IterForeachOp::create(
      builder, loc,
      /*resultTypes=*/SmallVector<Type>{MO::ChainType::get(ctx)},
      /*shape=*/shapeAttr,
      /*inputs=*/ValueRange{op.getInTensor(), op.getInBuffer()},
      /*chain=*/op.getInChain(),
      /*fusionInfo=*/
      MOGG::FusionInfoAttr::get(ctx, MOGG::createFusionLeaf(op)),
      /*hasReadEffect=*/hasReadEffect,
      /*hasWriteEffect=*/hasWriteEffect);
  stampAsyncLaunch(foreachOp, op);

  // Body block args: [tensor value, buffer, map.index<rank>]. No "outputs"
  // block arg at all -- the buffer input itself is the store target, and
  // the sole result (the output chain) has no body block argument.
  Block *body = &foreachOp.getBody().emplaceBlock();
  BlockArgument valueArg = body->addArgument(op.getInTensor().getType(), loc);
  BlockArgument bufferArg = body->addArgument(op.getInBuffer().getType(), loc);
  BlockArgument idxArg = body->addArgument(MAP::IndexType::get(ctx, rank), loc);

  OpBuilder bodyBuilder = OpBuilder::atBlockEnd(body);
  Value loaded =
      MAP::LoadOp::create(bodyBuilder, loc, elemTy, valueArg, idxArg);

  auto kernelSymbol =
      FlatSymbolRefAttr::get(ctx, adaptor.fullyResolvedName.strref());
  DictionaryAttr params = inferParameters(op, builder);
  if (params.empty())
    params = nullptr;
  Value scalar =
      MAP::CallElementwiseOp::create(bodyBuilder, loc, elemTy, kernelSymbol,
                                     params, idxArg, ValueRange{loaded});

  MAP::StoreOp::create(bodyBuilder, loc, bufferArg, idxArg, scalar);

  op->replaceAllUsesWith(foreachOp.getResults());
  op->erase();
  return success();
}

// Lowers mo.slice to a standalone (unfused) map.iter.foreach over the
// *output* shape (the "pull" model: each output index computes the input
// index it reads from, rather than each input index computing where it
// writes): the body computes the corresponding input index via
// map.index.slice, loads, and stores unchanged. One kernel per mo.slice,
// same as lowerElementwiseToForeach's starting point.
LogicalResult lowerSliceToForeach(MO::SliceOp op) {
  MLIRContext *ctx = op->getContext();
  Location loc = op->getLoc();
  OpBuilder builder(op);

  auto inputTy = cast<MO::TensorType>(op.getInput().getType());
  auto outputTy = cast<MO::TensorType>(op.getResult().getType());
  MOSH::ShapeAttr inShape = inputTy.getShape();
  MOSH::ShapeAttr outShape = outputTy.getShape();
  if (inShape.getRank() != outShape.getRank())
    return op->emitOpError("MOToMAP: slice must preserve rank");
  int64_t rank = outShape.getRank();

  // A bound is a compile-time constant only if the *whole* start/stop/step
  // tensor operand is one: MO has no per-dimension constant tracking finer
  // than "is this whole tensor a mo.constant".
  auto boundsOrDynamic = [&](Value v) {
    if (auto constVec = MO::tryGetConstantIntVector(cast<MO::TensorValue>(v)))
      return mlir::DenseI64ArrayAttr::get(ctx, *constVec);
    return mlir::DenseI64ArrayAttr::get(
        ctx, SmallVector<int64_t>(rank, ShapedType::kDynamic));
  };

  SmallVector<int64_t> inDims = inShape.getIntDims();
  SmallVector<int64_t> starts(boundsOrDynamic(op.getStart()).asArrayRef());
  SmallVector<int64_t> stops(boundsOrDynamic(op.getStop()).asArrayRef());
  SmallVector<int64_t> steps(boundsOrDynamic(op.getStep()).asArrayRef());

  // The spec stores resolved bounds, but MO accepts Python-style negative
  // indices and out-of-range values and normalizes them in its own shape
  // function. Do the same here, matching materializeSliceShape's clamp.
  for (int64_t i = 0; i < rank; ++i) {
    if (ShapedType::isDynamic(inDims[i]) || ShapedType::isDynamic(steps[i]) ||
        steps[i] == 0)
      continue;
    int64_t dim = inDims[i];
    // A reverse slice walks down to -1, so it clamps one below the forward
    // range.
    int64_t lo = steps[i] > 0 ? 0 : -1;
    int64_t hi = steps[i] > 0 ? dim : dim - 1;
    auto normalize = [&](int64_t v) {
      if (ShapedType::isDynamic(v))
        return v;
      return std::clamp<int64_t>(v < 0 ? v + dim : v, lo, hi);
    };
    starts[i] = normalize(starts[i]);
    stops[i] = normalize(stops[i]);
  }

  // Verify before constructing: the plain `get` asserts on a rejected spec,
  // which turns an unsupported slice into a crash instead of a diagnostic.
  auto emitSliceError = [&] { return op->emitOpError("MOToMAP: "); };
  if (failed(MAP::SliceSpecAttr::verify(
          emitSliceError, MAP::ViewSignatureAttr::get(ctx, inShape, outShape),
          mlir::DenseI64ArrayAttr::get(ctx, starts),
          mlir::DenseI64ArrayAttr::get(ctx, stops),
          mlir::DenseI64ArrayAttr::get(ctx, steps))))
    return failure();

  auto spec =
      MAP::SliceSpecAttr::get(ctx, inShape, outShape, starts, stops, steps);

  auto foreachOp = MAP::IterForeachOp::create(
      builder, loc,
      /*resultTypes=*/SmallVector<Type>{outputTy},
      /*shape=*/outShape,
      /*inputs=*/op->getOperands(),
      /*chain=*/Value(),
      /*fusionInfo=*/
      MOGG::FusionInfoAttr::get(ctx, MOGG::createFusionLeaf(op)));
  stampAsyncLaunch(foreachOp, op);

  // Body block args: [input, start, stop, step, output, map.index<rank>].
  Block *body = &foreachOp.getBody().emplaceBlock();
  for (Value operand : op->getOperands())
    body->addArgument(operand.getType(), loc);
  body->addArgument(outputTy, loc);
  BlockArgument idxArg = body->addArgument(MAP::IndexType::get(ctx, rank), loc);

  OpBuilder bodyBuilder = OpBuilder::atBlockEnd(body);
  BlockArgument inputArg = body->getArgument(0);
  BlockArgument startsArg = body->getArgument(1);
  BlockArgument stopsArg = body->getArgument(2);
  BlockArgument stepsArg = body->getArgument(3);
  BlockArgument outputArg = body->getArgument(4);

  Value innerIdx = MAP::SliceIndexOp::create(
      bodyBuilder, loc, MAP::IndexType::get(ctx, rank), spec, idxArg, startsArg,
      stopsArg, stepsArg);
  Value loaded = MAP::LoadOp::create(
      bodyBuilder, loc, outputTy.getElementType(), inputArg, innerIdx);
  MAP::StoreOp::create(bodyBuilder, loc, outputArg, idxArg, loaded);

  op->replaceAllUsesWith(foreachOp.getResults());
  op->erase();
  return success();
}

// Lowers mo.static.broadcast_to to a standalone (unfused) map.iter.foreach
// over the *output* shape (same pull model as lowerSliceToForeach): the body
// computes the input index via map.index.broadcast, loads, and stores
// unchanged. Unlike slice, broadcast can add rank (numpy semantics), so the
// loop's own index (output rank) and the projected load index (input rank)
// can differ.
LogicalResult lowerBroadcastToForeach(Static::BroadcastToOp op) {
  MLIRContext *ctx = op->getContext();
  Location loc = op->getLoc();
  OpBuilder builder(op);

  auto inputTy = cast<MO::TensorType>(op.getInput().getType());
  auto outputTy = cast<MO::TensorType>(op.getResult().getType());
  MOSH::ShapeAttr inShape = inputTy.getShape();
  MOSH::ShapeAttr outShape = outputTy.getShape();
  if (inShape.getRank() > outShape.getRank())
    return op->emitOpError("MOToMAP: broadcast cannot reduce rank");
  int64_t inRank = inShape.getRank();
  int64_t outRank = outShape.getRank();

  auto spec = MAP::BroadcastSpecAttr::get(
      ctx, MAP::ViewSignatureAttr::get(ctx, inShape, outShape));

  auto foreachOp = MAP::IterForeachOp::create(
      builder, loc,
      /*resultTypes=*/SmallVector<Type>{outputTy},
      /*shape=*/outShape,
      /*inputs=*/op->getOperands(),
      /*chain=*/Value(),
      /*fusionInfo=*/
      MOGG::FusionInfoAttr::get(ctx, MOGG::createFusionLeaf(op)));
  stampAsyncLaunch(foreachOp, op);

  // Body block args: [input, output, map.index<outRank>].
  Block *body = &foreachOp.getBody().emplaceBlock();
  for (Value operand : op->getOperands())
    body->addArgument(operand.getType(), loc);
  body->addArgument(outputTy, loc);
  BlockArgument idxArg =
      body->addArgument(MAP::IndexType::get(ctx, outRank), loc);

  OpBuilder bodyBuilder = OpBuilder::atBlockEnd(body);
  BlockArgument inputArg = body->getArgument(0);
  BlockArgument outputArg = body->getArgument(1);

  Value innerIdx = MAP::BroadcastIndexOp::create(
      bodyBuilder, loc, MAP::IndexType::get(ctx, inRank), spec, idxArg);
  Value loaded = MAP::LoadOp::create(
      bodyBuilder, loc, outputTy.getElementType(), inputArg, innerIdx);
  MAP::StoreOp::create(bodyBuilder, loc, outputArg, idxArg, loaded);

  op->replaceAllUsesWith(foreachOp.getResults());
  op->erase();
  return success();
}

// Lowers mo.static.reshape to a standalone (unfused) map.iter.foreach over
// the *output* shape (same pull model as lowerSliceToForeach): the body
// computes the input index via map.index.reshape, loads, and stores
// unchanged. Unlike slice/broadcast/transpose, rank can change in either
// direction -- reshape only preserves the total element count -- so the
// loop's own index (output rank) and the projected load index (input rank)
// can differ arbitrarily.
LogicalResult lowerReshapeToForeach(MO::Static::ReshapeOp op) {
  MLIRContext *ctx = op->getContext();
  Location loc = op->getLoc();
  OpBuilder builder(op);

  auto inputTy = cast<MO::TensorType>(op.getInput().getType());
  auto outputTy = cast<MO::TensorType>(op.getResult().getType());
  MOSH::ShapeAttr inShape = inputTy.getShape();
  MOSH::ShapeAttr outShape = outputTy.getShape();
  int64_t inRank = inShape.getRank();
  int64_t outRank = outShape.getRank();

  auto spec = MAP::ReshapeSpecAttr::get(
      ctx, MAP::ViewSignatureAttr::get(ctx, inShape, outShape));

  auto foreachOp = MAP::IterForeachOp::create(
      builder, loc,
      /*resultTypes=*/SmallVector<Type>{outputTy},
      /*shape=*/outShape,
      /*inputs=*/op->getOperands(),
      /*chain=*/Value(),
      /*fusionInfo=*/
      MOGG::FusionInfoAttr::get(ctx, MOGG::createFusionLeaf(op)));
  stampAsyncLaunch(foreachOp, op);

  // Body block args: [input, output, map.index<outRank>].
  Block *body = &foreachOp.getBody().emplaceBlock();
  for (Value operand : op->getOperands())
    body->addArgument(operand.getType(), loc);
  body->addArgument(outputTy, loc);
  BlockArgument idxArg =
      body->addArgument(MAP::IndexType::get(ctx, outRank), loc);

  OpBuilder bodyBuilder = OpBuilder::atBlockEnd(body);
  BlockArgument inputArg = body->getArgument(0);
  BlockArgument outputArg = body->getArgument(1);

  Value innerIdx = MAP::ReshapeIndexOp::create(
      bodyBuilder, loc, MAP::IndexType::get(ctx, inRank), spec, idxArg);
  Value loaded = MAP::LoadOp::create(
      bodyBuilder, loc, outputTy.getElementType(), inputArg, innerIdx);
  MAP::StoreOp::create(bodyBuilder, loc, outputArg, idxArg, loaded);

  op->replaceAllUsesWith(foreachOp.getResults());
  op->erase();
  return success();
}

// Lowers mo.transpose to a standalone (unfused) map.iter.foreach over the
// *output* shape (same pull model as lowerSliceToForeach): the body computes
// the input index via map.index.transpose, loads, and stores unchanged. The
// perm rides in as a runtime operand, exactly like slice's starts/stops/steps:
// where the whole perm tensor is a compile-time constant its values are also
// baked into `spec` (so a fully-static transpose keeps a static view and never
// reads the operand), and where it isn't -- the eager interpreter's
// one-graph-per-op transpose -- `spec` marks every entry dynamic and the
// operand supplies the permutation at runtime.
LogicalResult lowerTransposeToForeach(MO::TransposeOp op) {
  MLIRContext *ctx = op->getContext();
  Location loc = op->getLoc();
  OpBuilder builder(op);

  auto inputTy = cast<MO::TensorType>(op.getInput().getType());
  auto outputTy = cast<MO::TensorType>(op.getResult().getType());
  MOSH::ShapeAttr inShape = inputTy.getShape();
  MOSH::ShapeAttr outShape = outputTy.getShape();
  if (inShape.getRank() != outShape.getRank())
    return op->emitOpError("MOToMAP: transpose must preserve rank");
  int64_t rank = outShape.getRank();

  // A perm is compile-time only if the *whole* perm tensor operand is a
  // constant -- MO has no per-dimension constant tracking -- so it is
  // all-or-nothing, the same rule slice's bounds follow. Default to the
  // all-dynamic spec, then override when the operand is a constant.
  auto permAttr = mlir::DenseI64ArrayAttr::get(
      ctx, SmallVector<int64_t>(rank, ShapedType::kDynamic));
  if (auto permVec =
          MO::tryGetConstantIntVector(cast<MO::TensorValue>(op.getPerm())))
    permAttr = mlir::DenseI64ArrayAttr::get(ctx, *permVec);

  auto spec = MAP::TransposeSpecAttr::get(
      ctx, MAP::ViewSignatureAttr::get(ctx, inShape, outShape), permAttr);

  auto foreachOp = MAP::IterForeachOp::create(
      builder, loc,
      /*resultTypes=*/SmallVector<Type>{outputTy},
      /*shape=*/outShape,
      /*inputs=*/op->getOperands(),
      /*chain=*/Value(),
      /*fusionInfo=*/
      MOGG::FusionInfoAttr::get(ctx, MOGG::createFusionLeaf(op)));
  stampAsyncLaunch(foreachOp, op);

  // Body block args: [input, perm, output, map.index<rank>].
  Block *body = &foreachOp.getBody().emplaceBlock();
  for (Value operand : op->getOperands())
    body->addArgument(operand.getType(), loc);
  body->addArgument(outputTy, loc);
  BlockArgument idxArg = body->addArgument(MAP::IndexType::get(ctx, rank), loc);

  OpBuilder bodyBuilder = OpBuilder::atBlockEnd(body);
  BlockArgument inputArg = body->getArgument(0);
  BlockArgument permArg = body->getArgument(1);
  BlockArgument outputArg = body->getArgument(2);

  Value innerIdx = MAP::TransposeIndexOp::create(
      bodyBuilder, loc, MAP::IndexType::get(ctx, rank), spec, idxArg, permArg);
  Value loaded = MAP::LoadOp::create(
      bodyBuilder, loc, outputTy.getElementType(), inputArg, innerIdx);
  MAP::StoreOp::create(bodyBuilder, loc, outputArg, idxArg, loaded);

  op->replaceAllUsesWith(foreachOp.getResults());
  op->erase();
  return success();
}

LogicalResult lowerOp(Operation *op, MOGG::MojoRegistry &registry) {
  ErrorOr<MOGG::KernelDeclAdaptor> adaptor =
      MOGG::getMojoKernelAdaptor(registry, op);
  if (adaptor.isError())
    return success();

  if (auto mutableStoreOp = dyn_cast<MO::MutableStoreOp>(op))
    return lowerMutableStoreToForeach(mutableStoreOp, *adaptor);

  // Elementwise kernels don't have an executeFunction (only
  // elementwiseFunction), so this must be checked before the executeFunction
  // gate below.
  if (adaptor->isElementwise())
    return lowerElementwiseToForeach(op, *adaptor);

  if (!adaptor->executeFunction)
    return op->emitOpError("MOToMAP: kernel has no execute function");

  ErrorOr<MOGG::KernelDeclOperandMapping> mapping =
      MOGG::mapMOOpToKernelImplementation(op, *adaptor->executeFunction);
  if (mapping.isError())
    return op->emitOpError("MOToMAP: failed to map operands to kernel");

  if (isa<MO::InvokeShapeFuncOp>(op))
    return op->emitOpError(
        "MOToMAP: shape-function kernels are not yet supported");
  else if (auto sliceOp = dyn_cast<MO::SliceOp>(op))
    return lowerSliceToForeach(sliceOp);
  else if (auto broadcastOp = dyn_cast<MO::Static::BroadcastToOp>(op))
    return lowerBroadcastToForeach(broadcastOp);
  else if (auto reshapeOp = dyn_cast<MO::Static::ReshapeOp>(op))
    return lowerReshapeToForeach(reshapeOp);
  else if (auto transposeOp = dyn_cast<MO::TransposeOp>(op))
    return lowerTransposeToForeach(transposeOp);
  else if (adaptor->isViewKernel())
    return op->emitOpError("MOToMAP: view kernels are not yet supported");
  else
    return lowerToOpaque(op, *adaptor, *mapping);
}

struct MOToMAPPass : public M::MO::impl::MOToMAPBase<MOToMAPPass> {
  using Base::Base;

  void runOnOperation() override {
    if (!useAdvancedFusion())
      return;

    MOGG::MojoRegistry registry(getOperation());

    SmallVector<Operation *> worklist;
    getOperation()->walk([&](Operation *op) {
      if (shouldLower(op))
        worklist.push_back(op);
    });

    for (Operation *op : worklist)
      if (failed(lowerOp(op, registry)))
        return signalPassFailure();
  }
};

} // namespace

} // namespace M::MO
