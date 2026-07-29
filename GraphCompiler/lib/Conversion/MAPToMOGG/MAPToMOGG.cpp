//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Lowers `map` dialect kernels into `mogg` codegen ops. A `map.iter.opaque`
// becomes a `mogg._kernel`: its `!mo.tensor` inputs stay operands and its
// outputs stay results, but inside the body each tensor is reconstructed from a
// `!mogg.pointer_` block argument via `mogg._shape.create` /
// `mogg._tensor.create`, and the `map.call.opaque` becomes a
// `mogg._call.opaque` over those tensors.
//
// A fused (store-epilogue) output — a `map.iter.epilogue` feeding the call's
// outputs — becomes a `mogg._tensor.create.epilogue` holding the store
// directly in its own body.
//
// Dynamic shapes are handled: the kernel gains one `!mogg.dim_` block argument
// per unique dynamic dimension (deduped across inputs and outputs), placed
// between the pointers and the context, and each `mogg._shape.create` is fed
// the dims for its own tensor. Epilogue bodies that are not a single
// `map.store` are still deferred (the pass emits an error on them).
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/Conversion/MAPToMOGG/MAPToMOGG.h"
#include "GraphCompiler/Conversion/MOToMAP/MOToMAP.h"
#include "GraphCompiler/MAPDialect/Dialect/MAPDialect.h"
#include "GraphCompiler/MAPDialect/Dialect/MAPOps.h"
#include "GraphCompiler/MODialect/MODevices.h"
#include "GraphCompiler/MODialect/MODialect.h"
#include "GraphCompiler/MODialect/MOEnums.h"
#include "GraphCompiler/MODialect/MOTypes.h"
#include "GraphCompiler/MOGGDialect/MOGGDialect.h"
#include "GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GraphCompiler/MOGGDialect/MOGGTypes.h"
#include "GraphCompiler/MOGGDialect/Support/FusionUtils.h"
#include "GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "GraphCompiler/MOSHDialect/MOSHDialect.h"
#include "KGEN/KGENDialect/KGENAttrs.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/MapVector.h"

using namespace M;
using namespace mlir;

namespace M::MAP {

#define GEN_PASS_DEF_MAPTOMOGG
#include "GraphCompiler/Conversion/MAPToMOGG/MAPToMOGG.h.inc"

namespace {

// A dimension is dynamic unless it is a concrete `KGEN::SIMDAttr`.
bool isDynamicDim(Attribute dim) { return !isa<KGEN::SIMDAttr>(dim); }

// Reconstructs an `!mogg.tensor_` for `tensorType` from the `!mogg.pointer_`
// block argument `ptr`, via `mogg._shape.create` + `mogg._tensor.create`.
// `io`/`isMut` distinguish immutable inputs from mutable outputs. `device` is
// recorded on the tensor so emit can pick alignment locally. `dimArgs` are the
// kernel's per-unique-dim `!mogg.dim_` block args; `dimIndex` maps a dynamic
// dimension attribute to its position in `dimArgs`. When non-null, `*shapeOut`
// receives the `!mogg.shape` SSA value (an epilogue reuses its output's shape).
Value buildTensor(OpBuilder &builder, Location loc, Value ptr, Type tensorType,
                  MO::IOKind io, bool isMut, DeviceRefAttr device,
                  ArrayRef<BlockArgument> dimArgs,
                  const llvm::MapVector<Attribute, unsigned> &dimIndex,
                  Value *shapeOut = nullptr) {
  auto tensorTy = cast<MO::TensorType>(tensorType);
  MLIRContext *ctx = builder.getContext();
  MOSH::ShapeAttr shapeAttr = tensorTy.getTypedShapeAttr();

  // Canonicalize every dynamic dim in the shape ATTR to one uniform `dyn`
  // placeholder so shapes carrying different symbols (e.g. `[M, K]` vs `[N,
  // K]`) dedup to `[dyn, dyn]` / `[dyn, K]`. The `!mogg.dim_` OPERANDS below
  // stay the real per-dim handles (they carry runtime values); only the attr is
  // folded.
  TypedAttr canonicalDyn = KGEN::ParamDeclRefAttr::get(
      StringAttr::get(ctx, "dyn"), MOSH::ShapeAttr::getDimType(ctx));
  SmallVector<Value> dims;
  SmallVector<TypedAttr> canonicalDims;
  for (TypedAttr dim : shapeAttr.getValues()) {
    if (isDynamicDim(dim)) {
      dims.push_back(dimArgs[dimIndex.lookup(dim)]);
      canonicalDims.push_back(canonicalDyn);
    } else {
      canonicalDims.push_back(dim);
    }
  }
  auto canonicalShapeAttr =
      MOSH::ShapeAttr::get(canonicalDims, shapeAttr.getType());

  Value shape = MOGG::ShapeCreateOp::create(
      builder, loc, MOGG::ShapeType::get(ctx), dims, canonicalShapeAttr);
  if (shapeOut)
    *shapeOut = shape;
  return MOGG::TensorCreateOp::create(
      builder, loc, MOGG::TensorType::get(ctx), ptr, shape, device,
      TypeAttr::get(tensorTy.getElementType()), builder.getBoolAttr(isMut),
      MO::IOKindAttr::get(ctx, io));
}

// Translates a store `map.iter.epilogue` into a `mogg._tensor.create.epilogue`
// holding the `mogg._tensor.store` directly in its own body, and returns the
// epilogue `!mogg.tensor_` (what the kernel call writes into). `outputTensors`
// maps an opaque output block-arg number to its reconstructed tensor.
FailureOr<Value> buildEpilogueTensor(OpBuilder &builder, Location loc,
                                     MAP::IterEpilogueOp epilogue,
                                     ArrayRef<Value> outputTensors,
                                     ArrayRef<Value> outputShapes,
                                     DeviceRefAttr device) {
  MLIRContext *ctx = builder.getContext();
  Block &epiBody = epilogue.getBody().front();
  Block::BlockArgListType epiArgs = epiBody.getArguments();
  size_t numCaptures = epilogue.getCaptures().size();
  if (epiArgs.size() < numCaptures)
    return failure();
  // Leading block args are the callback params (value, index); the rest mirror
  // the captured operands.
  size_t numParams = epiArgs.size() - numCaptures;

  // The mogg captures are the reconstructed output tensors for the captured
  // opaque output block args. The epilogue tensor reuses the store target's
  // `!mogg.shape` so emit no longer reaches into the captured tensors for it.
  SmallVector<Value> moggCaptures;
  Type outElemType;
  Value outShape;
  for (Value cap : epilogue.getCaptures()) {
    auto barg = dyn_cast<BlockArgument>(cap);
    if (!barg)
      return failure();
    moggCaptures.push_back(outputTensors[barg.getArgNumber()]);
    outShape = outputShapes[barg.getArgNumber()];
    outElemType = cast<MO::TensorType>(cap.getType()).getElementType();
  }

  auto epilogueOp = MOGG::TensorCreateEpilogueOp::create(
      builder, loc, MOGG::TensorType::get(ctx), outShape, moggCaptures,
      /*args=*/ValueRange{}, device, TypeAttr::get(outElemType),
      builder.getBoolAttr(true), MO::IOKindAttr::get(ctx, MO::IOKind::Output));

  // Epilogue body: the epilogue's leading params, then one fresh block arg
  // per capture (typed as the reconstructed `!mogg.tensor_`, not the original
  // `!mo.tensor` capture, since the body operates on the mogg-side tensor).
  Block *body = builder.createBlock(&epilogueOp.getBody());
  IRMapping map;
  for (size_t i = 0; i < numParams; ++i) {
    BlockArgument p =
        body->addArgument(epiArgs[i].getType(), epiArgs[i].getLoc());
    map.map(epiArgs[i], p);
  }
  for (size_t j = 0; j < numCaptures; ++j) {
    BlockArgument capArg = body->addArgument(moggCaptures[j].getType(),
                                             epiArgs[numParams + j].getLoc());
    map.map(epiArgs[numParams + j], capArg);
  }

  // Only a single `map.store` epilogue body is supported for now.
  builder.setInsertionPointToEnd(body);
  for (Operation &op : epiBody) {
    auto store = dyn_cast<MAP::StoreOp>(op);
    if (!store)
      return failure();
    MOGG::TensorStoreOp::create(builder, loc, map.lookup(store.getTensor()),
                                map.lookup(store.getIndex()),
                                map.lookup(store.getValue()));
  }

  builder.setInsertionPointAfter(epilogueOp);
  return Value(epilogueOp.getTensor());
}

LogicalResult lowerOpaque(MAP::IterOpaqueOp iterOpaqueKernel) {
  MLIRContext *ctx = iterOpaqueKernel.getContext();
  Location loc = iterOpaqueKernel.getLoc();
  Block &iterOpaqueBody = iterOpaqueKernel.getBody().front();

  MAP::CallOpaqueOp call = nullptr;
  for (Operation &bodyOp : iterOpaqueBody)
    if (auto c = dyn_cast<MAP::CallOpaqueOp>(bodyOp))
      call = c;

  ValueRange inputs = iterOpaqueKernel.getInputs();
  TypeRange resultTypes = iterOpaqueKernel.getResultTypes();

  // The execution device is where the kernel writes its output, read from the
  // output tensor TYPE (`gpu:0`/`cpu:0`). The `map.iter.opaque` op itself
  // carries no device, so reading it would wrongly fall back to host for GPU
  // ops. This device is recorded on each reconstructed tensor; its ID is later
  // zeroed by `stripDeviceIdsAndDynamicShapes` for dedup, matching the stub's
  // device.
  DeviceRefAttr device =
      resultTypes.empty()
          ? MO::getDeviceOrDefaultAny(cast<MO::TensorType>(inputs[0].getType()))
          : MO::getDeviceOrDefaultAny(cast<MO::TensorType>(resultTypes[0]));

  OpBuilder builder(iterOpaqueKernel);
  auto kernel = MOGG::KernelRegionOp::create(builder, loc, resultTypes, inputs);
  // The `MOExplicitDevice` verifier requires a device; record the same device
  // used for the reconstructed tensors so outlining and `MOGGToMGP` resolve the
  // real execution device instead of the host fallback.
  kernel.setDeviceRef(device);

  // Forward the fusion-tree provenance `MOToMAP` recorded on `map.iter.opaque`
  // (typed `AnyAttr` there so the `map` dialect need not depend on `mogg`);
  // fall back to a fresh leaf only if it's absent (e.g. hand-written test IR).
  if (Attribute fusionInfo = iterOpaqueKernel.getFusionInfoAttr())
    kernel.setFusionInfoAttr(cast<MOGG::FusionInfoAttr>(fusionInfo));
  else
    kernel.setFusionInfoAttr(MOGG::FusionInfoAttr::get(
        ctx, MOGG::createFusionLeaf(iterOpaqueKernel)));

  // Unique dynamic dims across all input/output tensors (first-seen order); the
  // kernel gets one !mogg.dim_ block arg per unique dim.
  llvm::MapVector<Attribute, unsigned> dimIndex;
  auto collectDims = [&](TypeRange types) {
    for (Type ty : types)
      for (Attribute dim :
           cast<MO::TensorType>(ty).getTypedShapeAttr().getValues())
        if (isDynamicDim(dim))
          dimIndex.insert({dim, static_cast<unsigned>(dimIndex.size())});
  };
  collectDims(inputs.getTypes());
  collectDims(resultTypes);

  // Body block args: one !mogg.pointer_ per input, then per output, then one
  // !mogg.dim_ per unique dynamic dim, then the context.
  Block *body = builder.createBlock(&kernel.getBody());
  SmallVector<BlockArgument> inputPtrs, outputPtrs, dimArgs;
  for (size_t i = 0, e = inputs.size(); i < e; ++i)
    inputPtrs.push_back(body->addArgument(MOGG::PointerType::get(ctx), loc));
  for (size_t i = 0, e = resultTypes.size(); i < e; ++i)
    outputPtrs.push_back(body->addArgument(MOGG::PointerType::get(ctx), loc));
  // One `!mogg.dim_` per unique dynamic dim, each carrying that dim's parameter
  // name (e.g. `!mogg.dim_<batch>`); shapes never hold dim expressions, so the
  // dynamic dim is always a `ParamDeclRefAttr`.
  for (Attribute dim : llvm::make_first_range(dimIndex))
    dimArgs.push_back(body->addArgument(
        MOGG::DimType::get(ctx, dyn_cast<KGEN::ParamDeclRefAttr>(dim)), loc));
  BlockArgument contextArg =
      body->addArgument(MOGG::ContextType::get(ctx), loc);

  builder.setInsertionPointToEnd(body);

  SmallVector<Value> inputTensors;
  for (auto [i, in] : llvm::enumerate(inputs))
    inputTensors.push_back(buildTensor(builder, loc, inputPtrs[i], in.getType(),
                                       MO::IOKind::Input, /*isMut=*/false,
                                       device, dimArgs, dimIndex));
  SmallVector<Value> outputTensors, outputShapes;
  for (auto [i, ty] : llvm::enumerate(resultTypes)) {
    Value shape;
    outputTensors.push_back(buildTensor(builder, loc, outputPtrs[i], ty,
                                        MO::IOKind::Output, /*isMut=*/true,
                                        device, dimArgs, dimIndex, &shape));
    outputShapes.push_back(shape);
  }

  // A kernel need not contain a call; only build the `mogg._call.opaque` when a
  // `map.call.opaque` is present.
  if (call) {
    // Each call output is either a plain opaque output block arg (mapped to
    // its reconstructed tensor) or a `map.iter.epilogue` result (lowered to a
    // lambda + epilogue tensor).
    builder.setInsertionPointToEnd(body);
    SmallVector<Value> callOutputs;
    for (Value callOut : call.getOutputs()) {
      if (auto barg = dyn_cast<BlockArgument>(callOut)) {
        callOutputs.push_back(outputTensors[barg.getArgNumber()]);
        continue;
      }
      auto epilogue = callOut.getDefiningOp<MAP::IterEpilogueOp>();
      if (!epilogue)
        return iterOpaqueKernel.emitOpError(
            "MAPToMOGG: call output is neither a plain output nor an epilogue");
      FailureOr<Value> epiTensor = buildEpilogueTensor(
          builder, loc, epilogue, outputTensors, outputShapes, device);
      if (failed(epiTensor))
        return iterOpaqueKernel.emitOpError(
            "MAPToMOGG: unsupported epilogue body");
      callOutputs.push_back(*epiTensor);
    }

    builder.setInsertionPointToEnd(body);
    MOGG::CallOpaqueOp::create(builder, loc, call.getCalleeAttr(), inputTensors,
                               callOutputs, contextArg, call.getParams());
  }

  iterOpaqueKernel.replaceAllUsesWith(kernel.getResults());
  iterOpaqueKernel.erase();
  return success();
}

struct MAPToMOGGPass : public MAP::impl::MAPToMOGGBase<MAPToMOGGPass> {
  using Base::Base;

  void runOnOperation() override {
    if (!MO::useAdvancedFusion())
      return;

    SmallVector<MAP::IterOpaqueOp> worklist;
    getOperation()->walk([&](MAP::IterOpaqueOp op) {
      worklist.push_back(op);
      // No need to recurse inside a `map.iter.opaque`.
      return WalkResult::skip();
    });

    for (MAP::IterOpaqueOp op : worklist)
      if (failed(lowerOpaque(op)))
        return signalPassFailure();
  }
};

} // namespace

} // namespace M::MAP
