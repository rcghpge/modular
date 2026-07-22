//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Lowers MO ops into `map` kernels. Each lowerable MO op with a registered
// Mojo kernel becomes either a `map.iter.opaque` wrapping a `map.call` to that
// kernel (with a trivial store epilogue per result), or, for elementwise
// kernels, a `map.iter.foreach` expressing the loop body explicitly as scalar
// `map.load` / `map.call.elementwise` / `map.store` ops. Kernel selection
// reuses the same `mogg.kernel.decl` machinery as MOToMOGG (MojoRegistry /
// getMojoKernelAdaptor). Contexts, variadics and scalars are intentionally
// NOT materialized here; that is deferred until after fusion.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/GraphCompiler/Conversion/MOToMAP.h"
#include "GraphCompiler/GraphCompiler/MAPDialect/Dialect/MAPOps.h"
#include "GraphCompiler/GraphCompiler/MODialect/MODialect.h"
#include "GraphCompiler/GraphCompiler/MODialect/MOInterfaces.h"
#include "GraphCompiler/GraphCompiler/MODialect/MOOps.h"
#include "GraphCompiler/GraphCompiler/MOGGDialect/Analysis/KernelDeclAdaptor.h"
#include "GraphCompiler/GraphCompiler/MOGGDialect/Analysis/MojoRegistry.h"
#include "GraphCompiler/GraphCompiler/MOGGDialect/Analysis/Support.h"
#include "GraphCompiler/GraphCompiler/MOGGDialect/MOGGAttrs.h"
#include "GraphCompiler/GraphCompiler/MOGGDialect/Support/FusionUtils.h"
#include "GraphCompiler/GraphCompiler/MOGGDialect/Support/OperandMapper.h"
#include "GraphCompiler/GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "KGEN/KGENDialect/KGENAttrs.h"
#include "mlir/IR/Builders.h"

using namespace M;
using namespace mlir;

namespace M::MO {

#define GEN_PASS_DEF_MOTOMAP
#include "GraphCompiler/GraphCompiler/Conversion/MOToMAP.h.inc"

namespace {

bool shouldLower(Operation *op) {
  if (isa<MO::PreservedDuringKernelLowering>(op))
    return false;
  Dialect *dialect = op->getDialect();
  return dialect && isa<MO::MODialect>(dialect);
}

DictionaryAttr forwardParameters(Operation *op, OpBuilder &builder) {
  if (auto custom = dyn_cast<MO::CustomOp>(op)) {
    if (DictionaryAttr params = custom.getParameters())
      return params;
    return builder.getDictionaryAttr({});
  }

  SmallVector<NamedAttribute> params;
  for (NamedAttribute attr : op->getAttrs()) {
    Attribute value = attr.getValue();
    if (isa<M::DeviceRefAttr>(value))
      continue;
    if (auto typeAttr = dyn_cast<TypeAttr>(value)) {
      params.push_back(NamedAttribute(
          attr.getName(),
          KGEN::DTypeConstantAttr::get(
              value.getContext(), MO::getDTypeOrInvalid(typeAttr.getValue()))));
      continue;
    }
    if (isa<TypedAttr>(value))
      params.push_back(attr);
  }
  return builder.getDictionaryAttr(params);
}

MAP::IterEpilogueOp buildStoreEpilogue(OpBuilder &builder, Location loc,
                                       BlockArgument out) {
  auto outType = cast<MO::TensorType>(out.getType());
  auto indexType = MAP::IndexType::get(builder.getContext(), outType.getRank());

  auto epilogue =
      MAP::IterEpilogueOp::create(builder, loc, out.getType(), ValueRange{out});

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

LogicalResult lowerToOpaque(Operation *op,
                            const MOGG::KernelDeclAdaptor &adaptor,
                            const MOGG::KernelDeclOperandMapping &mapping) {
  for (const auto &[idx, arg] : mapping.operandMapping)
    if (arg.isFusedTensorType())
      return op->emitOpError(
          "MOToMAP: input (prologue) fusion is not yet supported");

  llvm::DenseMap<uint64_t, bool> outputFusible;
  for (const auto &[idx, arg] : mapping.resultMapping)
    outputFusible[idx] = arg.isFusedTensorType();

  OpBuilder builder(op);
  Location loc = op->getLoc();

  auto opaque = MAP::IterOpaqueOp::create(builder, loc, op->getResultTypes(),
                                          op->getOperands());
  opaque.setFusionInfoAttr(
      MOGG::FusionInfoAttr::get(op->getContext(), MOGG::createFusionLeaf(op)));

  Block *body = builder.createBlock(&opaque.getBody());
  SmallVector<BlockArgument> outs;
  for (Type resultType : op->getResultTypes())
    outs.push_back(body->addArgument(resultType, loc));

  builder.setInsertionPointToStart(body);
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

  builder.setInsertionPointToEnd(body);
  MAP::CallOp::create(builder, loc,
                      FlatSymbolRefAttr::get(
                          op->getContext(), adaptor.fullyResolvedName.strref()),
                      forwardParameters(op, builder),
                      /*inputs=*/op->getOperands(),
                      /*outputs=*/callOutputs);

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

  MOSH::ShapeAttr shapeAttr = resultTensorTy.getTypedShapeAttr();
  unsigned rank = static_cast<unsigned>(shapeAttr.getRank());
  Type elemTy = resultTensorTy.getElementType();

  auto foreachOp = MAP::IterForeachOp::create(
      builder, loc,
      /*resultTypes=*/SmallVector<Type>{resultTensorTy},
      /*shape=*/shapeAttr,
      /*inputs=*/op->getOperands(),
      /*fusionInfo=*/
      MOGG::FusionInfoAttr::get(ctx, MOGG::createFusionLeaf(op)));

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
  Value scalar = MAP::CallElementwiseOp::create(
      bodyBuilder, loc, elemTy, kernelSymbol, idxArg, scalarLoads);

  // map.store into the output block argument.
  BlockArgument outputArg =
      body->getArgument(static_cast<unsigned>(op->getNumOperands()));
  MAP::StoreOp::create(bodyBuilder, loc, outputArg, idxArg, scalar);

  op->replaceAllUsesWith(foreachOp.getResults());
  op->erase();
  return success();
}

LogicalResult lowerOp(Operation *op, MOGG::MojoRegistry &registry) {
  ErrorOr<MOGG::KernelDeclAdaptor> adaptor =
      MOGG::getMojoKernelAdaptor(registry, op);
  if (adaptor.isError())
    return success();

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
