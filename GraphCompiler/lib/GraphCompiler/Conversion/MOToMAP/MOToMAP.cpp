//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Lowers MO ops into `map` / `imap` kernels. Each lowerable MO op with a
// registered Mojo kernel becomes a `map.iter.opaque` wrapping a `map.call` to
// that kernel, with a trivial store epilogue per result. Kernel selection
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
  MAP::StoreOp::create(builder, loc, /*dest=*/cap, /*index=*/idx,
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

LogicalResult lowerOp(Operation *op, MOGG::MojoRegistry &registry) {
  ErrorOr<MOGG::KernelDeclAdaptor> adaptor =
      MOGG::getMojoKernelAdaptor(registry, op);
  if (adaptor.isError())
    return success();
  if (!adaptor->executeFunction)
    return op->emitOpError("MOToMAP: kernel has no execute function");

  ErrorOr<MOGG::KernelDeclOperandMapping> mapping =
      MOGG::mapMOOpToKernelImplementation(op, *adaptor->executeFunction);
  if (mapping.isError())
    return op->emitOpError("MOToMAP: failed to map operands to kernel");

  if (isa<MO::InvokeShapeFuncOp>(op))
    return op->emitOpError(
        "MOToMAP: shape-function kernels are not yet supported");
  else if (adaptor->isElementwise())
    return op->emitOpError(
        "MOToMAP: elementwise kernels are not yet supported");
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
