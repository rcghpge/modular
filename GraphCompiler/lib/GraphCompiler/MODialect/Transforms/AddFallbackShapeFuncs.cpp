//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/GraphCompiler/MOGGDialect/Support/FusionUtils.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"

#include "GraphCompiler/GraphCompiler/MODialect/MOInterfaces.h"
#include "GraphCompiler/GraphCompiler/MODialect/MOOps.h"
#include "GraphCompiler/GraphCompiler/MODialect/Transforms/MOTransforms.h"
#include "GraphCompiler/GraphCompiler/MOGGDialect/Analysis/KernelDeclAdaptor.h"
#include "GraphCompiler/GraphCompiler/MOGGDialect/Analysis/MojoRegistry.h"
#include "GraphCompiler/GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "Support/AssertStream.h"
#include "llvm/ADT/StringSet.h"
#include <numeric>

#define DEBUG_TYPE "add-fallback-shape-funcs"

namespace M::MO {

#define GEN_PASS_DEF_ADDFALLBACKSHAPEFUNCTIONS
#include "GraphCompiler/GraphCompiler/MODialect/Transforms/MOTransforms.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Fallback guard creation (to ensure all MO ops have one or error)
//===----------------------------------------------------------------------===//

// C++ workaround so `false` depends on `T`
template <class>
[[maybe_unused]] constexpr bool FALSE = false;

// Eventually all MO types will implement this function and we will instantiate
// all of them using the macros in the MLIR generated headers. Once we flip the
// switch on that adding a MO op without a fallback or a placeholder entry
// designating that it has no fallback will produce a compile time error.
template <typename T>
struct Fallback {
  static_assert(
      FALSE<T>,
      "All MO operations must have registered fallback shape "
      "functions but no function found for this MO op. (see template error)");
};

#define HAS_FALLBACK(MOOp)                                                     \
  template <>                                                                  \
  struct Fallback<MOOp> {                                                      \
    ErrorOrSuccess build(GraphOp &graph, OpBuilder &builder, MOOp op);         \
  };

#define NEEDS_NO_FALLBACK(MOOp)                                                \
  template <>                                                                  \
  struct Fallback<MOOp> {};

HAS_FALLBACK(MO::ArgMaxOp)
HAS_FALLBACK(MO::ArgMinOp)
HAS_FALLBACK(MO::AvgPoolOp)
HAS_FALLBACK(MO::AvgPoolCeilOp)
HAS_FALLBACK(MO::BatchMatmulOp)
HAS_FALLBACK(MO::BottomKOp)
HAS_FALLBACK(MO::BroadcastToOp)
HAS_FALLBACK(MO::BroadcastShapeOp)
HAS_FALLBACK(MO::ConcatOp)
HAS_FALLBACK(MO::ConvOp)
HAS_FALLBACK(MO::ConvTransposeOp)
HAS_FALLBACK(MO::GatherOp)
HAS_FALLBACK(MO::GatherNdOp)
HAS_FALLBACK(MO::ReduceGroupNormOp)
HAS_FALLBACK(MO::LayerNormOp)
HAS_FALLBACK(MO::ReduceMinAndMaxOp)
HAS_FALLBACK(MO::ReduceRMSNormOp)
HAS_FALLBACK(MO::ReduceRMSNormRoPEOp)
HAS_FALLBACK(MO::ReduceRMSNormFusedResidualAddOp)
HAS_FALLBACK(MO::MaxPoolOp)
HAS_FALLBACK(MO::MaxPoolCeilOp)
HAS_FALLBACK(MO::MeanOp)
HAS_FALLBACK(MO::NonMaximumSuppressionOp)
HAS_FALLBACK(MO::PadConstantOp)
HAS_FALLBACK(MO::PadReflectOp)
HAS_FALLBACK(MO::PadRepeatOp)
HAS_FALLBACK(MO::RandomNormalOp)
HAS_FALLBACK(MO::RandomUniformOp)
HAS_FALLBACK(MO::RangeOp)
HAS_FALLBACK(MO::ReduceAddOp)
HAS_FALLBACK(MO::ReduceMaxOp)
HAS_FALLBACK(MO::ReduceMinOp)
HAS_FALLBACK(MO::ReduceMulOp)
HAS_FALLBACK(MO::ReshapeOp)
HAS_FALLBACK(MO::RoiAlignOp)
HAS_FALLBACK(MO::ResizeLinearOp)
HAS_FALLBACK(MO::ResizeNearestOp)
HAS_FALLBACK(MO::ScatterOp)
HAS_FALLBACK(MO::ScatterAddOp)
HAS_FALLBACK(MO::ScatterMaxOp)
HAS_FALLBACK(MO::ScatterMinOp)
HAS_FALLBACK(MO::ScatterMulOp)
HAS_FALLBACK(MO::ScatterNdOp)
HAS_FALLBACK(MO::ScatterNdAddOp)
HAS_FALLBACK(MO::ScatterNdMaxOp)
HAS_FALLBACK(MO::ScatterNdMinOp)
HAS_FALLBACK(MO::ScatterNdMulOp)
HAS_FALLBACK(MO::SliceOp)
HAS_FALLBACK(MO::SplitOp)
HAS_FALLBACK(MO::SqueezeShapeOp)
HAS_FALLBACK(MO::TopKOp)
HAS_FALLBACK(MO::TransposeOp)
HAS_FALLBACK(MO::UnsqueezeShapeOp)
HAS_FALLBACK(MO::ArgNonzeroOp)
HAS_FALLBACK(MO::TileOp)

template <>
struct Fallback<MO::CustomOp> {
  ErrorOrSuccess build(GraphOp &graph, OpBuilder &builder, MO::CustomOp op,
                       StringSet<> &inputParamsNeeded);
};

// Eventually all ops must be accounted for here. They either must have a
// fallback or an explicit vouch that they need no fallback.
NEEDS_NO_FALLBACK(MO::MatmulOp)
NEEDS_NO_FALLBACK(MO::Static::BroadcastToOp)

// Creates the fallback for this op if it has parameters still needing decls.
#define CALL_FALLBACK(MOOp)                                                    \
  if (auto asOp = dyn_cast<MOOp>(op.getOperation())) {                         \
    auto result = Fallback<MOOp>{}.build(graph, builder, asOp);                \
    if (result.isError()) {                                                    \
      asOp.emitError() << "Failed to fallback shape function with error: "     \
                       << result.takeError();                                  \
      return mlir::WalkResult::interrupt();                                    \
    } else {                                                                   \
      op.setOutputParamDecls({});                                              \
    }                                                                          \
    return mlir::WalkResult::advance();                                        \
  }

//===----------------------------------------------------------------------===//
// Fallback creation pass.
//===----------------------------------------------------------------------===//

struct AddFallbackShapeFunctions
    : public impl::AddFallbackShapeFunctionsBase<AddFallbackShapeFunctions> {
  using Base::Base;

  void runOnOperation() override {
    MO::GraphOp graph = getOperation();
    MLIRContext *ctx = graph.getContext();
    OpBuilder builder{ctx};
    MOGG::MojoRegistry registry(graph->getParentOfType<mlir::ModuleOp>());

    WalkResult result = graph.walk([&](ParamDeclarationInterface op) {
      // Shape function was already materialized or never needed.
      if (op.getOutputParamDecls().empty())
        return mlir::WalkResult::advance();

      // Call the fallbacks.
      builder.setInsertionPoint(op);
      CALL_FALLBACK(MO::ArgMaxOp)
      CALL_FALLBACK(MO::ArgMinOp)
      CALL_FALLBACK(MO::AvgPoolOp)
      CALL_FALLBACK(MO::AvgPoolCeilOp)
      CALL_FALLBACK(MO::BatchMatmulOp)
      CALL_FALLBACK(MO::BottomKOp)
      CALL_FALLBACK(MO::BroadcastToOp)
      CALL_FALLBACK(MO::BroadcastShapeOp)
      CALL_FALLBACK(MO::ConcatOp)
      CALL_FALLBACK(MO::ConvOp)
      CALL_FALLBACK(MO::ConvTransposeOp)
      CALL_FALLBACK(MO::GatherOp)
      CALL_FALLBACK(MO::GatherNdOp)
      CALL_FALLBACK(MO::ReduceGroupNormOp)
      CALL_FALLBACK(MO::LayerNormOp)
      CALL_FALLBACK(MO::ReduceMinAndMaxOp)
      CALL_FALLBACK(MO::ReduceRMSNormOp)
      CALL_FALLBACK(MO::ReduceRMSNormRoPEOp)
      CALL_FALLBACK(MO::ReduceRMSNormFusedResidualAddOp)
      CALL_FALLBACK(MO::MaxPoolOp)
      CALL_FALLBACK(MO::MaxPoolCeilOp)
      CALL_FALLBACK(MO::MeanOp)
      CALL_FALLBACK(MO::NonMaximumSuppressionOp)
      CALL_FALLBACK(MO::PadConstantOp)
      CALL_FALLBACK(MO::PadReflectOp)
      CALL_FALLBACK(MO::PadRepeatOp)
      CALL_FALLBACK(MO::RandomNormalOp)
      CALL_FALLBACK(MO::RandomUniformOp)
      CALL_FALLBACK(MO::RangeOp)
      CALL_FALLBACK(MO::ReduceAddOp)
      CALL_FALLBACK(MO::ReduceMaxOp)
      CALL_FALLBACK(MO::ReduceMinOp)
      CALL_FALLBACK(MO::ReduceMulOp)
      CALL_FALLBACK(MO::ReshapeOp)
      CALL_FALLBACK(MO::RoiAlignOp)
      CALL_FALLBACK(MO::ResizeLinearOp)
      CALL_FALLBACK(MO::ResizeNearestOp)
      CALL_FALLBACK(MO::ScatterOp)
      CALL_FALLBACK(MO::ScatterAddOp)
      CALL_FALLBACK(MO::ScatterMaxOp)
      CALL_FALLBACK(MO::ScatterMinOp)
      CALL_FALLBACK(MO::ScatterMulOp)
      CALL_FALLBACK(MO::ScatterNdOp)
      CALL_FALLBACK(MO::ScatterNdAddOp)
      CALL_FALLBACK(MO::ScatterNdMaxOp)
      CALL_FALLBACK(MO::ScatterNdMinOp)
      CALL_FALLBACK(MO::ScatterNdMulOp)
      CALL_FALLBACK(MO::SliceOp)
      CALL_FALLBACK(MO::SplitOp)
      CALL_FALLBACK(MO::SqueezeShapeOp)
      CALL_FALLBACK(MO::TopKOp)
      CALL_FALLBACK(MO::TransposeOp)
      CALL_FALLBACK(MO::UnsqueezeShapeOp)
      CALL_FALLBACK(MO::ArgNonzeroOp)
      CALL_FALLBACK(MO::TileOp)

      if (auto asOp = dyn_cast<MO::CustomOp>(op.getOperation())) {
        if (!registry.containsKernel(asOp.getSymbol())) {
          asOp.emitError() << "Could not find registered mojo kernel named '"
                           << asOp.getSymbol() << "'";
          return mlir::WalkResult::interrupt();
        }
        MOGG::KernelDeclOp declOp =
            registry.getResolvedKernel(asOp.getSymbol());

        // Collect compile-time parameter names so the fallback can forward
        // matching custom-op attributes to the shape function call. Sourced
        // from the shape fn's own params (it may differ from execute, e.g.
        // shape typically omits `target`/`_trace_name`).
        StringSet<> inputParams;
        if (auto shapeAttr = declOp.getShape()) {
          for (mlir::Attribute attr : shapeAttr->getParams()) {
            inputParams.insert(mlir::cast<KGEN::ParamDeclAttr>(attr).getName());
          }
        }

        auto result =
            Fallback<MO::CustomOp>{}.build(graph, builder, asOp, inputParams);
        if (result.isError()) {
          asOp.emitError() << "Failed to fallback shape function with error: "
                           << result.takeError();
          return mlir::WalkResult::interrupt();

        } else {
          op.setOutputParamDecls({});
        }
      }

      return mlir::WalkResult::advance();
    });

    if (result.wasInterrupted())
      return signalPassFailure();
  }
};

//===----------------------------------------------------------------------===//
// Fallback helpers
//===----------------------------------------------------------------------===//

// Each fallback needs to produce a parameter for every dimension in the output
// even if those dimensions are static. This simplifies the lowering as without
// doing this we would need the lowering to know how many should be produced and
// to which dimension they should be referring to.
KGEN::ParamDeclArrayAttr
getNewParameterMapping(GraphOp &graph, TensorType outTy,
                       ArrayRef<ParamDeclAttr> outputParamDecls) {
  MLIRContext *ctx = graph.getContext();
  if (auto rankedParamDeclRef =
          dyn_cast<KGEN::ParamDeclRefAttr>(outTy.getShapeAttr())) {
    auto paramDecl = KGEN::ParamDeclAttr::get(rankedParamDeclRef);
    return KGEN::ParamDeclArrayAttr::get(ctx, {paramDecl});
  }

  llvm::SmallSetVector<ParamDeclAttr, 10> outputParamDeclSet(
      outputParamDecls.begin(), outputParamDecls.end());

  // New attrs. These are either the exact same as the old attribute or a new
  // attribute
  SmallVector<KGEN::ParamDeclAttr> params;

  // We want to move all parameters declared by the associated operator to
  // to be defined by the new shape func invocation.
  //
  // In the case of a dim whose results we don't need to define, we add a new
  // "F"allback dim which is useless but needed to ensure the contract of the
  // shape func.
  auto getShapeFuncSymbol = [&](TypedAttr dim) -> KGEN::ParamDeclAttr {
    auto asSymbolic = dyn_cast<KGEN::ParamDeclRefAttr>(dim);

    // Is an integer we already know.
    if (!asSymbolic)
      return graph.getNewDimParam("F");

    // Is defined by some other shape.
    auto asParamDeclAttr = KGEN::ParamDeclAttr::get(asSymbolic);
    if (!outputParamDeclSet.contains(asParamDeclAttr))
      return graph.getNewDimParam("F");

    return asParamDeclAttr;
  };

  // Parameters assumed to be in order. Add the index of each symbolic
  // dimension.
  for (auto dim : outTy.getTypedShapeAttr().getValues()) {
    params.push_back(getShapeFuncSymbol(dim));
  }
  return KGEN::ParamDeclArrayAttr::get(graph.getContext(), params);
}

//===----------------------------------------------------------------------===//
// Helpers for fallback implementations
//===----------------------------------------------------------------------===//

static ErrorOrSuccess buildForReductionOp(GraphOp &graph, OpBuilder &builder,
                                          MO::Reduction op) {
  KGEN::ParamDeclArrayAttr shapeParams = getNewParameterMapping(
      graph, op.getResult().getType(), op.getOutputParamDecls());

  // Create the shape function.
  MO::InvokeShapeFuncOp::create(
      builder, op.getLoc(), op->getOperands(), op->getName().getStringRef(),
      shapeParams,
      /*kgenAttrs*/ DictionaryAttr{},
      /*Data dependent input indices*/ builder.getI32ArrayAttr({1}));

  return success();
}

static ErrorOrSuccess buildForScatterLikeOp(GraphOp &graph, OpBuilder &builder,
                                            MO::ScatterLike op) {
  KGEN::ParamDeclArrayAttr shapeParams = getNewParameterMapping(
      graph, op.getResult().getType(), op.getOutputParamDecls());

  // Create the shape function.
  MO::InvokeShapeFuncOp::create(
      builder, op.getLoc(), op->getOperands(),
      op.getOperation()->getName().getStringRef(), shapeParams,
      /*kgenAttrs*/ DictionaryAttr{},
      /*Data dependent input indices*/ builder.getI32ArrayAttr({3}));

  return success();
}

void buildForOpResult(GraphOp &graph, OpBuilder &builder, TensorValue res,
                      StringRef shapeFuncName, ValueRange operands,
                      ArrayRef<ParamDeclAttr> outputParamDecls,
                      ArrayRef<std::pair<StringRef, Attribute>> kgenParams,
                      ArrayRef<int32_t> dataDepInputIdxs) {
  KGEN::ParamDeclArrayAttr shapeParams =
      getNewParameterMapping(graph, res.getType(), outputParamDecls);

  // TODO we can't deduce parameter from types of variadic inputs
  DictionaryAttr kgenParamAttrs;
  if (!kgenParams.empty()) {
    SmallVector<mlir::NamedAttribute> namedAttrs;
    for (auto [name, attr] : kgenParams)
      namedAttrs.push_back(builder.getNamedAttr(name, attr));
    kgenParamAttrs = builder.getDictionaryAttr(namedAttrs);
  }

  // Create the shape function.
  Location loc = res.getLoc();
  MO::InvokeShapeFuncOp::create(builder, loc, operands, shapeFuncName,
                                shapeParams, kgenParamAttrs,
                                builder.getI32ArrayAttr(dataDepInputIdxs));
}

//===----------------------------------------------------------------------===//
// Actual fallback implementations
//===----------------------------------------------------------------------===//

template <typename OpT>
static ErrorOrSuccess buildForScatterNdLikeOp(GraphOp &graph,
                                              OpBuilder &builder, OpT op) {

  KGEN::ParamDeclArrayAttr shapeParams = getNewParameterMapping(
      graph, op.getResult().getType(), op.getOutputParamDecls());
  buildForOpResult(graph, builder, op, OpT::getOperationName(),
                   op.getOperands(), shapeParams,
                   /*kgenParams=*/{},
                   /*dataDepInputIdxs=*/op.getDataDependentInputIndices());

  return success();
}

ErrorOrSuccess Fallback<MO::AvgPoolOp>::build(GraphOp &graph,
                                              OpBuilder &builder,
                                              MO::AvgPoolOp op) {
  buildForOpResult(graph, builder, op, MO::AvgPoolOp::getOperationName(),
                   op.getOperands(), op.getOutputParamDecls(),
                   /*kgenParams=*/{},
                   /*dataDepInputIdxs=*/op.getDataDependentInputIndices());
  return success();
}

ErrorOrSuccess Fallback<MO::AvgPoolCeilOp>::build(GraphOp &graph,
                                                  OpBuilder &builder,
                                                  MO::AvgPoolCeilOp op) {
  buildForOpResult(graph, builder, op, MO::AvgPoolCeilOp::getOperationName(),
                   op.getOperands(), op.getOutputParamDecls(),
                   /*kgenParams=*/{},
                   /*dataDepInputIdxs=*/op.getDataDependentInputIndices());
  return success();
}

ErrorOrSuccess Fallback<MO::BatchMatmulOp>::build(GraphOp &graph,
                                                  OpBuilder &builder,
                                                  MO::BatchMatmulOp op) {
  buildForOpResult(graph, builder, op, MO::BatchMatmulOp::getOperationName(),
                   op.getOperands(), op.getOutputParamDecls(),
                   /*kgenParams=*/{}, op.getDataDependentInputIndices());
  return success();
}

ErrorOrSuccess Fallback<MO::BottomKOp>::build(GraphOp &graph,
                                              OpBuilder &builder,
                                              MO::BottomKOp op) {
  buildForOpResult(graph, builder, op.getValues(),
                   MO::BottomKOp::getOperationName(), op.getOperands(),
                   op.getOutputParamDecls(),
                   /*kgenParams=*/{}, /*dataDepInputIdxs=*/{1, 2});
  // indices() guaranteed by verifier to have the same shape as values()
  return success();
}

ErrorOrSuccess Fallback<MO::BroadcastToOp>::build(GraphOp &graph,
                                                  OpBuilder &builder,
                                                  MO::BroadcastToOp op) {
  buildForOpResult(graph, builder, op, MO::BroadcastToOp::getOperationName(),
                   op.getOperands(), op.getOutputParamDecls(),
                   /*kgenParams=*/{},
                   /*dataDepInputIdxs=*/{1});
  return success();
}

ErrorOrSuccess Fallback<MO::BroadcastShapeOp>::build(GraphOp &graph,
                                                     OpBuilder &builder,
                                                     MO::BroadcastShapeOp op) {
  buildForOpResult(graph, builder, op, MO::BroadcastShapeOp::getOperationName(),
                   op.getOperands(), op.getOutputParamDecls(),
                   /*kgenParams=*/{}, /*dataDepInputIdxs=*/{});
  return success();
}

ErrorOrSuccess Fallback<MO::ConcatOp>::build(GraphOp &graph, OpBuilder &builder,
                                             MO::ConcatOp op) {
  // TODO we can't deduce parameter from types of variadic inputs
  auto inputTensorTy = cast<TensorType>(op.getInputs().front().getType());
  auto inputDTypeAttr = mlir::TypeAttr::get(inputTensorTy.getElementType());
  auto inputRankAttr = builder.getIndexAttr(inputTensorTy.getRank());

  buildForOpResult(graph, builder, op, MO::ConcatOp::getOperationName(),
                   op.getOperands(), op.getOutputParamDecls(),
                   /*kgenParams=*/
                   {{"input_type", inputDTypeAttr},
                    {"input_rank", inputRankAttr},
                    {"axis", op.getAxisAttr()}},
                   op.getDataDependentInputIndices());
  return success();
}

ErrorOrSuccess Fallback<MO::ConvOp>::build(GraphOp &graph, OpBuilder &builder,
                                           MO::ConvOp op) {
  buildForOpResult(graph, builder, op, MO::ConvOp::getOperationName(),
                   op.getOperands(), op.getOutputParamDecls(),
                   /*kgenParams=*/{}, op.getDataDependentInputIndices());
  return success();
}

ErrorOrSuccess Fallback<MO::ConvTransposeOp>::build(GraphOp &graph,
                                                    OpBuilder &builder,
                                                    MO::ConvTransposeOp op) {
  buildForOpResult(graph, builder, op, MO::ConvTransposeOp::getOperationName(),
                   op.getOperands(), op.getOutputParamDecls(),
                   /*kgenParams=*/{},
                   /*dataDepInputIdxs=*/op.getDataDependentInputIndices());
  return success();
}

ErrorOrSuccess Fallback<MO::CustomOp>::build(GraphOp &graph, OpBuilder &builder,
                                             MO::CustomOp op,
                                             StringSet<> &inputParamsNeeded) {
  // Mark all inputs by default.
  SmallVector<Value> operands{op.getOperands().begin(), op.getOperands().end()};

  // If any of the operands is a buffer, we must load it and update the chain.
  Value currentChain = nullptr;
  MO::MutableLoadOp firstLoadOp = nullptr;
  if (!operands.empty() && isa<MO::ChainType>(operands.back().getType())) {
    currentChain = operands.back();
    operands.pop_back();
  }

  for (Value &operand : operands) {
    auto bufferTy = dyn_cast<MO::BufferType>(operand.getType());
    if (!bufferTy)
      continue;

    // Replace the buffer with the loaded tensor and update the chain.
    auto bufferLoadOp = MO::MutableLoadOp::create(
        builder, op.getLoc(), bufferTy.toTensorType(), currentChain.getType(),
        operand, currentChain);
    if (!firstLoadOp)
      firstLoadOp = bufferLoadOp;
    operand = bufferLoadOp.getOutTensor();
    currentChain = bufferLoadOp.getOutChain();
  }

  SmallVector<int32_t> dataDepInputIdxs(operands.size());
  std::iota(dataDepInputIdxs.begin(), dataDepInputIdxs.end(), 0);

  for (auto [idx, res] : llvm::enumerate(op.getResults())) {
    SmallVector<std::pair<StringRef, Attribute>> kgenParams;
    kgenParams.emplace_back("output_idx", builder.getIndexAttr(idx));

    // Shape functions are only needed for the tensor outputs.
    if (!isa<MO::TensorType>(res.getType()))
      continue;

    // Add other bespoke kgen params declared by the custom operator if they are
    // needed
    for (const NamedAttribute &namedAttr : op.getParameters())
      if (inputParamsNeeded.contains(namedAttr.getName()))
        kgenParams.emplace_back(namedAttr.getName(), namedAttr.getValue());

    buildForOpResult(graph, builder, cast<TensorValue>(res), op.getSymbol(),
                     operands, op.getOutputParamDecls(),
                     /*kgenParams=*/kgenParams,
                     /*dataDepInputIdxs=*/dataDepInputIdxs);
  }

  // Replace the input chain of the custom op with the final chain after loading
  // all buffers.
  if (currentChain)
    op->getOperands().back().replaceAllUsesExcept(currentChain, firstLoadOp);

  return success();
}

ErrorOrSuccess Fallback<MO::GatherOp>::build(GraphOp &graph, OpBuilder &builder,
                                             MO::GatherOp op) {
  // `axis` is a compile-time `index` attribute threaded to the shape function.
  buildForOpResult(graph, builder, op, MO::GatherOp::getOperationName(),
                   op.getOperands(), op.getOutputParamDecls(),
                   /*kgenParams=*/{{"axis", op.getAxisAttr()}},
                   /*dataDepInputIdxs=*/op.getDataDependentInputIndices());
  return success();
}

ErrorOrSuccess Fallback<MO::GatherNdOp>::build(GraphOp &graph,
                                               OpBuilder &builder,
                                               MO::GatherNdOp op) {
  buildForOpResult(graph, builder, op, MO::GatherNdOp::getOperationName(),
                   op.getOperands(), op.getOutputParamDecls(),
                   /*kgenParams=*/
                   {
                       {"batch_dims", op.getBatchDimsAttr()},
                   },
                   op.getDataDependentInputIndices());
  return success();
}

ErrorOrSuccess Fallback<MO::LayerNormOp>::build(GraphOp &graph,
                                                OpBuilder &builder,
                                                MO::LayerNormOp op) {
  buildForOpResult(graph, builder, op, MO::LayerNormOp::getOperationName(),
                   op.getOperands(), op.getOutputParamDecls(),
                   /*kgenParams=*/{},
                   /*dataDepInputIdxs=*/{});
  return success();
}

ErrorOrSuccess Fallback<MO::MaxPoolOp>::build(GraphOp &graph,
                                              OpBuilder &builder,
                                              MO::MaxPoolOp op) {

  buildForOpResult(graph, builder, op, MO::MaxPoolOp::getOperationName(),
                   op.getOperands(), op.getOutputParamDecls(),
                   /*kgenParams=*/{},
                   /*dataDepInputIdxs=*/op.getDataDependentInputIndices());
  return success();
}

ErrorOrSuccess Fallback<MO::MaxPoolCeilOp>::build(GraphOp &graph,
                                                  OpBuilder &builder,
                                                  MO::MaxPoolCeilOp op) {

  buildForOpResult(graph, builder, op, MO::MaxPoolCeilOp::getOperationName(),
                   op.getOperands(), op.getOutputParamDecls(),
                   /*kgenParams=*/{},
                   /*dataDepInputIdxs=*/op.getDataDependentInputIndices());
  return success();
}

ErrorOrSuccess
Fallback<MO::NonMaximumSuppressionOp>::build(GraphOp &graph, OpBuilder &builder,
                                             MO::NonMaximumSuppressionOp op) {
  buildForOpResult(graph, builder, op,
                   MO::NonMaximumSuppressionOp::getOperationName(),
                   op.getOperands(), op.getOutputParamDecls(),
                   /*kgenParams=*/{}, /*dataDepInputIdxs=*/{0, 1, 2, 3, 4});
  return success();
}

ErrorOrSuccess Fallback<MO::PadConstantOp>::build(GraphOp &graph,
                                                  OpBuilder &builder,
                                                  MO::PadConstantOp op) {
  buildForOpResult(graph, builder, op, MO::PadConstantOp::getOperationName(),
                   op.getOperands(), op.getOutputParamDecls(),
                   /*kgenParams=*/{}, op.getDataDependentInputIndices());
  return success();
}

ErrorOrSuccess Fallback<MO::PadReflectOp>::build(GraphOp &graph,
                                                 OpBuilder &builder,
                                                 MO::PadReflectOp op) {
  buildForOpResult(graph, builder, op, MO::PadReflectOp::getOperationName(),
                   op.getOperands(), op.getOutputParamDecls(),
                   /*kgenParams=*/{}, op.getDataDependentInputIndices());
  return success();
}

ErrorOrSuccess Fallback<MO::PadRepeatOp>::build(GraphOp &graph,
                                                OpBuilder &builder,
                                                MO::PadRepeatOp op) {
  buildForOpResult(graph, builder, op, MO::PadRepeatOp::getOperationName(),
                   op.getOperands(), op.getOutputParamDecls(),
                   /*kgenParams=*/{}, op.getDataDependentInputIndices());
  return success();
}

ErrorOrSuccess Fallback<MO::RandomNormalOp>::build(GraphOp &graph,
                                                   OpBuilder &builder,
                                                   MO::RandomNormalOp op) {
  buildForOpResult(graph, builder, op, MO::RandomNormalOp::getOperationName(),
                   op.getOperands(), op.getOutputParamDecls(),
                   /*kgenParams=*/{}, op.getDataDependentInputIndices());
  return success();
}

ErrorOrSuccess Fallback<MO::RandomUniformOp>::build(GraphOp &graph,
                                                    OpBuilder &builder,
                                                    MO::RandomUniformOp op) {
  buildForOpResult(graph, builder, op, MO::RandomUniformOp::getOperationName(),
                   op.getOperands(), op.getOutputParamDecls(),
                   /*kgenParams=*/{}, op.getDataDependentInputIndices());
  return success();
}

ErrorOrSuccess Fallback<MO::RangeOp>::build(GraphOp &graph, OpBuilder &builder,
                                            MO::RangeOp op) {
  buildForOpResult(graph, builder, op, MO::RangeOp::getOperationName(),
                   op->getOperands(), op.getOutputParamDecls(),
                   /*kgenParams=*/{}, /*dataDepInputIdxs=*/{0, 1, 2});
  return success();
}

// Reduction ops

#define BUILD_REDUCTION_OP(OP)                                                 \
  ErrorOrSuccess Fallback<OP>::build(GraphOp &graph, OpBuilder &builder,       \
                                     OP op) {                                  \
    return buildForReductionOp(graph, builder, op);                            \
  }

BUILD_REDUCTION_OP(MO::ArgMaxOp)
BUILD_REDUCTION_OP(MO::ArgMinOp)
BUILD_REDUCTION_OP(MO::MeanOp)
BUILD_REDUCTION_OP(MO::ReduceAddOp)
BUILD_REDUCTION_OP(MO::ReduceMulOp)
BUILD_REDUCTION_OP(MO::ReduceMaxOp)
BUILD_REDUCTION_OP(MO::ReduceMinOp)

#undef BUILD_REDUCTION_OP

ErrorOrSuccess Fallback<MO::ReshapeOp>::build(GraphOp &graph,
                                              OpBuilder &builder,
                                              MO::ReshapeOp op) {
  buildForOpResult(graph, builder, op, MO::ReshapeOp::getOperationName(),
                   op.getOperands(), op.getOutputParamDecls(),
                   /*kgenParams=*/{}, op.getDataDependentInputIndices());
  return success();
}

ErrorOrSuccess Fallback<MO::ResizeNearestOp>::build(GraphOp &graph,
                                                    OpBuilder &builder,
                                                    MO::ResizeNearestOp op) {
  buildForOpResult(graph, builder, op, MO::ResizeNearestOp::getOperationName(),
                   op.getOperands(), op.getOutputParamDecls(),
                   /*kgenParams=*/{}, op.getDataDependentInputIndices());
  return success();
}

ErrorOrSuccess Fallback<MO::ResizeLinearOp>::build(GraphOp &graph,
                                                   OpBuilder &builder,
                                                   MO::ResizeLinearOp op) {
  buildForOpResult(graph, builder, op, MO::ResizeLinearOp::getOperationName(),
                   op.getOperands(), op.getOutputParamDecls(),
                   /*kgenParams=*/{}, op.getDataDependentInputIndices());
  return success();
}

ErrorOrSuccess Fallback<MO::RoiAlignOp>::build(GraphOp &graph,
                                               OpBuilder &builder,
                                               MO::RoiAlignOp op) {
  // The shape function doesn't need spatial_scale and sampling_ratio.
  buildForOpResult(graph, builder, op, MO::RoiAlignOp::getOperationName(),
                   op.getOperands().take_front(4), op.getOutputParamDecls(),
                   /*kgenParams=*/{},
                   /*dataDepInputIdxs=*/op.getDataDependentInputIndices());
  return success();
}

// `mo.scatter` carries its axis as a compile-time `index` attribute (threaded
// to the shape function), so it uses the generic op-result builder rather than
// the scatter-like one used by the reduction variants.
ErrorOrSuccess Fallback<MO::ScatterOp>::build(GraphOp &graph,
                                              OpBuilder &builder,
                                              MO::ScatterOp op) {
  buildForOpResult(graph, builder, op, MO::ScatterOp::getOperationName(),
                   op.getOperands(), op.getOutputParamDecls(),
                   /*kgenParams=*/{{"axis", op.getAxisAttr()}},
                   /*dataDepInputIdxs=*/op.getDataDependentInputIndices());
  return success();
}

// ScatterLike ops (axis is a scalar tensor operand)

#define BUILD_SCATTER_LIKE_OP(OP)                                              \
  ErrorOrSuccess Fallback<OP>::build(GraphOp &graph, OpBuilder &builder,       \
                                     OP op) {                                  \
    return buildForScatterLikeOp(graph, builder, op);                          \
  }

BUILD_SCATTER_LIKE_OP(MO::ScatterAddOp)
BUILD_SCATTER_LIKE_OP(MO::ScatterMaxOp)
BUILD_SCATTER_LIKE_OP(MO::ScatterMinOp)
BUILD_SCATTER_LIKE_OP(MO::ScatterMulOp)

#undef BUILD_SCATTER_LIKE_OP

// ScatterNdLike ops

#define BUILD_SCATTER_ND_LIKE_OP(OP)                                           \
  ErrorOrSuccess Fallback<OP>::build(GraphOp &graph, OpBuilder &builder,       \
                                     OP op) {                                  \
    return buildForScatterNdLikeOp(graph, builder, op);                        \
  }

BUILD_SCATTER_ND_LIKE_OP(MO::ScatterNdOp)
BUILD_SCATTER_ND_LIKE_OP(MO::ScatterNdAddOp)
BUILD_SCATTER_ND_LIKE_OP(MO::ScatterNdMaxOp)
BUILD_SCATTER_ND_LIKE_OP(MO::ScatterNdMinOp)
BUILD_SCATTER_ND_LIKE_OP(MO::ScatterNdMulOp)

#undef BUILD_SCATTER_ND_LIKE_OP

ErrorOrSuccess Fallback<MO::SliceOp>::build(GraphOp &graph, OpBuilder &builder,
                                            MO::SliceOp op) {
  buildForOpResult(graph, builder, op, MO::SliceOp::getOperationName(),
                   op.getOperands(), op.getOutputParamDecls(),
                   /*kgenParams=*/{},
                   /*dataDepInputIdxs=*/op.getDataDependentInputIndices());
  return success();
}

ErrorOrSuccess Fallback<MO::SplitOp>::build(GraphOp &graph, OpBuilder &builder,
                                            MO::SplitOp op) {
  Location loc = op->getLoc();
  SmallVector<Value> operands(op->getOperands());
  operands.push_back(nullptr); // placeholder for output_idx
  for (auto [idx, res] : llvm::enumerate(op.getResults())) {
    operands.back() =
        MO::ConstantOp::create(builder, loc, static_cast<int64_t>(idx));
    // This uses some special handling
    buildForOpResult(graph, builder, cast<TensorValue>(res),
                     "split_ith_output_shape", operands,
                     op.getOutputParamDecls(),
                     /*kgenParams=*/{},
                     /*dataDepInputIdxs=*/{1, 2});
  }
  return success();
}

ErrorOrSuccess Fallback<MO::SqueezeShapeOp>::build(GraphOp &graph,
                                                   OpBuilder &builder,
                                                   MO::SqueezeShapeOp op) {
  buildForOpResult(graph, builder, op, MO::SqueezeShapeOp::getOperationName(),
                   op.getOperands(), op.getOutputParamDecls(),
                   /*kgenParams=*/{},
                   /*dataDepInputIdxs=*/op.getDataDependentInputIndices());
  return success();
}

ErrorOrSuccess Fallback<MO::TopKOp>::build(GraphOp &graph, OpBuilder &builder,
                                           MO::TopKOp op) {
  buildForOpResult(graph, builder, op.getValues(),
                   MO::TopKOp::getOperationName(), op.getOperands(),
                   op.getOutputParamDecls(),
                   /*kgenParams=*/{}, op.getDataDependentInputIndices());

  // indices() guaranteed by verifier to have the same shape as values()
  return success();
}

ErrorOrSuccess Fallback<MO::TransposeOp>::build(GraphOp &graph,
                                                OpBuilder &builder,
                                                MO::TransposeOp op) {
  buildForOpResult(graph, builder, op, MO::TransposeOp::getOperationName(),
                   op.getOperands(), op.getOutputParamDecls(),
                   /*kgenParams=*/{}, op.getDataDependentInputIndices());
  return success();
}

ErrorOrSuccess Fallback<MO::UnsqueezeShapeOp>::build(GraphOp &graph,
                                                     OpBuilder &builder,
                                                     MO::UnsqueezeShapeOp op) {
  buildForOpResult(graph, builder, op, MO::UnsqueezeShapeOp::getOperationName(),
                   op.getOperands(), op.getOutputParamDecls(),
                   /*kgenParams=*/{}, op.getDataDependentInputIndices());
  return success();
}

ErrorOrSuccess Fallback<MO::ArgNonzeroOp>::build(GraphOp &graph,
                                                 OpBuilder &builder,
                                                 MO::ArgNonzeroOp op) {
  buildForOpResult(graph, builder, op, MO::ArgNonzeroOp::getOperationName(),
                   op->getOperands(), op.getOutputParamDecls(),
                   /*kgenParams=*/{}, /*dataDepInputIdxs=*/{0});
  return success();
}

ErrorOrSuccess Fallback<MO::TileOp>::build(GraphOp &graph, OpBuilder &builder,
                                           MO::TileOp op) {
  buildForOpResult(graph, builder, op, MO::TileOp::getOperationName(),
                   op.getOperands(), op.getOutputParamDecls(),
                   /*kgenParams=*/{}, op.getDataDependentInputIndices());
  return success();
}

ErrorOrSuccess Fallback<MO::ReduceRMSNormOp>::build(GraphOp &graph,
                                                    OpBuilder &builder,
                                                    MO::ReduceRMSNormOp op) {
  buildForOpResult(graph, builder, op, MO::ReduceRMSNormOp::getOperationName(),
                   op.getOperands(), op.getOutputParamDecls(),
                   /*kgenParams=*/
                   {{"multiply_before_cast", op.getMultiplyBeforeCastAttr()}},
                   /*dataDepInputIdxs=*/{});
  return success();
}

ErrorOrSuccess
Fallback<MO::ReduceRMSNormRoPEOp>::build(GraphOp &graph, OpBuilder &builder,
                                         MO::ReduceRMSNormRoPEOp op) {
  buildForOpResult(graph, builder, op,
                   MO::ReduceRMSNormRoPEOp::getOperationName(),
                   op.getOperands(), op.getOutputParamDecls(),
                   /*kgenParams=*/
                   {{"multiply_before_cast", op.getMultiplyBeforeCastAttr()}},
                   /*dataDepInputIdxs=*/{});
  return success();
}

ErrorOrSuccess
Fallback<MO::ReduceGroupNormOp>::build(GraphOp &graph, OpBuilder &builder,
                                       MO::ReduceGroupNormOp op) {
  buildForOpResult(graph, builder, op,
                   MO::ReduceGroupNormOp::getOperationName(), op.getOperands(),
                   op.getOutputParamDecls(),
                   /*kgenParams=*/{},
                   /*dataDepInputIdxs=*/{});
  return success();
}

ErrorOrSuccess
Fallback<MO::ReduceMinAndMaxOp>::build(GraphOp &graph, OpBuilder &builder,
                                       MO::ReduceMinAndMaxOp op) {
  buildForOpResult(graph, builder, op,
                   MO::ReduceMinAndMaxOp::getOperationName(), op.getOperands(),
                   op.getOutputParamDecls(),
                   /*kgenParams=*/{},
                   /*dataDepInputIdxs=*/{1});
  return success();
}

ErrorOrSuccess Fallback<MO::ReduceRMSNormFusedResidualAddOp>::build(
    GraphOp &graph, OpBuilder &builder,
    MO::ReduceRMSNormFusedResidualAddOp op) {
  // Both output and intermediate have the same shape as input.
  buildForOpResult(graph, builder, op.getOutput(),
                   MO::ReduceRMSNormFusedResidualAddOp::getOperationName(),
                   op.getOperands(), op.getOutputParamDecls(),
                   /*kgenParams=*/{},
                   /*dataDepInputIdxs=*/{});
  return success();
}

#undef HAS_FALLBACK
#undef NEEDS_NO_FALLBACK
#undef CALL_FALLBACK

} // namespace

} // namespace M::MO
