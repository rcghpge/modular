//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/MODialect/MOInterfaces.h"
#include "GenericML/GraphCompiler/MODialect/MOOps.h"
#include "GenericML/GraphCompiler/MOGGDialect/Analysis/MojoLibraryAnalysis.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGPasses.h"
#include "GenericML/GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "GenericML/GraphCompiler/Transforms/MODialect/MOTransforms.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"

#include <numeric>

#define DEBUG_TYPE "add-fallback-shape-funcs"

namespace M::MO {

#define GEN_PASS_DEF_ADDFALLBACKSHAPEFUNCTIONS
#include "GenericML/GraphCompiler/Transforms/MODialect/MOTransforms.h.inc"

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
HAS_FALLBACK(MO::ConcatFromListOp)
HAS_FALLBACK(MO::ConvOp)
HAS_FALLBACK(MO::ConvTransposeOp)
HAS_FALLBACK(MO::GatherOp)
HAS_FALLBACK(MO::GatherNdOp)
HAS_FALLBACK(MO::LayerNormOp)
HAS_FALLBACK(MO::LinAlg::SolveOp)
HAS_FALLBACK(MO::MaxPoolOp)
HAS_FALLBACK(MO::MaxPoolCeilOp)
HAS_FALLBACK(MO::MeanOp)
HAS_FALLBACK(MO::NonMaximumSuppressionOp)
HAS_FALLBACK(MO::PadConstantOp)
HAS_FALLBACK(MO::PadReflectOp)
HAS_FALLBACK(MO::PadRepeatOp)
HAS_FALLBACK(MO::RandomNormalOp)
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
                       KGEN::GeneratorOp mojoKernel,
                       MOGG::MojoLibraryAnalysis &mojoLibraryAnalysis);
};

// Eventually all ops must be accounted for here. They either must have a
// fallback or an explict vouch that they need no fallback.
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

  AddFallbackShapeFunctions() : AddFallbackShapeFunctionsBase() {}

  AddFallbackShapeFunctions(const AddFallbackShapeFunctionsOptions &opts)
      : AddFallbackShapeFunctionsBase(opts) {}

  void runOnOperation() override {
    MO::GraphOp graph = getOperation();
    MLIRContext *ctx = graph.getContext();
    OpBuilder builder{ctx};

    MOGG::MojoLibraryAnalysis *cachedAnalysis = nullptr;

    // This pass actually only needs to touch the analysis for custom ops, most
    // ops don't use it.
    auto getLazyAnalysis = [&]() -> MOGG::MojoLibraryAnalysis * {
      if (cachedAnalysis)
        return cachedAnalysis;

      auto &analysis = getAnalysis<MOGG::LoadMojoAnalysisPass>();
      ErrorOr<MOGG::MojoLibraryAnalysis *> errOr =
          analysis.initIfNeeded(ctx, kernelLibraryPaths);
      if (errOr.isError())
        return nullptr;
      cachedAnalysis = errOr.takeValue();
      return cachedAnalysis;
    };

    // We will never modify the state.
    markAnalysesPreserved<MOGG::LoadMojoAnalysisPass>();

    WalkResult result = graph.walk([&](ParamDeclarationInterface op) {
      // Shape function was already materialized or never needed.
      if (op.getOutputParamDecls().empty())
        return mlir::WalkResult::advance();

      // NOTE for most ops, we can materialize their shape function calls once
      // and for all, since if there is any shape-changing rewrite, we can
      // generally compose the new shape out of existing shape parameters.
      //
      // However, there are certain ops like `mo.arg_nonzero` where
      // 1. it's never possible to resolve its output shape at compile time
      // 2. we have rewrite (e.g., sink broadcast through `mo.arg_nonzero`) that
      // changes its output shape.
      //
      // For these ops, we avoid materializing their shape function call until
      // the very end.
      //
      // TODO(GRA-627) move this pass after `PropagateShapesPass` and remove
      // `ShapeMaterialization` here.
      if (restricted &&
          isa<MO::ArgNonzeroOp, ShapeMaterialization>(op.getOperation()))
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
      CALL_FALLBACK(MO::ConcatFromListOp)
      CALL_FALLBACK(MO::ConvOp)
      CALL_FALLBACK(MO::ConvTransposeOp)
      CALL_FALLBACK(MO::GatherOp)
      CALL_FALLBACK(MO::GatherNdOp)
      CALL_FALLBACK(MO::LayerNormOp)
      CALL_FALLBACK(MO::LinAlg::SolveOp)
      CALL_FALLBACK(MO::MaxPoolOp)
      CALL_FALLBACK(MO::MaxPoolCeilOp)
      CALL_FALLBACK(MO::MeanOp)
      CALL_FALLBACK(MO::NonMaximumSuppressionOp)
      CALL_FALLBACK(MO::PadConstantOp)
      CALL_FALLBACK(MO::PadReflectOp)
      CALL_FALLBACK(MO::PadRepeatOp)
      CALL_FALLBACK(MO::RandomNormalOp)
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
        MOGG::MojoLibraryAnalysis *analysis = getLazyAnalysis();
        if (!analysis)
          signalPassFailure();

        auto mojoKernel =
            analysis->getRegisteredMojoKernelFor(asOp.getSymbol());
        if (!mojoKernel) {
          asOp.emitError() << "Could not find registered mojo kernel named "
                           << asOp.getSymbol();
          return mlir::WalkResult::interrupt();
        }

        // Skip inplace kernels.
        if (mojoKernel->hasAttr(KGEN::MOGGPreElab::ALLOCS_ATTR))
          return mlir::WalkResult::skip();

        auto result = Fallback<MO::CustomOp>{}.build(graph, builder, asOp,
                                                     mojoKernel, *analysis);
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
// to which dimension they should be refering to.
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
  builder.create<MO::InvokeShapeFuncOp>(
      op.getLoc(), op->getOperands(), "reduce_shape", shapeParams,
      /*kgenAttrs*/ DictionaryAttr{},
      /*Data dependent input indices*/ builder.getI32ArrayAttr({1}));

  return success();
}

static ErrorOrSuccess buildForScatterLikeOp(GraphOp &graph, OpBuilder &builder,
                                            MO::ScatterLike op) {
  KGEN::ParamDeclArrayAttr shapeParams = getNewParameterMapping(
      graph, op.getResult().getType(), op.getOutputParamDecls());

  // Create the shape function.
  builder.create<MO::InvokeShapeFuncOp>(
      op.getLoc(), op->getOperands(), "scatter_shape", shapeParams,
      /*kgenAttrs*/ DictionaryAttr{},
      /*Data dependent input indices*/ builder.getI32ArrayAttr({3}));

  return success();
}

static ErrorOrSuccess buildForScatterNdLikeOp(GraphOp &graph,
                                              OpBuilder &builder,
                                              MO::ScatterNdLike op) {
  KGEN::ParamDeclArrayAttr shapeParams = getNewParameterMapping(
      graph, op.getResult().getType(), op.getOutputParamDecls());

  // Create the shape function.
  builder.create<MO::InvokeShapeFuncOp>(
      op.getLoc(), op->getOperands(), "scatter_nd_shape", shapeParams,
      /*kgenAttrs*/ DictionaryAttr{},
      /*Data dependent input indices*/ builder.getI32ArrayAttr({}));

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
  builder.create<MO::InvokeShapeFuncOp>(
      loc, operands, shapeFuncName, shapeParams, kgenParamAttrs,
      builder.getI32ArrayAttr(dataDepInputIdxs));
}

//===----------------------------------------------------------------------===//
// Actual fallback implementations
//===----------------------------------------------------------------------===//

ErrorOrSuccess Fallback<MO::AvgPoolOp>::build(GraphOp &graph,
                                              OpBuilder &builder,
                                              MO::AvgPoolOp op) {
  buildForOpResult(graph, builder, op, "pool_shape", op.getOperands(),
                   op.getOutputParamDecls(),
                   /*kgenParams=*/{},
                   /*dataDepInputIdxs=*/op.getDataDependentInputIndices());
  return success();
}

ErrorOrSuccess Fallback<MO::AvgPoolCeilOp>::build(GraphOp &graph,
                                                  OpBuilder &builder,
                                                  MO::AvgPoolCeilOp op) {
  buildForOpResult(graph, builder, op, "pool_shape_ceil", op.getOperands(),
                   op.getOutputParamDecls(),
                   /*kgenParams=*/{},
                   /*dataDepInputIdxs=*/op.getDataDependentInputIndices());
  return success();
}

ErrorOrSuccess Fallback<MO::BatchMatmulOp>::build(GraphOp &graph,
                                                  OpBuilder &builder,
                                                  MO::BatchMatmulOp op) {
  buildForOpResult(graph, builder, op, "batch_matmul_shape", op.getOperands(),
                   op.getOutputParamDecls(),
                   /*kgenParams=*/{}, op.getDataDependentInputIndices());
  return success();
}

ErrorOrSuccess Fallback<MO::BottomKOp>::build(GraphOp &graph,
                                              OpBuilder &builder,
                                              MO::BottomKOp op) {
  buildForOpResult(graph, builder, op.getValues(), "top_k_shape",
                   op.getOperands(), op.getOutputParamDecls(),
                   /*kgenParams=*/{}, /*dataDepInputIdxs=*/{1, 2});
  // indices() guaranteed by verifier to have the same shape as values()
  return success();
}

ErrorOrSuccess Fallback<MO::BroadcastToOp>::build(GraphOp &graph,
                                                  OpBuilder &builder,
                                                  MO::BroadcastToOp op) {
  buildForOpResult(graph, builder, op, "broadcast_to_shape", op.getOperands(),
                   op.getOutputParamDecls(),
                   /*kgenParams=*/{},
                   /*dataDepInputIdxs=*/{1});
  return success();
}

ErrorOrSuccess Fallback<MO::BroadcastShapeOp>::build(GraphOp &graph,
                                                     OpBuilder &builder,
                                                     MO::BroadcastShapeOp op) {
  buildForOpResult(graph, builder, op, "broadcast_shape_shape",
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

  buildForOpResult(
      graph, builder, op, "concat_shape", op.getOperands(),
      op.getOutputParamDecls(),
      /*kgenParams=*/
      {{"input_type", inputDTypeAttr}, {"input_rank", inputRankAttr}},
      op.getDataDependentInputIndices());
  return success();
}

ErrorOrSuccess Fallback<MO::ConcatFromListOp>::build(GraphOp &graph,
                                                     OpBuilder &builder,
                                                     MO::ConcatFromListOp op) {
  buildForOpResult(graph, builder, op, "concat_from_list_shape",
                   op.getOperands(), op.getOutputParamDecls(),
                   /*kgenParams=*/{},
                   /*dataDepInputIdxs=*/{0, 1});
  return success();
}

ErrorOrSuccess Fallback<MO::ConvOp>::build(GraphOp &graph, OpBuilder &builder,
                                           MO::ConvOp op) {
  buildForOpResult(graph, builder, op, "conv_shape", op.getOperands(),
                   op.getOutputParamDecls(),
                   /*kgenParams=*/{}, op.getDataDependentInputIndices());
  return success();
}

ErrorOrSuccess Fallback<MO::ConvTransposeOp>::build(GraphOp &graph,
                                                    OpBuilder &builder,
                                                    MO::ConvTransposeOp op) {
  buildForOpResult(graph, builder, op, "conv_transpose_shape", op.getOperands(),
                   op.getOutputParamDecls(),
                   /*kgenParams=*/{},
                   /*dataDepInputIdxs=*/op.getDataDependentInputIndices());
  return success();
}

ErrorOrSuccess
Fallback<MO::CustomOp>::build(GraphOp &graph, OpBuilder &builder,
                              MO::CustomOp op, KGEN::GeneratorOp mojoKernel,
                              MOGG::MojoLibraryAnalysis &mojoLibraryAnalysis) {
  // Mark all inputs by default.
  SmallVector<Value> operands{op.getOperands().begin(), op.getOperands().end()};

  auto shapeFunction =
      mojoLibraryAnalysis.getRegisteredMojoShapeFunctionFor(op.getSymbol());
  if (!shapeFunction)
    return Error("Couldn't find shape function for mojo kernel");
  StringRef shapeFunctionName = shapeFunction.getName();

  SmallVector<int32_t> dataDepInputIdxs(operands.size());
  std::iota(dataDepInputIdxs.begin(), dataDepInputIdxs.end(), 0);

  for (auto [idx, res] : llvm::enumerate(op.getResults())) {
    buildForOpResult(graph, builder, cast<TensorValue>(res), shapeFunctionName,
                     operands, op.getOutputParamDecls(),
                     /*kgenParams=*/{{"output_idx", builder.getIndexAttr(idx)}},
                     /*dataDepInputIdxs=*/dataDepInputIdxs);
  }

  return success();
}

ErrorOrSuccess Fallback<MO::GatherOp>::build(GraphOp &graph, OpBuilder &builder,
                                             MO::GatherOp op) {
  buildForOpResult(graph, builder, op, "gather_shape", op.getOperands(),
                   op.getOutputParamDecls(),
                   /*kgenParams=*/{},
                   /*dataDepInputIdxs=*/op.getDataDependentInputIndices());
  return success();
}

ErrorOrSuccess Fallback<MO::GatherNdOp>::build(GraphOp &graph,
                                               OpBuilder &builder,
                                               MO::GatherNdOp op) {
  buildForOpResult(graph, builder, op, "gather_nd_shape", op.getOperands(),
                   op.getOutputParamDecls(),
                   /*kgenParams=*/
                   {
                       {"batch_dims", op.getBatchDims()},
                   },
                   op.getDataDependentInputIndices());
  return success();
}

ErrorOrSuccess Fallback<MO::LayerNormOp>::build(GraphOp &graph,
                                                OpBuilder &builder,
                                                MO::LayerNormOp op) {
  buildForOpResult(graph, builder, op, "layer_norm_shape", op.getOperands(),
                   op.getOutputParamDecls(),
                   /*kgenParams=*/{},
                   /*dataDepInputIdxs=*/{});
  return success();
}

ErrorOrSuccess Fallback<MO::LinAlg::SolveOp>::build(GraphOp &graph,
                                                    OpBuilder &builder,
                                                    MO::LinAlg::SolveOp op) {
  buildForOpResult(graph, builder, op, "matrix_solve_shape", op.getOperands(),
                   op.getOutputParamDecls(),
                   /*kgenParams=*/{}, op.getDataDependentInputIndices());
  return success();
}

ErrorOrSuccess Fallback<MO::MaxPoolOp>::build(GraphOp &graph,
                                              OpBuilder &builder,
                                              MO::MaxPoolOp op) {

  buildForOpResult(graph, builder, op, "pool_shape", op.getOperands(),
                   op.getOutputParamDecls(),
                   /*kgenParams=*/{},
                   /*dataDepInputIdxs=*/op.getDataDependentInputIndices());
  return success();
}

ErrorOrSuccess Fallback<MO::MaxPoolCeilOp>::build(GraphOp &graph,
                                                  OpBuilder &builder,
                                                  MO::MaxPoolCeilOp op) {

  buildForOpResult(graph, builder, op, "pool_shape_ceil", op.getOperands(),
                   op.getOutputParamDecls(),
                   /*kgenParams=*/{},
                   /*dataDepInputIdxs=*/op.getDataDependentInputIndices());
  return success();
}

ErrorOrSuccess
Fallback<MO::NonMaximumSuppressionOp>::build(GraphOp &graph, OpBuilder &builder,
                                             MO::NonMaximumSuppressionOp op) {
  SmallVector<Value> shapeFuncInputs({});
  buildForOpResult(graph, builder, op, "non_maximum_suppression_shape",
                   op.getOperands(), op.getOutputParamDecls(),
                   /*kgenParams=*/{}, /*dataDepInputIdxs=*/{0, 1, 2, 3, 4});
  return success();
}

ErrorOrSuccess Fallback<MO::PadConstantOp>::build(GraphOp &graph,
                                                  OpBuilder &builder,
                                                  MO::PadConstantOp op) {
  SmallVector<Value> shapeFuncInputs({op.getInput(), op.getPaddings()});
  buildForOpResult(graph, builder, op, "pad_shape", shapeFuncInputs,
                   op.getOutputParamDecls(),
                   /*kgenParams=*/{}, op.getDataDependentInputIndices());
  return success();
}

ErrorOrSuccess Fallback<MO::PadReflectOp>::build(GraphOp &graph,
                                                 OpBuilder &builder,
                                                 MO::PadReflectOp op) {
  SmallVector<Value> shapeFuncInputs({op.getInput(), op.getPaddings()});
  buildForOpResult(graph, builder, op, "pad_shape", shapeFuncInputs,
                   op.getOutputParamDecls(),
                   /*kgenParams=*/{}, op.getDataDependentInputIndices());
  return success();
}

ErrorOrSuccess Fallback<MO::PadRepeatOp>::build(GraphOp &graph,
                                                OpBuilder &builder,
                                                MO::PadRepeatOp op) {
  SmallVector<Value> shapeFuncInputs({op.getInput(), op.getPaddings()});
  buildForOpResult(graph, builder, op, "pad_shape", shapeFuncInputs,
                   op.getOutputParamDecls(),
                   /*kgenParams=*/{}, op.getDataDependentInputIndices());
  return success();
}

ErrorOrSuccess Fallback<MO::RandomNormalOp>::build(GraphOp &graph,
                                                   OpBuilder &builder,
                                                   MO::RandomNormalOp op) {
  SmallVector<Value> shapeFuncInputs({op.getOperand(0)});
  buildForOpResult(graph, builder, op, "random_shape", shapeFuncInputs,
                   op.getOutputParamDecls(),
                   /*kgenParams=*/{}, /*dataDepInputIdxs=*/{0});
  return success();
}

ErrorOrSuccess Fallback<MO::RangeOp>::build(GraphOp &graph, OpBuilder &builder,
                                            MO::RangeOp op) {
  buildForOpResult(graph, builder, op, "range_shape", op->getOperands(),
                   op.getOutputParamDecls(),
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
  buildForOpResult(graph, builder, op, "reshape_shape", op.getOperands(),
                   op.getOutputParamDecls(),
                   /*kgenParams=*/{}, op.getDataDependentInputIndices());
  return success();
}

ErrorOrSuccess Fallback<MO::ResizeNearestOp>::build(GraphOp &graph,
                                                    OpBuilder &builder,
                                                    MO::ResizeNearestOp op) {
  buildForOpResult(graph, builder, op, "resize_shape", op.getOperands(),
                   op.getOutputParamDecls(),
                   /*kgenParams=*/{}, /*dataDepInputIdxs=*/{1});
  return success();
}

ErrorOrSuccess Fallback<MO::ResizeLinearOp>::build(GraphOp &graph,
                                                   OpBuilder &builder,
                                                   MO::ResizeLinearOp op) {
  buildForOpResult(graph, builder, op, "resize_shape", op.getOperands(),
                   op.getOutputParamDecls(),
                   /*kgenParams=*/{}, /*dataDepInputIdxs=*/{1});
  return success();
}

ErrorOrSuccess Fallback<MO::RoiAlignOp>::build(GraphOp &graph,
                                               OpBuilder &builder,
                                               MO::RoiAlignOp op) {
  // The shape function doesn't need spatial_scale and sampling_ratio.
  buildForOpResult(graph, builder, op, "roi_align_shape",
                   op.getOperands().take_front(4), op.getOutputParamDecls(),
                   /*kgenParams=*/{}, /*dataDepInputIdxs=*/{2, 3});
  return success();
}

// ScatterLike ops

#define BUILD_SCATTER_LIKE_OP(OP)                                              \
  ErrorOrSuccess Fallback<OP>::build(GraphOp &graph, OpBuilder &builder,       \
                                     OP op) {                                  \
    return buildForScatterLikeOp(graph, builder, op);                          \
  }

BUILD_SCATTER_LIKE_OP(MO::ScatterOp)
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
  buildForOpResult(graph, builder, op, "slice_shape", op.getOperands(),
                   op.getOutputParamDecls(),
                   /*kgenParams=*/{}, /*dataDepInputIdxs=*/{1, 2, 3});
  return success();
}

ErrorOrSuccess Fallback<MO::SplitOp>::build(GraphOp &graph, OpBuilder &builder,
                                            MO::SplitOp op) {
  for (auto [idx, res] : llvm::enumerate(op.getResults())) {
    buildForOpResult(graph, builder, cast<TensorValue>(res),
                     "split_ith_output_shape", op.getOperands(),
                     op.getOutputParamDecls(),
                     /*kgenParams=*/{{"output_idx", builder.getIndexAttr(idx)}},
                     /*dataDepInputIdxs=*/{1, 2});
  }
  return success();
}

ErrorOrSuccess Fallback<MO::SqueezeShapeOp>::build(GraphOp &graph,
                                                   OpBuilder &builder,
                                                   MO::SqueezeShapeOp op) {
  buildForOpResult(graph, builder, op, "squeeze_shape_shape", op.getOperands(),
                   op.getOutputParamDecls(),
                   /*kgenParams=*/{}, /*dataDepInputIdxs=*/{});
  return success();
}

ErrorOrSuccess Fallback<MO::TopKOp>::build(GraphOp &graph, OpBuilder &builder,
                                           MO::TopKOp op) {
  buildForOpResult(graph, builder, op.getValues(), "top_k_shape",
                   op.getOperands(), op.getOutputParamDecls(),
                   /*kgenParams=*/{}, /*dataDepInputIdxs=*/{1, 2});

  // indices() guaranteed by verifier to have the same shape as values()
  return success();
}

ErrorOrSuccess Fallback<MO::TransposeOp>::build(GraphOp &graph,
                                                OpBuilder &builder,
                                                MO::TransposeOp op) {
  buildForOpResult(graph, builder, op, "transpose_shape", op.getOperands(),
                   op.getOutputParamDecls(),
                   /*kgenParams=*/{}, /*dataDepInputIdxs=*/{1});
  return success();
}

ErrorOrSuccess Fallback<MO::UnsqueezeShapeOp>::build(GraphOp &graph,
                                                     OpBuilder &builder,
                                                     MO::UnsqueezeShapeOp op) {
  buildForOpResult(graph, builder, op, "unsqueeze_shape_shape",
                   op.getOperands(), op.getOutputParamDecls(),
                   /*kgenParams=*/{}, /*dataDepInputIdxs=*/{});
  return success();
}

ErrorOrSuccess Fallback<MO::ArgNonzeroOp>::build(GraphOp &graph,
                                                 OpBuilder &builder,
                                                 MO::ArgNonzeroOp op) {
  buildForOpResult(graph, builder, op, "arg_nonzero_shape", op->getOperands(),
                   op.getOutputParamDecls(),
                   /*kgenParams=*/{}, /*dataDepInputIdxs=*/{0});
  return success();
}

ErrorOrSuccess Fallback<MO::TileOp>::build(GraphOp &graph, OpBuilder &builder,
                                           MO::TileOp op) {
  buildForOpResult(graph, builder, op, "tile_shape", op.getOperands(),
                   op.getOutputParamDecls(),
                   /*kgenParams=*/{}, /*dataDepInputIdxs=*/{1});
  return success();
}

} // namespace

} // namespace M::MO

#undef HAS_FALLBACK
#undef NEEDS_NO_FALLBACK
#undef CALL_FALLBACK
