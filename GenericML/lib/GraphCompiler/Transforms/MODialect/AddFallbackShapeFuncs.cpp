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
#include "GenericML/Support/Assert.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"

#include <numeric>

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
    void build(GraphOp &graph, OpBuilder &builder, MOOp op);                   \
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
HAS_FALLBACK(MO::ConcatOp)
HAS_FALLBACK(MO::ConcatFromListOp)
HAS_FALLBACK(MO::ConvOp)
HAS_FALLBACK(MO::ConvTransposeOp)
HAS_FALLBACK(MO::GatherOp)
HAS_FALLBACK(MO::GatherNdOp)
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
  void build(GraphOp &graph, OpBuilder &builder, MO::CustomOp op,
             MOGG::MojoLibraryAnalysis &mojoLibraryAnalysis);
};

// Eventually all ops must be accounted for here. They either must have a
// fallback or an explict vouch that they need no fallback.
NEEDS_NO_FALLBACK(MO::MatmulOp)
NEEDS_NO_FALLBACK(MO::Static::BroadcastToOp)

// Creates the fallback for this op if it has parameters still needing decls.
#define CALL_FALLBACK(MOOp)                                                    \
  if (auto asOp = dyn_cast<MOOp>(op.getOperation())) {                         \
    Fallback<MOOp>{}.build(graph, builder, asOp);                              \
    op.setOutputParamDecls({});                                                \
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
    OpBuilder builder{graph.getContext()};

    auto mojoLibraryAnalysis =
        MOGG::MojoLibraryAnalysis::get(graph.getContext(), kernelLibraryPaths);
    if (mojoLibraryAnalysis.isError())
      return signalPassFailure();

    graph.walk([&](ParamDeclarationInterface op) {
      // Shape function was already materialized or never needed.
      if (op.getOutputParamDecls().empty())
        return mlir::WalkResult::advance();

      // NOTE this is a temporary guard that ensures we don't symbolicize the
      // following ops' fallback shape functions in the experimental shape
      // inference path.
      // TODO(#28624) remove `BroadcastToOp` here
      // TODO(#28538) remove `ReshapeOp` here
      // TODO(#28539) remove `ArgNonzeroOp` here
      if (restricted && isa<MO::BroadcastToOp, MO::ReshapeOp, MO::ArgNonzeroOp>(
                            op.getOperation()))
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
      CALL_FALLBACK(MO::ConcatOp)
      CALL_FALLBACK(MO::ConcatFromListOp)
      CALL_FALLBACK(MO::ConvOp)
      CALL_FALLBACK(MO::ConvTransposeOp)
      CALL_FALLBACK(MO::GatherOp)
      CALL_FALLBACK(MO::GatherNdOp)
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
        Fallback<MO::CustomOp>{}.build(graph, builder, asOp,
                                       *mojoLibraryAnalysis);
        op.setOutputParamDecls({});
      }

      return mlir::WalkResult::advance();
    });
  }
};

//===----------------------------------------------------------------------===//
// Fallback helpers
//===----------------------------------------------------------------------===//

// Each fallback needs to produce a parameter for every dimension in the output
// even if those dimensions are static. This simplifies the lowering as without
// doing this we would need the lowering to know how many should be produced and
// to which dimension they should be refering to.
KGEN::ParamDeclArrayAttr getNewParameterMapping(GraphOp &graph,
                                                TensorType outTy) {
  // TODO(#20169)
  GENERICML_ASSERT(outTy.hasRank()) << "dynamic rank isn't supported";

  // New attrs. These are either the exact same as the old attribute or a new
  // attribute
  SmallVector<KGEN::ParamDeclAttr> params;

  // Parameters assumed to be in order. Add the index of each symbolic
  // dimension.
  for (auto dim : outTy.getTypedShapeAttr().getValues()) {
    if (auto asSymbolic = dyn_cast<KGEN::ParamDeclRefAttr>(dim)) {
      // Since this pass happens before shape propagation, for any op that
      // declares output parameters, the parameters in its output shape must
      // come from those output parameters.
      params.push_back(
          KGEN::ParamDeclAttr::get(asSymbolic.getName(), asSymbolic.getType()));
    } else {
      // Add a new placeholder parameter. F for fallback.
      params.push_back(graph.getNewDimParam("F"));
    }
  }
  return KGEN::ParamDeclArrayAttr::get(graph.getContext(), params);
}

//===----------------------------------------------------------------------===//
// Helpers for fallback implementations
//===----------------------------------------------------------------------===//

static void buildForReductionOp(GraphOp &graph, OpBuilder &builder,
                                MO::Reduction op) {
  KGEN::ParamDeclArrayAttr shapeParams =
      getNewParameterMapping(graph, op.getResult().getType());

  // Create the shape function.
  builder.create<MO::InvokeShapeFuncOp>(
      op.getLoc(), op->getOperands(), "reduce_shape", shapeParams,
      /*kgenAttrs*/ DictionaryAttr{},
      /*Data dependent input indices*/ builder.getI32ArrayAttr({1}));
}

static void buildForScatterLikeOp(GraphOp &graph, OpBuilder &builder,
                                  MO::ScatterLike op) {
  KGEN::ParamDeclArrayAttr shapeParams =
      getNewParameterMapping(graph, op.getResult().getType());

  // Create the shape function.
  builder.create<MO::InvokeShapeFuncOp>(
      op.getLoc(), op->getOperands(), "scatter_shape", shapeParams,
      /*kgenAttrs*/ DictionaryAttr{},
      /*Data dependent input indices*/ builder.getI32ArrayAttr({3}));
}

static void buildForScatterNdLikeOp(GraphOp &graph, OpBuilder &builder,
                                    MO::ScatterNdLike op) {
  KGEN::ParamDeclArrayAttr shapeParams =
      getNewParameterMapping(graph, op.getResult().getType());

  // Create the shape function.
  builder.create<MO::InvokeShapeFuncOp>(
      op.getLoc(), op->getOperands(), "scatter_nd_shape", shapeParams,
      /*kgenAttrs*/ DictionaryAttr{},
      /*Data dependent input indices*/ builder.getI32ArrayAttr({}));
}

void buildForOpResult(GraphOp &graph, OpBuilder &builder, TensorValue res,
                      StringRef shapeFuncName, ValueRange operands,
                      ArrayRef<std::pair<StringRef, Attribute>> kgenParams,
                      ArrayRef<int32_t> dataDepInputIdxs) {
  KGEN::ParamDeclArrayAttr shapeParams =
      getNewParameterMapping(graph, res.getType());

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

void Fallback<MO::AvgPoolOp>::build(GraphOp &graph, OpBuilder &builder,
                                    MO::AvgPoolOp op) {
  BoolAttr ceilModeAttr = builder.getBoolAttr(false);
  buildForOpResult(graph, builder, op, "pool_shape", op.getOperands(),
                   /*kgenParams=*/{{"ceil_mode", ceilModeAttr}},
                   /*dataDepInputIdxs=*/{1, 2, 3, 4});
}

void Fallback<MO::AvgPoolCeilOp>::build(GraphOp &graph, OpBuilder &builder,
                                        MO::AvgPoolCeilOp op) {
  BoolAttr ceilModeAttr = builder.getBoolAttr(true);
  buildForOpResult(graph, builder, op, "pool_shape_ceil", op.getOperands(),
                   /*kgenParams=*/{{"ceil_mode", ceilModeAttr}},
                   /*dataDepInputIdxs=*/{1, 2, 3, 4});
}

void Fallback<MO::BatchMatmulOp>::build(GraphOp &graph, OpBuilder &builder,
                                        MO::BatchMatmulOp op) {
  buildForOpResult(graph, builder, op, "batch_matmul_shape", op.getOperands(),
                   /*kgenParams=*/{},
                   /*dataDepInputIdxs=*/{});
}

void Fallback<MO::BottomKOp>::build(GraphOp &graph, OpBuilder &builder,
                                    MO::BottomKOp op) {
  buildForOpResult(graph, builder, op.getValues(), "top_k_shape",
                   op.getOperands(),
                   /*kgenParams=*/{}, /*dataDepInputIdxs=*/{1, 2});
  buildForOpResult(graph, builder, op.getIndices(), "top_k_shape",
                   op.getOperands(),
                   /*kgenParams=*/{}, /*dataDepInputIdxs=*/{1, 2});
}

void Fallback<MO::BroadcastToOp>::build(GraphOp &graph, OpBuilder &builder,
                                        MO::BroadcastToOp op) {
  TensorType outputTy = op.getType();
  GENERICML_ASSERT(outputTy.hasRank()) << "Unranked tensor isn't supported";
  IntegerAttr outputRankAttr = builder.getIndexAttr(outputTy.getRank());

  buildForOpResult(graph, builder, op, "broadcast_to_shape", op.getOperands(),
                   /*kgenParams=*/{{"output_rank", outputRankAttr}},
                   /*dataDepInputIdxs=*/{1});
}

void Fallback<MO::ConcatOp>::build(GraphOp &graph, OpBuilder &builder,
                                   MO::ConcatOp op) {
  // TODO we can't deduce parameter from types of variadic inputs
  auto inputTensorTy = cast<TensorType>(op.getInputs().front().getType());
  auto inputDTypeAttr = mlir::TypeAttr::get(inputTensorTy.getElementType());
  auto inputRankAttr = builder.getIndexAttr(inputTensorTy.getRank());

  buildForOpResult(
      graph, builder, op, "concat_shape", op.getOperands(),
      /*kgenParams=*/
      {{"input_type", inputDTypeAttr}, {"input_rank", inputRankAttr}},
      /*dataDepInputIdxs=*/{0});
}

void Fallback<MO::ConcatFromListOp>::build(GraphOp &graph, OpBuilder &builder,
                                           MO::ConcatFromListOp op) {
  buildForOpResult(graph, builder, op, "concat_from_list_shape",
                   op.getOperands(),
                   /*kgenParams=*/{},
                   /*dataDepInputIdxs=*/{0, 1});
}

void Fallback<MO::ConvOp>::build(GraphOp &graph, OpBuilder &builder,
                                 MO::ConvOp op) {
  buildForOpResult(graph, builder, op, "conv_shape", op.getOperands(),
                   /*kgenParams=*/{}, /*dataDepInputIdxs=*/{2, 3, 4, 5});
}

void Fallback<MO::ConvTransposeOp>::build(GraphOp &graph, OpBuilder &builder,
                                          MO::ConvTransposeOp op) {
  buildForOpResult(graph, builder, op, "conv_transpose_shape", op.getOperands(),
                   /*kgenParams=*/{}, /*dataDepInputIdxs=*/{2, 3, 4, 5});
}

void Fallback<MO::CustomOp>::build(
    GraphOp &graph, OpBuilder &builder, MO::CustomOp op,
    MOGG::MojoLibraryAnalysis &mojoLibraryAnalysis) {

  auto mojoKernel =
      mojoLibraryAnalysis.getRegisteredMojoKernelFor(op.getSymbol());

  // TODO: Poke error through
  GENERICML_ASSERT(mojoKernel) << "Could not find registered mojo kernel.";
  StringRef shapeFunctionName;

  // Mark all inputs by default.
  SmallVector<Value> operands{op.getOperands().begin(), op.getOperands().end()};

  // If we are dealing with a tensor API kernel then the kernel will have the
  // shape function attached.
  if (auto attr =
          dyn_cast_or_null<mlir::UnitAttr>(mojoKernel->getAttr("_sliced"))) {
    auto shapeFuncAttr =
        dyn_cast_or_null<StringAttr>(mojoKernel->getAttr("_has_shape_func"));

    // TODO: Poke error through.
    assert(shapeFuncAttr && "Couldn't find shape function for mojo kernel");
    shapeFunctionName = shapeFuncAttr;

    auto shapeKernel = mojoLibraryAnalysis.getMojoFunctionNamed(shapeFuncAttr);

    auto usedInputAttrs = cast<ArrayAttr>(shapeKernel->getAttr("_used_inputs"));

    // Only pass the arguments which are actually used.
    operands.clear();
    for (APInt i : usedInputAttrs.getAsValueRange<IntegerAttr>())
      operands.push_back(op.getOperands()[i.getLimitedValue()]);

  } else {
    auto shapeFunction =
        mojoLibraryAnalysis.getRegisteredMojoShapeFunctionFor(op.getSymbol());
    assert(shapeFunction);
    shapeFunctionName = shapeFunction.getName();
  }

  SmallVector<int32_t> dataDepInputIdxs(operands.size());
  std::iota(dataDepInputIdxs.begin(), dataDepInputIdxs.end(), 0);

  for (auto [idx, res] : llvm::enumerate(op.getResults())) {
    buildForOpResult(graph, builder, cast<TensorValue>(res), shapeFunctionName,
                     operands,
                     /*kgenParams=*/{{"output_idx", builder.getIndexAttr(idx)}},
                     /*dataDepInputIdxs=*/dataDepInputIdxs);
  }
}

void Fallback<MO::GatherOp>::build(GraphOp &graph, OpBuilder &builder,
                                   MO::GatherOp op) {
  TensorType outputTy = op.getType();
  GENERICML_ASSERT(outputTy.hasRank()) << "Unranked tensor isn't supported";
  IntegerAttr outputRankAttr = builder.getIndexAttr(outputTy.getRank());

  buildForOpResult(graph, builder, op, "gather_shape", op.getOperands(),
                   /*kgenParams=*/{{"output_rank", outputRankAttr}},
                   /*dataDepInputIdxs=*/{2});
}

void Fallback<MO::GatherNdOp>::build(GraphOp &graph, OpBuilder &builder,
                                     MO::GatherNdOp op) {
  TensorType outputTy = op.getType();
  GENERICML_ASSERT(outputTy.hasRank()) << "Unranked tensor isn't supported";
  IntegerAttr outputRankAttr = builder.getIndexAttr(outputTy.getRank());

  buildForOpResult(graph, builder, op, "gather_nd_shape", op.getOperands(),
                   /*kgenParams=*/
                   {
                       {"output_rank", outputRankAttr},
                       {"batch_dims", op.getBatchDims()},
                   },
                   /*dataDepInputIdxs=*/{});
}

void Fallback<MO::LinAlg::SolveOp>::build(GraphOp &graph, OpBuilder &builder,
                                          MO::LinAlg::SolveOp op) {
  buildForOpResult(graph, builder, op, "matrix_solve_shape", op.getOperands(),
                   /*kgenParams=*/{},
                   /*dataDepInputIdxs=*/{});
}

void Fallback<MO::MaxPoolOp>::build(GraphOp &graph, OpBuilder &builder,
                                    MO::MaxPoolOp op) {

  BoolAttr ceilModeAttr = builder.getBoolAttr(false);

  buildForOpResult(graph, builder, op, "pool_shape", op.getOperands(),
                   /*kgenParams=*/{{"ceil_mode", ceilModeAttr}},
                   /*dataDepInputIdxs=*/{1, 2, 3, 4});
}

void Fallback<MO::MaxPoolCeilOp>::build(GraphOp &graph, OpBuilder &builder,
                                        MO::MaxPoolCeilOp op) {

  BoolAttr ceilModeAttr = builder.getBoolAttr(true);

  buildForOpResult(graph, builder, op, "pool_shape_ceil", op.getOperands(),
                   /*kgenParams=*/{{"ceil_mode", ceilModeAttr}},
                   /*dataDepInputIdxs=*/{1, 2, 3, 4});
}

void Fallback<MO::NonMaximumSuppressionOp>::build(
    GraphOp &graph, OpBuilder &builder, MO::NonMaximumSuppressionOp op) {
  SmallVector<Value> shapeFuncInputs({});
  buildForOpResult(graph, builder, op, "non_maximum_suppression_shape",
                   op.getOperands(),
                   /*kgenParams=*/{}, /*dataDepInputIdxs=*/{0, 1, 2, 3, 4});
}

void Fallback<MO::PadConstantOp>::build(GraphOp &graph, OpBuilder &builder,
                                        MO::PadConstantOp op) {
  SmallVector<Value> shapeFuncInputs({op.getInput(), op.getPaddings()});
  buildForOpResult(graph, builder, op, "pad_shape", shapeFuncInputs,
                   /*kgenParams=*/{}, /*dataDepInputIdxs=*/{1});
}

void Fallback<MO::PadReflectOp>::build(GraphOp &graph, OpBuilder &builder,
                                       MO::PadReflectOp op) {
  SmallVector<Value> shapeFuncInputs({op.getInput(), op.getPaddings()});
  buildForOpResult(graph, builder, op, "pad_shape", shapeFuncInputs,
                   /*kgenParams=*/{}, /*dataDepInputIdxs=*/{1});
}

void Fallback<MO::PadRepeatOp>::build(GraphOp &graph, OpBuilder &builder,
                                      MO::PadRepeatOp op) {
  SmallVector<Value> shapeFuncInputs({op.getInput(), op.getPaddings()});
  buildForOpResult(graph, builder, op, "pad_shape", shapeFuncInputs,
                   /*kgenParams=*/{}, /*dataDepInputIdxs=*/{1});
}

void Fallback<MO::RandomNormalOp>::build(GraphOp &graph, OpBuilder &builder,
                                         MO::RandomNormalOp op) {
  SmallVector<Value> shapeFuncInputs({op.getOperand(0)});
  buildForOpResult(graph, builder, op, "random_shape", shapeFuncInputs,
                   /*kgenParams=*/{}, /*dataDepInputIdxs=*/{0});
}

void Fallback<MO::RangeOp>::build(GraphOp &graph, OpBuilder &builder,
                                  MO::RangeOp op) {
  buildForOpResult(graph, builder, op, "range_shape", op->getOperands(),
                   /*kgenParams=*/{}, /*dataDepInputIdxs=*/{0, 1, 2});
}

// Reduction ops

#define BUILD_REDUCTION_OP(OP)                                                 \
  void Fallback<OP>::build(GraphOp &graph, OpBuilder &builder, OP op) {        \
    buildForReductionOp(graph, builder, op);                                   \
  }

BUILD_REDUCTION_OP(MO::ArgMaxOp)
BUILD_REDUCTION_OP(MO::ArgMinOp)
BUILD_REDUCTION_OP(MO::MeanOp)
BUILD_REDUCTION_OP(MO::ReduceAddOp)
BUILD_REDUCTION_OP(MO::ReduceMulOp)
BUILD_REDUCTION_OP(MO::ReduceMaxOp)
BUILD_REDUCTION_OP(MO::ReduceMinOp)

#undef BUILD_REDUCTION_OP

void Fallback<MO::ReshapeOp>::build(GraphOp &graph, OpBuilder &builder,
                                    MO::ReshapeOp op) {
  TensorType outputTy = op.getType();
  GENERICML_ASSERT(outputTy.hasRank()) << "Unranked tensor isn't supported";
  IntegerAttr outputRankAttr = builder.getIndexAttr(outputTy.getRank());

  buildForOpResult(graph, builder, op, "reshape_shape", op.getOperands(),
                   /*kgenParams=*/{{"output_rank", outputRankAttr}},
                   /*dataDepInputIdxs=*/{1});
}

void Fallback<MO::ResizeNearestOp>::build(GraphOp &graph, OpBuilder &builder,
                                          MO::ResizeNearestOp op) {
  buildForOpResult(graph, builder, op, "resize_shape", op.getOperands(),
                   /*kgenParams=*/{}, /*dataDepInputIdxs=*/{1});
}

void Fallback<MO::ResizeLinearOp>::build(GraphOp &graph, OpBuilder &builder,
                                         MO::ResizeLinearOp op) {
  buildForOpResult(graph, builder, op, "resize_shape", op.getOperands(),
                   /*kgenParams=*/{}, /*dataDepInputIdxs=*/{1});
}

void Fallback<MO::RoiAlignOp>::build(GraphOp &graph, OpBuilder &builder,
                                     MO::RoiAlignOp op) {
  // The shape function doesn't need spatial_scale and sampling_ratio.
  buildForOpResult(graph, builder, op, "roi_align_shape",
                   op.getOperands().take_front(4),
                   /*kgenParams=*/{}, /*dataDepInputIdxs=*/{2, 3});
}

// ScatterLike ops

#define BUILD_SCATTER_LIKE_OP(OP)                                              \
  void Fallback<OP>::build(GraphOp &graph, OpBuilder &builder, OP op) {        \
    buildForScatterLikeOp(graph, builder, op);                                 \
  }

BUILD_SCATTER_LIKE_OP(MO::ScatterOp)
BUILD_SCATTER_LIKE_OP(MO::ScatterAddOp)
BUILD_SCATTER_LIKE_OP(MO::ScatterMaxOp)
BUILD_SCATTER_LIKE_OP(MO::ScatterMinOp)
BUILD_SCATTER_LIKE_OP(MO::ScatterMulOp)

#undef BUILD_SCATTER_LIKE_OP

// ScatterNdLike ops

#define BUILD_SCATTER_ND_LIKE_OP(OP)                                           \
  void Fallback<OP>::build(GraphOp &graph, OpBuilder &builder, OP op) {        \
    buildForScatterNdLikeOp(graph, builder, op);                               \
  }

BUILD_SCATTER_ND_LIKE_OP(MO::ScatterNdOp)
BUILD_SCATTER_ND_LIKE_OP(MO::ScatterNdAddOp)
BUILD_SCATTER_ND_LIKE_OP(MO::ScatterNdMaxOp)
BUILD_SCATTER_ND_LIKE_OP(MO::ScatterNdMinOp)
BUILD_SCATTER_ND_LIKE_OP(MO::ScatterNdMulOp)

#undef BUILD_SCATTER_ND_LIKE_OP

void Fallback<MO::SliceOp>::build(GraphOp &graph, OpBuilder &builder,
                                  MO::SliceOp op) {
  buildForOpResult(graph, builder, op, "slice_shape", op.getOperands(),
                   /*kgenParams=*/{}, /*dataDepInputIdxs=*/{1, 2, 3});
}

void Fallback<MO::SplitOp>::build(GraphOp &graph, OpBuilder &builder,
                                  MO::SplitOp op) {
  for (auto [idx, res] : llvm::enumerate(op.getResults())) {
    buildForOpResult(graph, builder, cast<TensorValue>(res),
                     "split_ith_output_shape", op.getOperands(),
                     /*kgenParams=*/{{"output_idx", builder.getIndexAttr(idx)}},
                     /*dataDepInputIdxs=*/{1, 2});
  }
}

void Fallback<MO::SqueezeShapeOp>::build(GraphOp &graph, OpBuilder &builder,
                                         MO::SqueezeShapeOp op) {
  buildForOpResult(graph, builder, op, "squeeze_shape_shape", op.getOperands(),
                   /*kgenParams=*/{}, /*dataDepInputIdxs=*/{});
}

void Fallback<MO::TopKOp>::build(GraphOp &graph, OpBuilder &builder,
                                 MO::TopKOp op) {
  buildForOpResult(graph, builder, op.getValues(), "top_k_shape",
                   op.getOperands(),
                   /*kgenParams=*/{}, /*dataDepInputIdxs=*/{1, 2});
  buildForOpResult(graph, builder, op.getIndices(), "top_k_shape",
                   op.getOperands(),
                   /*kgenParams=*/{}, /*dataDepInputIdxs=*/{1, 2});
}

void Fallback<MO::TransposeOp>::build(GraphOp &graph, OpBuilder &builder,
                                      MO::TransposeOp op) {
  buildForOpResult(graph, builder, op, "transpose_shape", op.getOperands(),
                   /*kgenParams=*/{}, /*dataDepInputIdxs=*/{1});
}

void Fallback<MO::UnsqueezeShapeOp>::build(GraphOp &graph, OpBuilder &builder,
                                           MO::UnsqueezeShapeOp op) {
  buildForOpResult(graph, builder, op, "unsqueeze_shape_shape",
                   op.getOperands(),
                   /*kgenParams=*/{}, /*dataDepInputIdxs=*/{});
}

void Fallback<MO::ArgNonzeroOp>::build(GraphOp &graph, OpBuilder &builder,
                                       MO::ArgNonzeroOp op) {
  buildForOpResult(graph, builder, op, "arg_nonzero_shape", op->getOperands(),
                   /*kgenParams=*/{}, /*dataDepInputIdxs=*/{0});
}

void Fallback<MO::TileOp>::build(GraphOp &graph, OpBuilder &builder,
                                 MO::TileOp op) {
  buildForOpResult(graph, builder, op, "tile_shape", op.getOperands(),
                   /*kgenParams=*/{}, /*dataDepInputIdxs=*/{1});
}

} // namespace

} // namespace M::MO

#undef HAS_FALLBACK
#undef NEEDS_NO_FALLBACK
#undef CALL_FALLBACK
