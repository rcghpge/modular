//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/MODialect/MOInterfaces.h"
#include "GenericML/GraphCompiler/MODialect/MOOps.h"
#include "GenericML/GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "GenericML/GraphCompiler/Transforms/MODialect/MOTransforms.h"

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
HAS_FALLBACK(MO::BottomKOp)
HAS_FALLBACK(MO::BroadcastToOp)
HAS_FALLBACK(MO::ConcatOp)
HAS_FALLBACK(MO::ConvOp)
HAS_FALLBACK(MO::GatherOp)
HAS_FALLBACK(MO::MaxPoolOp)
HAS_FALLBACK(MO::MeanOp)
HAS_FALLBACK(MO::PadOp)
HAS_FALLBACK(MO::RangeOp)
HAS_FALLBACK(MO::ReduceAddOp)
HAS_FALLBACK(MO::ReduceMaxOp)
HAS_FALLBACK(MO::ReduceMinOp)
HAS_FALLBACK(MO::ReduceMulOp)
HAS_FALLBACK(MO::ReshapeOp)
HAS_FALLBACK(MO::ScatterOp)
HAS_FALLBACK(MO::ScatterAddOp)
HAS_FALLBACK(MO::ScatterMaxOp)
HAS_FALLBACK(MO::ScatterMinOp)
HAS_FALLBACK(MO::ScatterMulOp)
HAS_FALLBACK(MO::SliceOp)
HAS_FALLBACK(MO::SplitOp)
HAS_FALLBACK(MO::TopKOp)
HAS_FALLBACK(MO::TransposeOp)
HAS_FALLBACK(MO::ArgNonzeroOp)

// Eventually all ops must be accounted for here. They either must have a
// fallback or an explict vouch that they need no fallback.
NEEDS_NO_FALLBACK(MO::BatchMatmulOp)
NEEDS_NO_FALLBACK(MO::MatmulOp)
NEEDS_NO_FALLBACK(MO::ScatterNdOp)
NEEDS_NO_FALLBACK(MO::Static::BroadcastToOp)
NEEDS_NO_FALLBACK(MO::GatherNdOp)

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

  void runOnOperation() override {
    MO::GraphOp graph = getOperation();
    OpBuilder builder{graph.getContext()};

    graph.walk([&](ParamDeclarationInterface op) {
      // Shape function was already materialized or never needed.
      if (op.getOutputParamDecls().empty())
        return mlir::WalkResult::advance();

      // Call the fallbacks.
      builder.setInsertionPoint(op);
      CALL_FALLBACK(MO::ArgMaxOp)
      CALL_FALLBACK(MO::ArgMinOp)
      CALL_FALLBACK(MO::AvgPoolOp)
      CALL_FALLBACK(MO::BottomKOp)
      CALL_FALLBACK(MO::BroadcastToOp)
      CALL_FALLBACK(MO::ConcatOp)
      CALL_FALLBACK(MO::ConvOp)
      CALL_FALLBACK(MO::GatherOp)
      CALL_FALLBACK(MO::MaxPoolOp)
      CALL_FALLBACK(MO::MeanOp)
      CALL_FALLBACK(MO::PadOp)
      CALL_FALLBACK(MO::RangeOp)
      CALL_FALLBACK(MO::ReduceAddOp)
      CALL_FALLBACK(MO::ReduceMaxOp)
      CALL_FALLBACK(MO::ReduceMinOp)
      CALL_FALLBACK(MO::ReduceMulOp)
      CALL_FALLBACK(MO::ReshapeOp)
      CALL_FALLBACK(MO::ScatterOp)
      CALL_FALLBACK(MO::ScatterAddOp)
      CALL_FALLBACK(MO::ScatterMaxOp)
      CALL_FALLBACK(MO::ScatterMinOp)
      CALL_FALLBACK(MO::ScatterMulOp)
      CALL_FALLBACK(MO::SliceOp)
      CALL_FALLBACK(MO::SplitOp)
      CALL_FALLBACK(MO::TopKOp)
      CALL_FALLBACK(MO::TransposeOp)
      CALL_FALLBACK(MO::ArgNonzeroOp)

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
  assert(outTy.hasRank() && "dynamic rank isn't supported");

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
      op.getLoc(), op->getOperands(), "scatter_elements_shape", shapeParams,
      /*kgenAttrs*/ DictionaryAttr{},
      /*Data dependent input indices*/ builder.getI32ArrayAttr({3}));
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
  buildForOpResult(graph, builder, op, "pool_shape", op.getOperands(),
                   /*kgenParams=*/{},
                   /*dataDepInputIdxs=*/{1, 2, 3, 4});
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
  buildForOpResult(graph, builder, op, "broadcast_to_shape", op.getOperands(),
                   /*kgenParams=*/{}, /*dataDepInputIdxs=*/{1});
}

void Fallback<MO::ConcatOp>::build(GraphOp &graph, OpBuilder &builder,
                                   MO::ConcatOp op) {
  // TODO we can't deduce parameter from types of variadic inputs
  Type inputDType =
      cast<TensorType>(op.getInputs().front().getType()).getElementType();
  auto inputDTypeAttr = mlir::TypeAttr::get(inputDType);

  buildForOpResult(graph, builder, op, "concat_shape", op.getOperands(),
                   /*kgenParams=*/{{"input_type", inputDTypeAttr}},
                   /*dataDepInputIdxs=*/{0});
}

void Fallback<MO::ConvOp>::build(GraphOp &graph, OpBuilder &builder,
                                 MO::ConvOp op) {
  buildForOpResult(graph, builder, op, "conv_shape", op.getOperands(),
                   /*kgenParams=*/{}, /*dataDepInputIdxs=*/{2, 3, 4});
}

void Fallback<MO::GatherOp>::build(GraphOp &graph, OpBuilder &builder,
                                   MO::GatherOp op) {
  TensorType outputTy = op.getType();
  assert(outputTy.hasRank() && "Unranked tensor isn't supported");
  IntegerAttr outputRankAttr = builder.getIndexAttr(outputTy.getRank());

  buildForOpResult(graph, builder, op, "gather_shape", op.getOperands(),
                   /*kgenParams=*/{{"output_rank", outputRankAttr}},
                   /*dataDepInputIdxs=*/{2});
}

void Fallback<MO::MaxPoolOp>::build(GraphOp &graph, OpBuilder &builder,
                                    MO::MaxPoolOp op) {
  buildForOpResult(graph, builder, op, "pool_shape", op.getOperands(),
                   /*kgenParams=*/{},
                   /*dataDepInputIdxs=*/{1, 2, 3, 4});
}

void Fallback<MO::PadOp>::build(GraphOp &graph, OpBuilder &builder,
                                MO::PadOp op) {
  SmallVector<Value> shapeFuncInputs({op.getInput(), op.getPaddings()});
  buildForOpResult(graph, builder, op, "pad_shape", shapeFuncInputs,
                   /*kgenParams=*/{}, /*dataDepInputIdxs=*/{1});
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
  buildForOpResult(graph, builder, op, "reshape_shape", op.getOperands(),
                   /*kgenParams=*/{}, /*dataDepInputIdxs=*/{1});
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

void Fallback<MO::ArgNonzeroOp>::build(GraphOp &graph, OpBuilder &builder,
                                       MO::ArgNonzeroOp op) {
  buildForOpResult(graph, builder, op, "arg_nonzero_shape", op->getOperands(),
                   /*kgenParams=*/{}, /*dataDepInputIdxs=*/{0});
}

} // namespace

} // namespace M::MO

#undef HAS_FALLBACK
#undef NEEDS_NO_FALLBACK
#undef CALL_FALLBACK
