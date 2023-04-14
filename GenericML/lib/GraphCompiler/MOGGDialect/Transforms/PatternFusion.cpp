//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/MODialect/MOInterfaces.h"
#include "mlir/IR/BuiltinOps.h"

#include "GenericML/GraphCompiler/MODialect/MOOps.h"
#include "GenericML/GraphCompiler/MODialect/MOTypes.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGPasses.h"
#include "GenericML/GraphCompiler/Matchers.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/MathExtras.h"

#include "mlir/IR/IRMapping.h"

using namespace M;
using namespace mlir;
using namespace mlir::matchers;

namespace M::MOGG {

#define GEN_PASS_DEF_PATTERNFUSION
#include "GenericML/GraphCompiler/MOGGDialect/MOGGPasses.h.inc"

namespace {

struct GatherSumPattern : public OpRewritePattern<MO::ReduceAddOp> {
  GatherSumPattern(MLIRContext *ctx) : OpRewritePattern(ctx, 2) {}

  LogicalResult matchAndRewrite(MO::ReduceAddOp op,
                                PatternRewriter &rewriter) const override {
    Value input, indices, gatherAxis, reduceAxis;

    if (!m_Op<MO::ReduceAddOp>(
             m_Op<MO::GatherOp>(m_MOMatrix(m_Any(&input)),
                                m_MOMatrix(m_Any(&indices)),
                                m_ScalarConstant(&gatherAxis, 0)),
             m_ScalarConstant(&reduceAxis, 1))
             .match(op))
      return rewriter.notifyMatchFailure(op, "failed to match pattern");

    if (!input || !indices || !gatherAxis || !reduceAxis)
      return rewriter.notifyMatchFailure(op, "null input value");

    rewriter.replaceOpWithNewOp<MOGG::ExtOp>(
        op,
        /*outputTypes=*/op->getResultTypes(),
        /*op_name=*/"mop.kernels.gather_sum",
        /*inputOperands=*/
        SmallVector<Value>{input, indices, gatherAxis, reduceAxis});

    return mlir::success();
  }
};

struct MatmulConstBPattern : public OpRewritePattern<MO::MatmulOp> {
  MatmulConstBPattern(MLIRContext *ctx) : OpRewritePattern(ctx, 2) {}

  LogicalResult matchAndRewrite(MO::MatmulOp op,
                                PatternRewriter &rewriter) const override {
    Value input = op.getInputA();
    Value weight = op.getInputB();

    if (!input || !isa_and_nonnull<MO::ConstantOp>(weight.getDefiningOp()))
      return failure();

    MO::TensorType weightsType = cast<MO::TensorType>(weight.getType());
    SmallVector<uint64_t> weightsDims;
    for (int64_t dim : weightsType.getIntDims()) {
      // NOTE this (constants with unknown shape) may occur as a result of
      // constant folding. Once we have shape inference on by default, we won't
      // have this anymore.
      if (dim < 0)
        return failure();
      weightsDims.push_back(dim);
    }

    if (!weightsType.hasStaticShape() || weightsType.hasUnknownShape())
      return failure();

    //   res = matmul(any, const_weight) : [_, _]
    // ===>
    //   packed_weight = pack(const_weight) : [?, ?]
    //   res = matmul_packb(any, packed_weight) : [_, _]
    // where `_` could be a parameter, integer, or `?`
    MO::TensorType packedWeightsType = weightsType;
    if (weightsType.hasRank()) {
      SmallVector<int64_t> unknownDims(weightsType.getRank(),
                                       mlir::ShapedType::kDynamic);
      Type dtype = weightsType.getElementType();
      packedWeightsType = weightsType.cloneWith(unknownDims, dtype);
    }
    Value packedWeights =
        rewriter
            .create<MOGG::ExtOp>(op->getLoc(),
                                 /*outputTypes=*/packedWeightsType,
                                 /*op_name=*/"mop.kernels.pack_matmul_b",
                                 /*inputOperands=*/SmallVector<Value>{weight})
            .getResult(0);

    rewriter.replaceOpWithNewOp<MOGG::ExtOp>(
        op, op->getResultTypes(), "mop.kernels.matmul_packb",
        SmallVector<Value>{input, packedWeights}, weightsDims);

    return mlir::success();
  }
};

struct MatmulTranspBPattern : public OpRewritePattern<MO::MatmulOp> {
  MatmulTranspBPattern(MLIRContext *ctx) : OpRewritePattern(ctx, 2) {}

  LogicalResult matchAndRewrite(MO::MatmulOp op,
                                PatternRewriter &rewriter) const override {
    Value input, weight, axes;
    if (!m_Op<MO::MatmulOp>(
             m_Any(&input),
             m_Op<MO::TransposeOp>(m_Any(&weight), m_MOVector(m_Any(&axes))))
             .match(op))
      return rewriter.notifyMatchFailure(op, "failed to match pattern");

    if (!input || !weight || !axes)
      return rewriter.notifyMatchFailure(op, "null input value");

    std::optional<SmallVector<int64_t>> maybeAxes =
        MO::tryGetConstantIntVector(cast<MO::TensorValue>(axes));

    if (!maybeAxes.has_value())
      return rewriter.notifyMatchFailure(op, "axes not constant");

    SmallVector<int64_t> &axesValues = maybeAxes.value();

    if (axesValues[0] != 1 || axesValues[1] != 0)
      return rewriter.notifyMatchFailure(op, "axes not transposing");

    rewriter.replaceOpWithNewOp<MOGG::ExtOp>(
        op,
        /*outputTypes=*/op->getResultTypes(),
        /*op_name=*/"mop.kernels.matmul_transpb",
        /*inputOperands=*/
        SmallVector<Value>{input, weight});

    return mlir::success();
  }
};

struct BatchMatmulAdjBPattern : public OpRewritePattern<MO::BatchMatmulOp> {
  BatchMatmulAdjBPattern(MLIRContext *ctx) : OpRewritePattern(ctx, 2) {}

  LogicalResult matchAndRewrite(MO::BatchMatmulOp op,
                                PatternRewriter &rewriter) const override {
    Value input, weight, axes;
    if (!m_Op<MO::BatchMatmulOp>(
             m_Any(&input),
             m_Op<MO::TransposeOp>(m_Any(&weight), m_MOVector(m_Any(&axes))))
             .match(op))
      return rewriter.notifyMatchFailure(op, "failed to match pattern");

    if (!input || !weight || !axes)
      return rewriter.notifyMatchFailure(op, "null input value");

    std::optional<SmallVector<int64_t>> maybeAxes =
        MO::tryGetConstantIntVector(cast<MO::TensorValue>(axes));

    if (!maybeAxes.has_value())
      return rewriter.notifyMatchFailure(op, "axes not constant");

    SmallVector<int64_t> &axesValues = maybeAxes.value();

    int64_t rank = axesValues.size();

    if (axesValues[rank - 1] != rank - 2 || axesValues[rank - 2] != rank - 1)
      return rewriter.notifyMatchFailure(op, "axes not adjoint");

    for (int64_t i = 0, e = rank - 2; i < e; ++i)
      if (axesValues[i] != i)
        return rewriter.notifyMatchFailure(op, "axes not adjoint");

    rewriter.replaceOpWithNewOp<MOGG::ExtOp>(
        op,
        /*outputTypes=*/op->getResultTypes(),
        /*op_name=*/"mop.kernels.batch_matmul_adjb",
        /*inputOperands=*/
        SmallVector<Value>{input, weight});

    return mlir::success();
  }
};

struct FullyConnectedPattern : public RewritePattern {
  FullyConnectedPattern(MLIRContext *ctx)
      : RewritePattern(MatchAnyOpTypeTag{}, 3, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    OpBuilder b(op);
    b.setInsertionPoint(op);

    Value input, weight, bias;
    Operation *matmul = nullptr, *add = nullptr, *reshape = nullptr,
              *gelu = nullptr, *relu = nullptr, *tanh = nullptr;
    auto matmulP = m_Op<MO::MatmulOp>(&matmul, m_MOMatrix(m_Any(&input)),
                                      m_MOMatrix(m_Any(&weight)));
    auto broadcastP =
        m_Optional(matmulP, m_Op<MO::Static::BroadcastToOp>(matmulP));
    auto bias_add = m_BinaryCommutativeOp<MO::AddOp>(
        &add,
        m_Optional(broadcastP,
                   m_Op<MO::Static::ReshapeOp>(&reshape, broadcastP)),
        m_Op<MO::Static::BroadcastToOp>(
            m_Ranked(1, m_Typed<M::MO::TensorType>(m_Any(&bias)))));

    auto fully_connected = m_Optional(
        m_Op<MO::GeluOp>(&gelu, bias_add),
        m_Optional(m_Op<MO::ReluOp>(&relu, bias_add),
                   m_Optional(m_Op<MO::TanhOp>(&tanh, bias_add), bias_add)));
    // TODO: Make the Optional Matcher work with Operation* directly
    if (op->getNumResults() > 0 && !fully_connected.match(op->getResult(0)))
      return rewriter.notifyMatchFailure(op, "failed to match pattern");

    if (!input || !weight || !bias)
      return rewriter.notifyMatchFailure(op, "null input value");

    Value fc;
    if (!isa_and_nonnull<MO::ConstantOp>(weight.getDefiningOp())) {
      if (gelu || relu || tanh)
        return rewriter.notifyMatchFailure(
            op, "haven't implemented activation fusions with out packb");
      fc = rewriter
               .create<MOGG::ExtOp>(op->getLoc(),
                                    reshape ? matmul->getResultTypes()[0]
                                            : add->getResultTypes()[0],
                                    "mop.kernels.fully_connected",
                                    SmallVector<Value>{input, weight, bias})
               .getResult(0);
    } else {
      MO::TensorType weightsType = cast<MO::TensorType>(weight.getType());
      SmallVector<uint64_t> weightsDims;
      for (int64_t dim : weightsType.getIntDims()) {
        // NOTE this (constants with unknown shape) may occur as a result of
        // constant folding. Once we have shape inference on by default, we
        // won't have this anymore.
        if (dim < 0)
          return failure();
        weightsDims.push_back(dim);
      }
      MO::TensorType packedWeightsType = weightsType;
      if (weightsType.hasRank()) {
        SmallVector<int64_t> unknownDims(weightsType.getRank(),
                                         mlir::ShapedType::kDynamic);
        Type dtype = weightsType.getElementType();
        packedWeightsType = weightsType.cloneWith(unknownDims, dtype);
      }
      Value packedWeights =
          rewriter
              .create<MOGG::ExtOp>(op->getLoc(),
                                   /*outputTypes=*/packedWeightsType,
                                   /*op_name=*/"mop.kernels.pack_matmul_b",
                                   /*inputOperands=*/SmallVector<Value>{weight})
              .getResult(0);

      std::string kernel;
      if (gelu)
        kernel = "mop.kernels.fully_connected.packb.gelu";
      else if (relu)
        kernel = "mop.kernels.fully_connected.packb.relu";
      else if (tanh)
        kernel = "mop.kernels.fully_connected.packb.tanh";
      else
        kernel = "mop.kernels.fully_connected.packb";

      fc = rewriter
               .create<MOGG::ExtOp>(
                   op->getLoc(),
                   reshape ? matmul->getResultTypes()[0]
                           : add->getResultTypes()[0],
                   kernel, SmallVector<Value>{input, packedWeights, bias},
                   weightsDims)
               .getResult(0);
    }

    if (reshape) {
      rewriter.replaceOpWithNewOp<MO::Static::ReshapeOp>(
          op, op->getResultTypes()[0], fc);
    } else {
      rewriter.replaceOp(op, fc);
    }

    return mlir::success();
  }
};

// TODO: Move this to a MO transformation pass after refactoring patterns out of
// MOGG
struct SinkTransposeThroughUnaryOpPattern
    : public OpInterfaceRewritePattern<MO::ElementWiseUnary> {
  SinkTransposeThroughUnaryOpPattern(MLIRContext *ctx)
      : OpInterfaceRewritePattern(ctx, 2) {}

  LogicalResult matchAndRewrite(MO::ElementWiseUnary op,
                                PatternRewriter &rewriter) const override {
    //  input     perm       input
    //    \        /           |
    //    transpose     ==>  unary(*)    perm
    //        |                 \       /
    //      unary (op)          transpose(*)
    //
    // NOTE: (*) means result type or operands of the op were updated
    Value input, perm;
    Operation *opaqueTransposeOp;
    auto unaryP = m_Op<MO::ElementWiseUnary>(
        m_Op<MO::TransposeOp>(&opaqueTransposeOp, m_Any(&input), m_Any(&perm)));
    if (!unaryP.match(op))
      return rewriter.notifyMatchFailure(op, "failed to match pattern");

    if (!input || !perm)
      return rewriter.notifyMatchFailure(op, "null input or permutation value");

    // If transpose op has multiple uses, we don't sink the transposes because
    // we'll have to recompute the original transpose op for other users.
    // TODO add [get|set]Input method to `ElementWiseUnary` interface
    Value transpose = opaqueTransposeOp->getResult(0);
    if (!transpose.hasOneUse())
      return rewriter.notifyMatchFailure(op, "transpose op has multiple uses");

    auto transposeOp = cast<MO::TransposeOp>(opaqueTransposeOp);
    rewriter.replaceAllUsesWith(op->getResults(), transposeOp.getResult());
    op->setOperand(0, input);
    op->getResult(0).setType(input.getType());
    transposeOp.getInputMutable().assign(op->getResult(0));
    transposeOp->moveAfter(op);

    return mlir::success();
  }
};

struct PatternFusionPass : public impl::PatternFusionBase<PatternFusionPass> {
  using PatternFusionBase::PatternFusionBase;

  void runOnOperation() override;
};

} // namespace

void PatternFusionPass::runOnOperation() {
  auto *ctx = &getContext();

  RewritePatternSet patterns{ctx};
  patterns.add<GatherSumPattern, BatchMatmulAdjBPattern, MatmulTranspBPattern,
               SinkTransposeThroughUnaryOpPattern>(ctx);
  // TODO(#12590): enable FullyConnectedPattern
  if (useMatmulPacking.getValue())
    patterns.add<MatmulConstBPattern>(ctx);

  if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
    return signalPassFailure();
}

} // namespace M::MOGG
