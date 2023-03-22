//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Converts MO ops to MGP and MEF ops.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/Conversion/MOToMOGG.h"
#include "GenericML/GraphCompiler/MODialect/MOOps.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGMatchers.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGPasses.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "LoweringHelper.h"

using namespace M;
using namespace mlir;

namespace M::MO {

#define GEN_PASS_DEF_MOTOMOGG
#include "GenericML/GraphCompiler/Conversion/MOToMOGG.h.inc"

} // namespace M::MO

namespace {

// A struct containing the specification we should search for to find a kernel
// for a given mo type.
struct KernelRecord {
  KernelRecord(StringLiteral moggID, StringLiteral shimName, bool hasLit)
      : moggID(moggID), shim(shimName), shouldHaveLit(hasLit) {}

  ~KernelRecord() = default;

  // The begining of the name. May have multiple matches we distinguish between
  // using the number of arguments.
  StringLiteral moggID;

  // The C++ shim which implements this (i.e non-jitted kernel path).
  StringLiteral shim;

  // Should we have a lit kernel for this?
  bool shouldHaveLit;
};

// Helper to add a map to a jit kernel AND a shim kernel for a given mo op.
// Add the ( to the name so we only match direct function matches.
#define MO_TO_LIT(MOOp, ShimName)                                              \
  {                                                                            \
    TypeID::get<MOOp>(), KernelRecord {                                        \
      MOOp::getOperationName(), ShimName, true                                 \
    }                                                                          \
  }

// Helper to add a map to a shim kernel for a given mo op.
#define MO_SHIM_ONLY(MOOp, ShimName)                                           \
  {                                                                            \
    TypeID::get<MOOp>(), KernelRecord { "", ShimName, false }                  \
  }

template <typename T>
auto MOMatrix(T *input) {
  return m_Ranked(2, m_Typed<MO::TensorType>(input, mlir::matchers::m_Any()));
}

struct GatherSumPattern : public RewritePattern {
  GatherSumPattern(MLIRContext *ctx)
      : RewritePattern(MatchAnyOpTypeTag{}, 2, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    OpBuilder b(op);
    b.setInsertionPoint(op);
    Value input, indices, gatherAxis, reduceAxis;
    Operation *gather, *sum;
    // TODO: Simplify this once we have parameterized shape calculations
    auto reduceAxisPattern = m_ScalarConstantInt(&reduceAxis, 1);
    auto sumPattern = m_Op<MO::ReduceAddOp>(
        &sum,
        m_Op<MO::GatherOp>(&gather, MOMatrix(&input), MOMatrix(&indices),
                           m_ScalarConstantInt(&gatherAxis, 0)),
        reduceAxisPattern);
    auto any = matchers::m_Any();
    auto zero = m_ScalarConstantInt(0);
    auto one = m_ScalarConstantInt(1);
    auto sumShapePattern = m_Op<MO::ShapeOfOp>(sumPattern);
    auto squeezeShapePattern = m_Op<MO::ConcatOp>(
        m_Op<MO::GatherOp>(sumShapePattern, m_Op<MO::RangeOp>(any, any, one),
                           zero),
        m_Op<MO::GatherOp>(sumShapePattern, m_Op<MO::RangeOp>(any, any, one),
                           zero),
        zero);

    if (!m_Op<MO::ReshapeOp>(sumPattern, squeezeShapePattern).match(op))
      return failure();

    auto results =
        b.create<MOGG::ExtOp>(
             op->getLoc(),
             /*outputTypes=*/op->getResultTypes(),
             /*op_name=*/"mop.kernels.gather_sum",
             /*inputOperands=*/
             SmallVector<Value>{input, indices, gatherAxis, reduceAxis})
            .getResults();

    rewriter.replaceOp(op, results);

    return mlir::success();
  }
};

using MoToKernelMapTy = SmallDenseMap<mlir::TypeID, KernelRecord, 32>;

struct MOOpPattern : public RewritePattern {
  MOOpPattern(MLIRContext *ctx, MoToKernelMapTy &moToLitKernels, bool useJit)
      : RewritePattern(MatchAnyOpTypeTag{}, 1, ctx),
        moToLitKernels(moToLitKernels), useJit(useJit) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    // skip non-mo ops and container/constant/output mo ops
    if (!isa<MO::MODialect>(op->getDialect()) || isa<MO::GraphOp>(op) ||
        isa<MO::ConstantOp>(op) || isa<MO::OutputOp>(op))
      return failure();

    auto results = selectMOGGOp(op);
    if (results.isError()) {
      return failure();
    } else {
      rewriter.replaceOp(op, results.takeValue());
      return mlir::success();
    }
  }

  ErrorOr<ValueRange> selectMOGGOp(Operation *op) const {
    // Get the implementation info for this operation.
    auto itr = moToLitKernels.find(op->getName().getTypeID());
    if (itr == moToLitKernels.end())
      return Error("[MO_TO_MOGG] Failed to find a kernel for " +
                   op->getName().getStringRef().str());
    KernelRecord &kernel = itr->second;

    // Helper to encapsulate all the code required to lower to a kernel.
    MO::KernelLowererHelper helper{op};

    // Current jit kernel constraints:
    // Only f32.
    // Ranked.
    // Rank between 1-5 (due to kernel bug #10551)
    // Only within the top level graph and not within e.g control flow.
    bool shouldUseJit = useJit && kernel.shouldHaveLit;
    for (Type t : op->getOperandTypes()) {
      auto tensor = dyn_cast<MO::TensorType>(t);
      if (!tensor || !tensor.getElementType().isF32() || !tensor.hasRank() ||
          tensor.getRank() > 5)
        shouldUseJit = false;
    }

    if (!isa<MO::GraphOp>(op->getParentOp()))
      shouldUseJit = false;

    // If we should be using JIT generate a jit target. Otherwise use the shim.
    if (shouldUseJit)
      return helper.lowerToKernel(kernel.moggID);
    else
      return helper.lowerToShim(kernel.shim);
  }

  MoToKernelMapTy &moToLitKernels;

  bool useJit;
};

struct MOToMOGGPass : public M::MO::impl::MOToMOGGBase<MOToMOGGPass> {
  using MOToMOGGBase::MOToMOGGBase;

  void runOnOperation() override;
};

} // namespace

void MOToMOGGPass::runOnOperation() {
  auto *ctx = &getContext();

  // A map of MO types to the known implementations. Ops can be backed by two
  // implementations. A C++ shim in the mop primitives and a LIT kernel which
  // will be jit compiled.
  MoToKernelMapTy moToLitKernels{
      MO_SHIM_ONLY(MO::AbsOp, "mop.kernels.abs"),
      MO_TO_LIT(MO::AddOp, "mop.kernels.add"),
      MO_SHIM_ONLY(MO::BroadcastShapeOp, "mop.kernels.broadcast_shape"),
      MO_SHIM_ONLY(MO::BroadcastToOp, "mop.kernels.broadcast_to"),
      MO_SHIM_ONLY(MO::CastOp, "mop.kernels.cast"),
      MO_SHIM_ONLY(MO::ConcatOp, "mop.kernels.concat"),
      MO_SHIM_ONLY(MO::Static::BroadcastToOp,
                   "mop.kernels.static.broadcast_to"),
      MO_SHIM_ONLY(MO::ConvOp, "mop.kernels.conv"),
      MO_TO_LIT(MO::DivOp, "mop.kernels.div"),
      MO_TO_LIT(MO::ErfOp, "mop.kernels.erf"),
      MO_TO_LIT(MO::ExpOp, "mop.kernels.exp"),
      MO_SHIM_ONLY(MO::LogSoftmaxOp, "mop.kernels.logsoftmax"),
      MO_SHIM_ONLY(MO::TransposeOp, "mop.kernels.transpose"),
      MO_SHIM_ONLY(MO::PadOp, "mop.kernels.pad"),
      MO_SHIM_ONLY(MO::GatherOp, "mop.kernels.gather"),
      MO_SHIM_ONLY(MO::MatmulOp, "mop.kernels.matmul"),
      MO_SHIM_ONLY(MO::BatchMatmulOp, "mop.kernels.batch_matmul"),
      MO_SHIM_ONLY(MO::MatrixBandPartOp, "mop.kernels.matrix_band_part"),
      MO_SHIM_ONLY(MO::MaxOp, "mop.kernels.max"),
      MO_SHIM_ONLY(MO::MeanOp, "mop.kernels.mean"),
      MO_SHIM_ONLY(MO::ReduceAddOp, "mop.kernels.reduce.add"),
      MO_SHIM_ONLY(MO::ReduceMulOp, "mop.kernels.reduce.mul"),
      MO_SHIM_ONLY(MO::ReduceMinOp, "mop.kernels.reduce.min"),
      MO_SHIM_ONLY(MO::ReduceMaxOp, "mop.kernels.reduce.max"),
      MO_SHIM_ONLY(MO::MaxPoolOp, "mop.kernels.pool.max"),
      MO_SHIM_ONLY(MO::AvgPoolOp, "mop.kernels.pool.avg"),
      MO_SHIM_ONLY(MO::MinOp, "mop.kernels.min"),
      MO_TO_LIT(MO::MulOp, "mop.kernels.mul"),
      MO_SHIM_ONLY(MO::PowOp, "mop.kernels.pow"),
      MO_SHIM_ONLY(MO::RangeOp, "mop.kernels.range"),
      MO_TO_LIT(MO::ReluOp, "mop.kernels.relu"),
      MO_TO_LIT(MO::RsqrtOp, "mop.kernels.rsqrt"),
      MO_SHIM_ONLY(MO::SelectOp, "mop.kernels.select"),
      MO_SHIM_ONLY(MO::SigmoidOp, "mop.kernels.sigmoid"),
      MO_SHIM_ONLY(MO::ShapeOfOp, "mop.kernels.shape_of"),
      MO_SHIM_ONLY(MO::SoftmaxOp, "mop.kernels.softmax"),
      MO_TO_LIT(MO::SqrtOp, "mop.kernels.sqrt"),
      MO_TO_LIT(MO::SubOp, "mop.kernels.sub"),
      MO_TO_LIT(MO::TanhOp, "mop.kernels.tanh"),
  };

  RewritePatternSet patterns{ctx};
  patterns.add<MOOpPattern>(ctx, moToLitKernels, useJit.getValue());
  patterns.add<GatherSumPattern>(ctx);

  if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
    return signalPassFailure();
}

#undef MO_TO_LIT
#undef MO_SHIM_ONLY
