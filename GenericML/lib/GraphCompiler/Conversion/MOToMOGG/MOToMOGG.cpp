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
#include "GenericML/GraphCompiler/MOGGDialect/MOGGPasses.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace M;

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
      return success();
    }
  }

  ErrorOr<ValueRange> selectMOGGOp(Operation *op) const {
    OpBuilder b(op);
    b.setInsertionPoint(op);

    auto loc = op->getLoc();

    // Get the implementation info for this operation.
    auto itr = moToLitKernels.find(op->getName().getTypeID());
    if (itr == moToLitKernels.end())
      return Error("[MO_TO_MOGG] Failed to find a kernel for " +
                   op->getName().getStringRef().str());
    KernelRecord &kernel = itr->second;

    // If we should be using JIT generate a jit target. Otherwise use the shim.
    if (useJit && kernel.shouldHaveLit) {
      // Wrap in an elementwise block for now.
      auto fusedOp =
          b.create<MOGG::ElementwiseOp>(loc,
                                        /*outputTypes=*/op->getResultTypes(),
                                        /*inputOperands=*/op->getOperands());

      // Build up a block representing an elementwise lambda.
      mlir::Block &body = fusedOp.getBody().emplaceBlock();
      b.setInsertionPoint(&body, body.begin());
      for (mlir::Type operandTy : op->getOperandTypes())
        body.addArgument(operandTy, loc);

      // Create the lit target which implements this operation and add it within
      // the block.
      auto litTarget = b.create<MOGG::LitOp>(
          op->getLoc(),
          /*outputTypes=*/op->getResultTypes(),
          /*kernelTarget=*/kernel.moggID,
          /*originalOperationName=*/op->getName().stripDialect(),
          /*inputOperands=*/body.getArguments());

      b.create<MOGG::OutputOp>(loc, litTarget->getResults());
      return fusedOp.getResults();
    } else {
      // Create the mogg_ext_op
      return b
          .create<MOGG::ExtOp>(loc,
                               /*outputTypes=*/op->getResultTypes(),
                               /*op_name=*/kernel.shim,
                               /*inputOperands=*/op->getOperands())
          .getResults();
    }
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
      MO_SHIM_ONLY(MO::ConvOp, "mop.kernels.conv"),
      MO_SHIM_ONLY(MO::DivOp, "mop.kernels.div"),
      MO_TO_LIT(MO::ErfOp, "mop.kernels.erf"),
      MO_TO_LIT(MO::ExpOp, "mop.kernels.exp"),
      MO_SHIM_ONLY(MO::LogSoftmaxOp, "mop.kernels.logsoftmax"),
      MO_SHIM_ONLY(MO::TransposeOp, "mop.kernels.transpose"),
      MO_SHIM_ONLY(MO::PadOp, "mop.kernels.pad"),
      MO_SHIM_ONLY(MO::GatherOp, "mop.kernels.gather"),
      MO_SHIM_ONLY(MO::MatmulOp, "mop.kernels.matmul"),
      MO_SHIM_ONLY(MO::BatchMatmulOp, "mop.kernels.batch_matmul"),
      MO_SHIM_ONLY(MO::MeanOp, "mop.kernels.mean"),
      MO_TO_LIT(MO::MulOp, "mop.kernels.mul"),
      MO_SHIM_ONLY(MO::Pow3Op, "mop.kernels.pow3"),
      MO_SHIM_ONLY(MO::ReluOp, "mop.kernels.relu"),
      MO_TO_LIT(MO::RsqrtOp, "mop.kernels.rsqrt"),
      MO_SHIM_ONLY(MO::SigmoidOp, "mop.kernels.sigmoid"),
      MO_SHIM_ONLY(MO::ShapeOfOp, "mop.kernels.shape_of"),
      MO_SHIM_ONLY(MO::SoftmaxOp, "mop.kernels.softmax"),
      MO_TO_LIT(MO::SqrtOp, "mop.kernels.sqrt"),
      MO_TO_LIT(MO::SubOp, "mop.kernels.sub"),
      MO_TO_LIT(MO::TanhOp, "mop.kernels.tanh"),
  };

  RewritePatternSet patterns{ctx};
  patterns.add<MOOpPattern>(ctx, moToLitKernels, useJit.getValue());

  if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
    return signalPassFailure();
}

#undef MO_TO_LIT
#undef MO_SHIM_ONLY
