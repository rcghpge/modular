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
#include "GenericML/GraphCompiler/Matchers.h"
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
  KernelRecord(StringLiteral moggID, StringLiteral shimName, bool hasMojo,
               bool hasShim = true, bool isView = false)
      : moggID(moggID), shim(shimName), shouldHaveMojo(hasMojo),
        shouldHaveShim(hasShim), isView(isView) {}

  ~KernelRecord() = default;

  // The begining of the name. May have multiple matches we distinguish between
  // using the number of arguments.
  StringLiteral moggID;

  // The C++ shim which implements this (i.e non-jitted kernel path).
  StringLiteral shim;

  // Should we have a mojo kernel for this?
  bool shouldHaveMojo;

  // Operations only expected to be lowered by jit do not need a shim.
  bool shouldHaveShim;

  // Does this operation represent a view which then has special lowering code.
  bool isView;
};

// Helper to add a map to a jit kernel AND a shim kernel for a given mo op.
// Add the ( to the name so we only match direct function matches.
// MOOps is the C++ class type of the MO op which provides the op name for the
// map. ShimName is the name of the C++ shim kernel to target in MGP.
#define MO_TO_MOJO(MOOp, ShimName)                                             \
  {                                                                            \
    TypeID::get<MOOp>(), KernelRecord {                                        \
      MOOp::getOperationName(), ShimName, true                                 \
    }                                                                          \
  }

// Same as the above but a special case for view ops.
// MOOps is the C++ class type of the MO op which provides the op name for the
// map. ShimName is the name of the C++ shim kernel to target in MGP.
#define MO_TO_MOJO_VIEW(MOOp, ShimName)                                        \
  {                                                                            \
    TypeID::get<MOOp>(), KernelRecord {                                        \
      MOOp::getOperationName(), ShimName, true, true, true                     \
    }                                                                          \
  }

// Helper to add a map to a shim kernel for a given mo op.
// MOOps is the C++ class type of the MO op which provides the op name for the
// map. ShimName is the name of the C++ shim kernel to target in MGP.
#define MO_SHIM_ONLY(MOOp, ShimName)                                           \
  {                                                                            \
    TypeID::get<MOOp>(), KernelRecord { "", ShimName, false }                  \
  }

using MoToKernelMapTy = SmallDenseMap<mlir::TypeID, KernelRecord, 64>;

struct MOOpPattern : public RewritePattern {
  MOOpPattern(MLIRContext *ctx, MoToKernelMapTy &moToMojoKernels, bool useJit)
      : RewritePattern(MatchAnyOpTypeTag{}, 1, ctx),
        moToMojoKernels(moToMojoKernels), useJit(useJit) {}

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
    auto itr = moToMojoKernels.find(op->getName().getTypeID());
    if (itr == moToMojoKernels.end())
      return Error("[MO_TO_MOGG] Failed to find a kernel for " +
                   op->getName().getStringRef().str());
    KernelRecord &kernel = itr->second;

    // Helper to encapsulate all the code required to lower to a kernel.
    MO::KernelLowererHelper helper{op};

    // Current jit kernel constraints:
    // Ranked.
    // Rank between 1-5 (due to kernel bug #10551)
    // Dimensions are static or shape infered (i.e not '?')
    // Only within the top level graph and not within e.g control flow.
    bool shouldUseJit = useJit && kernel.shouldHaveMojo;

    // Returns true if a type is compatable with jit.
    auto isTypeAllowed = [&](Type t) {
      auto tensor = dyn_cast<MO::TensorType>(t);
      bool allowedElementTypes = tensor.getElementType().isF32() ||
                                 tensor.getElementType().isSignedInteger(32) ||
                                 tensor.getElementType().isSignedInteger(64);

      if (!tensor || !allowedElementTypes || !tensor.hasRank() ||
          tensor.getRank() > 5 || tensor.hasUnknownShape())
        return false;
      return true;
    };

    for (Type t : op->getOperandTypes())
      shouldUseJit &= isTypeAllowed(t);

    for (Type t : op->getResultTypes())
      shouldUseJit &= isTypeAllowed(t);

    // Check general op level constraints. Currently only ops within the main
    // block and single result ops are allowed.
    if (!isa<MO::GraphOp>(op->getParentOp()) || op->getNumResults() != 1)
      shouldUseJit = false;

    // If we should be using JIT generate a jit target. Otherwise use the shim.
    if (shouldUseJit) {
      return helper.lowerToKernel(kernel.moggID, kernel.isView);
    } else {
      if (!kernel.shouldHaveShim)
        return Error("[MOTOMOGG] Kernel does not have a shim but did not "
                     "select a jit kernel.");

      return helper.lowerToShim(kernel.shim);
    }
  }

  MoToKernelMapTy &moToMojoKernels;

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
  // implementations. A C++ shim in the mop primitives and a Mojo kernel which
  // will be jit compiled.
  MoToKernelMapTy moToMojoKernels{
      MO_SHIM_ONLY(MO::AbsOp, "mop.kernels.abs"),
      MO_TO_MOJO(MO::AddOp, "mop.kernels.add"),
      MO_SHIM_ONLY(MO::ArgMaxOp, "mop.kernels.arg_max"),
      MO_SHIM_ONLY(MO::AvgPoolOp, "mop.kernels.pool.avg"),
      MO_SHIM_ONLY(MO::BatchMatmulOp, "mop.kernels.batch_matmul"),
      MO_SHIM_ONLY(MO::BroadcastShapeOp, "mop.kernels.broadcast_shape"),
      MO_SHIM_ONLY(MO::BroadcastToOp, "mop.kernels.broadcast_to"),
      MO_TO_MOJO_VIEW(MO::Static::BroadcastToOp,
                      "mop.kernels.static.broadcast_to"),
      MO_TO_MOJO(MO::CastOp, "mop.kernels.cast"),
      MO_SHIM_ONLY(MO::ConcatOp, "mop.kernels.concat"),
      MO_SHIM_ONLY(MO::ConvOp, "mop.kernels.conv"),
      MO_TO_MOJO(MO::DivOp, "mop.kernels.div"),
      MO_TO_MOJO(MO::ErfOp, "mop.kernels.erf"),
      MO_TO_MOJO(MO::ExpOp, "mop.kernels.exp"),
      MO_SHIM_ONLY(MO::EqualOp, "mop.kernels.equal"),
      MO_TO_MOJO(MO::GeluOp, "mop.kernels.gelu"),
      MO_SHIM_ONLY(MO::GreaterOp, "mop.kernels.greater"),
      MO_SHIM_ONLY(MO::GreaterEqualOp, "mop.kernels.greater_equal"),
      MO_SHIM_ONLY(MO::LogSoftmaxOp, "mop.kernels.logsoftmax"),
      MO_SHIM_ONLY(MO::GatherOp, "mop.kernels.gather"),
      MO_SHIM_ONLY(MO::MatmulOp, "mop.kernels.matmul"),
      MO_SHIM_ONLY(MO::MatrixBandPartOp, "mop.kernels.matrix_band_part"),
      MO_SHIM_ONLY(MO::MaxOp, "mop.kernels.max"),
      MO_SHIM_ONLY(MO::MaxPoolOp, "mop.kernels.pool.max"),
      MO_SHIM_ONLY(MO::MeanOp, "mop.kernels.mean"),
      MO_SHIM_ONLY(MO::MinOp, "mop.kernels.min"),
      MO_TO_MOJO(MO::MulOp, "mop.kernels.mul"),
      MO_SHIM_ONLY(MO::NotEqualOp, "mop.kernels.not_equal"),
      MO_SHIM_ONLY(MO::PadOp, "mop.kernels.pad"),
      MO_SHIM_ONLY(MO::ReduceAddOp, "mop.kernels.reduce.add"),
      MO_SHIM_ONLY(MO::ReduceMulOp, "mop.kernels.reduce.mul"),
      MO_SHIM_ONLY(MO::ReduceMinOp, "mop.kernels.reduce.min"),
      MO_SHIM_ONLY(MO::ReduceMaxOp, "mop.kernels.reduce.max"),
      MO_SHIM_ONLY(MO::PowOp, "mop.kernels.pow"),
      MO_SHIM_ONLY(MO::RangeOp, "mop.kernels.range"),
      MO_TO_MOJO(MO::ReluOp, "mop.kernels.relu"),
      MO_TO_MOJO(MO::RsqrtOp, "mop.kernels.rsqrt"),
      MO_SHIM_ONLY(MO::SelectOp, "mop.kernels.select"),
      MO_SHIM_ONLY(MO::ShapeOfOp, "mop.kernels.shape_of"),
      MO_TO_MOJO(MO::SigmoidOp, "mop.kernels.sigmoid"),
      MO_SHIM_ONLY(MO::SliceOp, "mop.kernels.slice"),
      MO_SHIM_ONLY(MO::SoftmaxOp, "mop.kernels.softmax"),
      MO_SHIM_ONLY(MO::SplitOp, "mop.kernels.split"),
      MO_TO_MOJO(MO::SqrtOp, "mop.kernels.sqrt"),
      MO_TO_MOJO(MO::SubOp, "mop.kernels.sub"),
      MO_TO_MOJO(MO::TanhOp, "mop.kernels.tanh"),
      MO_SHIM_ONLY(MO::TransposeOp, "mop.kernels.transpose"),
      MO_SHIM_ONLY(MO::WhereOp, "mop.kernels.where"),
  };

  RewritePatternSet patterns{ctx};
  patterns.add<MOOpPattern>(ctx, moToMojoKernels, useJit.getValue());

  if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
    return signalPassFailure();
}

#undef MO_TO_MOJO
#undef MO_SHIM_ONLY
#undef MO_TO_MOJO_VIEW
