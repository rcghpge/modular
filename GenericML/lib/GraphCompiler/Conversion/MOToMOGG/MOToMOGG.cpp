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
#include "GenericML/GraphCompiler/MODialect/MODialect.h"
#include "GenericML/GraphCompiler/MODialect/MOOps.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGPasses.h"
#include "GenericML/GraphCompiler/Matchers.h"
#include "GenericML/GraphCompiler/Support/MOGGTypeTraits.h"
#include "Support/ML/TensorShape.h"
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
               bool isView = false, bool isNonElemwise = false)
      : moggID(moggID), shim(shimName), shouldHaveMojo(hasMojo), isView(isView),
        isNonElemwise(isNonElemwise) {}

  ~KernelRecord() = default;

  // The begining of the name. May have multiple matches we distinguish between
  // using the number of arguments.
  StringLiteral moggID;

  // The C++ shim which implements this (i.e non-jitted kernel path).
  StringLiteral shim;

  // Should we have a mojo kernel for this?
  bool shouldHaveMojo;

  // Does this operation represent a view which then has special lowering code.
  bool isView;

  // Is this operation an actual operation which is *not* implemented by an
  // elementwise invoke over a lambda body.
  bool isNonElemwise;
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

// Same as the above but a special case for non-elementwise ops. Eventually this
// will be replaced through inspecting the mogg kernel.
#define MO_TO_MOJO_NON_ELEMWISE(MOOp, ShimName)                                \
  {                                                                            \
    TypeID::get<MOOp>(), KernelRecord {                                        \
      MOOp::getOperationName(), ShimName, true, false, true                    \
    }                                                                          \
  }

#define MO_MOJO_ONLY(MOOp)                                                     \
  {                                                                            \
    TypeID::get<MOOp>(), KernelRecord {                                        \
      MOOp::getOperationName(), "None", true                                   \
    }                                                                          \
  }

// Same as the above but a special case for view ops.
// MOOps is the C++ class type of the MO op which provides the op name for the
// map. ShimName is the name of the C++ shim kernel to target in MGP.
#define MO_TO_MOJO_VIEW(MOOp, ShimName)                                        \
  {                                                                            \
    TypeID::get<MOOp>(), KernelRecord {                                        \
      MOOp::getOperationName(), ShimName, true, true                           \
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
  MOOpPattern(MLIRContext *ctx, MoToKernelMapTy &moToMojoKernels,
              MOGG::MojoRegisterOp registry, mlir::ModuleOp kgenMod,
              bool useJit)
      : RewritePattern(MatchAnyOpTypeTag{}, 1, ctx),
        moToMojoKernels(moToMojoKernels), registry(registry),
        moggBindings(kgenMod), useJit(useJit) {}

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
    MO::KernelLowererHelper helper{op, registry, moggBindings};

    // Current jit kernel constraints:
    // Ranked.
    // Dimensions are static or shape infered (i.e not '?')
    // Only within the top level graph and not within e.g control flow.
    bool shouldUseJit = useJit && kernel.shouldHaveMojo;

    // Returns true if a type is compatable with jit.
    auto isTypeAllowed = [&](Type t) {
      auto tensor = dyn_cast<MO::TensorType>(t);
      Type dtype = tensor.getElementType();
      bool allowedElementTypes =
          MOGG::MOGGTypeTraits::isSupportedMOGGType(dtype);

      if (!tensor || !allowedElementTypes || !tensor.hasRank() ||
          tensor.getRank() > kMaxRank || tensor.hasUnknownShape())
        return false;

      // The symbolic shape must be trivial and not a compound expresion. I.E
      // N*M.
      for (TypedAttr symbolicDim : tensor.getDims()) {
        if (!(isa<KGEN::ParamDeclRefAttr>(symbolicDim) ||
              isa<IntegerAttr>(symbolicDim)))
          return false;
      }

      return true;
    };

    for (Type t : op->getOperandTypes())
      shouldUseJit &= isTypeAllowed(t);

    for (Type t : op->getResultTypes())
      shouldUseJit &= isTypeAllowed(t);

    // Check general op level constraints. Currently only single result ops
    // are allowed.
    if (op->getNumResults() != 1)
      shouldUseJit = false;

    StringRef shimTarget = kernel.shim;

    // We read the matmul attributes to allow finer grained control exactly when
    // we jit matmuls and to target the specific shims.
    if (auto matmul = dyn_cast<MO::MatmulOp>(op)) {
      if (matmul.getTransposeB().value_or(false)) {
        shimTarget = "mop.kernels.matmul_transpb";
      } else if (matmul.getPackedB().value_or(false)) {
        shimTarget = "mop.kernels.matmul_packb";
        shouldUseJit = false;
      }
    }

    // If we should be using JIT generate a jit target. Otherwise use the shim.
    if (shouldUseJit)
      return helper.lowerToKernel(kernel.moggID, kernel.isView,
                                  kernel.isNonElemwise);
    else if (shimTarget != "None")
      return helper.lowerToShim(shimTarget);
    else
      return Error("Attempting to lower a JIT-only kernel to a C++ shim");
  }

  MoToKernelMapTy &moToMojoKernels;

  // The mapping of MO operations to their Mojo kernel.
  MOGG::MojoRegisterOp registry;

  // The KGEN ir containing the mogg bindings.
  mlir::ModuleOp moggBindings;

  bool useJit;
};

struct MOToMOGGPass : public M::MO::impl::MOToMOGGBase<MOToMOGGPass> {
  using MOToMOGGBase::MOToMOGGBase;

  void runOnOperation() override;
};

} // namespace

void MOToMOGGPass::runOnOperation() {
  auto *ctx = &getContext();

  ModuleOp m = getOperation();

  // Load the MOGG registry of mojo ops.
  auto registryOr = MOGG::getOrLoadMojoLibrary(m);
  if (failed(registryOr))
    return signalPassFailure();

  MOGG::MojoRegisterOp registry = *registryOr;

  // Get mlir IR with all the bindings so we can introspect it.
  auto errOr = MOGG::loadMojoBindings(m->getContext());
  if (errOr.isError()) {
    m->emitError() << errOr.getError();
    return signalPassFailure();
  }

  OwningOpRef<mlir::ModuleOp> moggBindings = errOr.takeValue();

  // A map of MO types to the known implementations. Ops can be backed by two
  // implementations. A C++ shim in the mop primitives and a Mojo kernel which
  // will be jit compiled.
  MoToKernelMapTy moToMojoKernels{
      MO_MOJO_ONLY(MO::AbsOp),
      MO_MOJO_ONLY(MO::AddOp),
      MO_SHIM_ONLY(MO::ArgMaxOp, "mop.kernels.arg_max"),
      MO_SHIM_ONLY(MO::AvgPoolOp, "mop.kernels.pool.avg"),
      MO_SHIM_ONLY(MO::BroadcastShapeOp, "mop.kernels.broadcast_shape"),
      MO_SHIM_ONLY(MO::BroadcastToOp, "mop.kernels.broadcast_to"),
      MO_TO_MOJO_VIEW(MO::Static::BroadcastToOp, "None"),
      MO_MOJO_ONLY(MO::CastOp),
      MO_MOJO_ONLY(MO::CeilOp),
      MO_SHIM_ONLY(MO::ConcatOp, "mop.kernels.concat"),
      MO_SHIM_ONLY(MO::ConvOp, "mop.kernels.conv"),
      MO_MOJO_ONLY(MO::DivOp),
      MO_MOJO_ONLY(MO::ErfOp),
      MO_MOJO_ONLY(MO::ExpOp),
      MO_MOJO_ONLY(MO::EqualOp),
      MO_MOJO_ONLY(MO::FloorOp),
      MO_MOJO_ONLY(MO::GeluOp),
      MO_MOJO_ONLY(MO::GreaterOp),
      MO_MOJO_ONLY(MO::GreaterEqualOp),
      MO_MOJO_ONLY(MO::Log1pOp),
      MO_SHIM_ONLY(MO::LinAlg::BandPartOp, "mop.kernels.linalg.band_part"),
      MO_TO_MOJO_NON_ELEMWISE(MO::LinAlg::SolveOp, "mop.kernels.linalg.solve"),
      MO_SHIM_ONLY(MO::LogSoftmaxOp, "mop.kernels.logsoftmax"),
      MO_TO_MOJO_NON_ELEMWISE(MO::GatherOp, "mop.kernels.gather"),
      MO_TO_MOJO_NON_ELEMWISE(MO::MatmulOp, "mop.kernels.matmul"),
      MO_TO_MOJO_NON_ELEMWISE(MO::BatchMatmulOp, "mop.kernels.batch_matmul"),
      MO_MOJO_ONLY(MO::MaxOp),
      MO_SHIM_ONLY(MO::MaxPoolOp, "mop.kernels.pool.max"),
      MO_TO_MOJO_NON_ELEMWISE(MO::MeanOp, "mop.kernels.mean"),
      MO_MOJO_ONLY(MO::MinOp),
      MO_TO_MOJO(MO::MulOp, "mop.kernels.mul"),
      MO_MOJO_ONLY(MO::NotEqualOp),
      MO_SHIM_ONLY(MO::PadOp, "mop.kernels.pad"),
      MO_TO_MOJO_NON_ELEMWISE(MO::ReduceAddOp, "mop.kernels.reduce.add"),
      MO_SHIM_ONLY(MO::ReduceMulOp, "mop.kernels.reduce.mul"),
      MO_SHIM_ONLY(MO::ReduceMinOp, "mop.kernels.reduce.min"),
      MO_SHIM_ONLY(MO::ReduceMaxOp, "mop.kernels.reduce.max"),
      MO_MOJO_ONLY(MO::PowOp),
      MO_SHIM_ONLY(MO::RangeOp, "mop.kernels.range"),
      MO_MOJO_ONLY(MO::ReluOp),
      MO_MOJO_ONLY(MO::RsqrtOp),
      MO_SHIM_ONLY(MO::ScatterNdOp, "mop.kernels.scatter_nd"),
      MO_MOJO_ONLY(MO::SelectOp),
      MO_SHIM_ONLY(MO::ShapeOfOp, "mop.kernels.shape_of"),
      MO_TO_MOJO(MO::SigmoidOp, "mop.kernels.sigmoid"),
      MO_TO_MOJO_VIEW(MO::SliceOp, "mop.kernels.slice"),
      MO_SHIM_ONLY(MO::SoftmaxOp, "mop.kernels.softmax"),
      MO_SHIM_ONLY(MO::SplitOp, "mop.kernels.split"),
      MO_TO_MOJO(MO::SqrtOp, "mop.kernels.sqrt"),
      MO_TO_MOJO(MO::SubOp, "mop.kernels.sub"),
      MO_TO_MOJO(MO::TanhOp, "mop.kernels.tanh"),
      MO_TO_MOJO_VIEW(MO::TransposeOp, "mop.kernels.transpose"),
      MO_SHIM_ONLY(MO::WhereOp, "mop.kernels.where"),
  };

  RewritePatternSet patterns{ctx};
  patterns.add<MOOpPattern>(ctx, moToMojoKernels, registry, moggBindings.get(),
                            useJit.getValue());

  if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
    return signalPassFailure();

  // Just delete it for now. We should figure out what we want to do with it but
  // for now it can just be reloaded in the jit. We don't want to have this huge
  // thing in the IR when we run the other passes. The mojo is all in memory
  // anyway so should be cheap-ish.
  registry.erase();
}

#undef MO_TO_MOJO
#undef MO_TO_MOJO_NON_ELEMWISE
#undef MO_MOJO_ONLY
#undef MO_SHIM_ONLY
#undef MO_TO_MOJO_NON_ELEMWISE
#undef MO_TO_MOJO_VIEW
