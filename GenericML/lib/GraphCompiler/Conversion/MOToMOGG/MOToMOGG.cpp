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
#include "GenericML/GraphCompiler/MODialect/MOInterfaces.h"
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
  // Overload for an unreachable op which should just assert.
  KernelRecord(StringLiteral moggID)
      : moggID(moggID), shim("None"), shouldHaveMojo(false),
        isNonElemwise(false), shouldBeUnreachable(true) {}

  KernelRecord(StringLiteral moggID, StringLiteral shimName)
      : moggID(moggID), shim(shimName), shouldHaveMojo(false),
        isNonElemwise(false), shouldBeUnreachable(false) {}

  KernelRecord(StringLiteral moggID, bool isNonElemwise)
      : moggID(moggID), shim("None"), shouldHaveMojo(true),
        isNonElemwise(isNonElemwise), shouldBeUnreachable(false) {}

  ~KernelRecord() = default;

  // The begining of the name. May have multiple matches we distinguish between
  // using the number of arguments.
  StringLiteral moggID;

  // The C++ shim which implements this (i.e non-jitted kernel path).
  StringLiteral shim;

  // Should we have a mojo kernel for this?
  bool shouldHaveMojo;

  // Is this operation an actual operation which is *not* implemented by an
  // elementwise invoke over a lambda body.
  bool isNonElemwise;

  // If we should assert when we see this op.
  bool shouldBeUnreachable;
};

// Helper to add a map to a jit kernel for a given mo op.
// MOOps is the C++ class type of the MO op which provides the op name for the
// map.
#define MO_TO_ELEMWISE(MOOp)                                                   \
  {                                                                            \
    TypeID::get<MOOp>(), KernelRecord { MOOp::getOperationName(), false }      \
  }

// Same as the above but a special case for non-elementwise ops. Eventually this
// will be replaced through inspecting the mogg kernel.
#define MO_TO_KERNEL(MOOp)                                                     \
  {                                                                            \
    TypeID::get<MOOp>(), KernelRecord { MOOp::getOperationName(), true }       \
  }

// Operations which are handled specially and not lowered to MOGG
#define MO_NO_LOWER(MOOp)                                                      \
  {                                                                            \
    TypeID::get<MOOp>(), KernelRecord {                                        \
      MOOp::getOperationName(), StringLiteral { "None" }                       \
    }                                                                          \
  }

// Operations which should not exist at this stage of lowering.
#define MO_UNLOWERABLE_OP(MOOp)                                                \
  {                                                                            \
    TypeID::get<MOOp>(), KernelRecord { MOOp::getOperationName() }             \
  }

// Helper to add a map to a shim kernel for a given mo op.
// MOOps is the C++ class type of the MO op which provides the op name for the
// map. ShimName is the name of the C++ shim kernel to target in MGP.
#define MO_SHIM_ONLY(MOOp, ShimName)                                           \
  {                                                                            \
    TypeID::get<MOOp>(), KernelRecord {                                        \
      MOOp::getOperationName(), StringLiteral { ShimName }                     \
    }                                                                          \
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
    // skip non-mo ops and container/output mo ops
    if (!isa<MO::MODialect>(op->getDialect()) || isa<MO::GraphOp>(op) ||
        isa<MO::OutputOp>(op))
      return failure();

    auto maybeResults = selectMOGGOp(op);
    if (!maybeResults.has_value())
      return failure();

    ErrorOr<ValueRange> &results = maybeResults.value();
    if (results.isError()) {
      op->emitOpError("[MO_TO_MOGG] ") << results.getError();
      return failure();
    } else {
      rewriter.replaceOp(op, results.takeValue());
      return mlir::success();
    }
  }

  // nullopt --> expected no lowering. Error --> unexpected error. Else lowering
  std::optional<ErrorOr<ValueRange>> selectMOGGOp(Operation *op) const {
    // Get the implementation info for this operation.
    auto itr = moToMojoKernels.find(op->getName().getTypeID());
    if (itr == moToMojoKernels.end())
      return Error("Failed to find a kernel registration for " +
                   op->getName().getStringRef().str());
    KernelRecord &kernel = itr->second;

    if (kernel.shouldBeUnreachable) {
      return Error("Op is has no legal lowering and should have been removed "
                   "before lowering" +
                   op->getName().getStringRef().str());
    }

    if (!kernel.shouldHaveMojo && kernel.shim == "None")
      return std::nullopt;

    // Helper to encapsulate all the code required to lower to a kernel.
    MO::KernelLowererHelper helper{op, registry, moggBindings};

    // Current jit kernel constraints:
    // Ranked.
    // Dimensions are static or shape infered (i.e not '?')
    // Only within the top level graph and not within e.g control flow.
    bool shouldUseJit = useJit && kernel.shouldHaveMojo;

    StringRef shimTarget = kernel.shim;

    auto isDTypeAllowed = [&](Type t) {
      auto tensor = dyn_cast<MO::TensorType>(t);
      Type dtype = tensor.getElementType();
      return MOGG::MOGGTypeTraits::isSupportedMOGGType(dtype);
    };

    // Returns true if a type is compatable with jit.
    auto isShapeAllowed = [&](Type t) {
      auto tensor = dyn_cast<MO::TensorType>(t);

      if (!tensor || !tensor.hasRank() || tensor.getRank() > kMaxRank ||
          tensor.hasUnknownShape())
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

    bool hasCompatibleJITDtypes = true;
    bool hasCompatibleJITShape = true;
    for (Type t : op->getOperandTypes()) {
      hasCompatibleJITDtypes &= isDTypeAllowed(t);
      hasCompatibleJITShape &= isShapeAllowed(t);
    }
    for (Type t : op->getResultTypes()) {
      hasCompatibleJITDtypes &= isDTypeAllowed(t);
      hasCompatibleJITShape &= isShapeAllowed(t);
    }

    if (!hasCompatibleJITDtypes && shouldUseJit) {
      std::string errStr;
      llvm::raw_string_ostream os(errStr);
      os << "Found at least one incompatible DType for a JIT-able operation"
         << ". Operand types " << op->getOperandTypes()
         << ". Results Types: " << op->getResultTypes();
      return Error(errStr);
    }

    // If we should be using JIT generate a jit target.
    // Otherwise use the shim but only if the dtype is specially supported.
    if (shouldUseJit && hasCompatibleJITDtypes && hasCompatibleJITShape) {
      if (auto invokeShapeFuncOp = dyn_cast<MO::InvokeShapeFuncOp>(op))
        return helper.lowerShapeFuncToKernel(invokeShapeFuncOp);
      if (auto layoutTransformOp = dyn_cast<MO::LayoutTransformOp>(op))
        return helper.lowerLayoutTransformToKernel(layoutTransformOp);

      return helper.lowerToKernel(kernel.isNonElemwise);
    } else if (shimTarget != "None") {
      return helper.lowerToShim(shimTarget);
    } else {
      return Error("Attempting to lower a JIT-only kernel to a C++ shim");
    }
  }

  MoToKernelMapTy &moToMojoKernels;

  // The mapping of MO operations to their Mojo kernel.
  MOGG::MojoRegisterOp registry;

  // The KGEN ir containing the mogg bindings.
  mlir::ModuleOp moggBindings;

  // Extra libraries to source kernels from.
  std::string extraLibPaths;

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
  auto registryOr = MOGG::getOrLoadMojoLibrary(m, extraLibPaths.getValue());
  if (failed(registryOr))
    return signalPassFailure();

  MOGG::MojoRegisterOp registry = *registryOr;

  // Get mlir IR with all the bindings so we can introspect it.
  auto errOr =
      MOGG::loadMojoBindings(m->getContext(), extraLibPaths.getValue());
  if (errOr.isError()) {
    m->emitError() << errOr.getError();
    return signalPassFailure();
  }

  OwningOpRef<mlir::ModuleOp> moggBindings = errOr.takeValue();

  // A map of MO types to the known implementations. Ops can be backed by two
  // implementations. A C++ shim in the mop primitives and a Mojo kernel which
  // will be jit compiled.
  // clang-format off
  // ^-- otherwise tries to put 2+ entries per line
  MoToKernelMapTy moToMojoKernels{
      MO_TO_ELEMWISE(MO::AbsOp),
      MO_TO_ELEMWISE(MO::AddOp),
      MO_TO_KERNEL(MO::ArgMaxOp),
      MO_TO_KERNEL(MO::ArgMinOp),
      MO_TO_KERNEL(MO::AvgPoolOp),
      MO_TO_KERNEL(MO::Static::BroadcastToOp),
      MO_TO_ELEMWISE(MO::CastOp),
      MO_TO_ELEMWISE(MO::CeilOp),
      MO_TO_KERNEL(MO::ConcatOp),
      MO_TO_KERNEL(MO::ConvOp),
      MO_TO_ELEMWISE(MO::DivOp),
      MO_TO_ELEMWISE(MO::ErfOp),
      MO_TO_ELEMWISE(MO::ExpOp),
      MO_TO_ELEMWISE(MO::EqualOp),
      MO_TO_ELEMWISE(MO::FloorOp),
      MO_TO_ELEMWISE(MO::GeluOp),
      MO_TO_ELEMWISE(MO::GreaterOp),
      MO_TO_ELEMWISE(MO::GreaterEqualOp),
      MO_TO_ELEMWISE(MO::InvokeShapeFuncOp),
      MO_TO_ELEMWISE(MO::IsInfOp),
      MO_TO_ELEMWISE(MO::IsNaNOp),
      MO_TO_ELEMWISE(MO::Log1pOp),
      MO_TO_KERNEL(MO::LayoutTransformOp),
      MO_TO_KERNEL(MO::LinAlg::BandPartOp),
      MO_TO_KERNEL(MO::LinAlg::SolveOp),
      MO_TO_KERNEL(MO::LogSoftmaxOp),
      MO_TO_KERNEL(MO::MatmulOp),
      MO_TO_KERNEL(MO::BatchMatmulOp),
      MO_TO_KERNEL(MO::GatherOp),
      MO_TO_ELEMWISE(MO::MaxOp),
      MO_TO_KERNEL(MO::MaxPoolOp),
      MO_TO_KERNEL(MO::MeanOp),
      MO_TO_ELEMWISE(MO::MinOp),
      MO_TO_ELEMWISE(MO::MulOp),
      MO_TO_ELEMWISE(MO::NotEqualOp),
      MO_TO_KERNEL(MO::PadOp),
      MO_TO_KERNEL(MO::ReduceAddOp),
      MO_TO_KERNEL(MO::ReduceMulOp),
      MO_TO_KERNEL(MO::ReduceMinOp),
      MO_TO_KERNEL(MO::ReduceMaxOp),
      MO_TO_ELEMWISE(MO::PowOp),
      MO_TO_KERNEL(MO::RangeOp),
      MO_TO_ELEMWISE(MO::ReluOp),
      MO_TO_ELEMWISE(MO::RsqrtOp),
      MO_TO_KERNEL(MO::ScatterNdOp),
      MO_TO_ELEMWISE(MO::SelectOp),
      MO_TO_ELEMWISE(MO::SigmoidOp),
      MO_TO_KERNEL(MO::SliceOp),
      MO_TO_KERNEL(MO::SoftmaxOp),
      MO_TO_KERNEL(MO::SplitOp),
      MO_TO_ELEMWISE(MO::SqrtOp),
      MO_TO_ELEMWISE(MO::SubOp),
      MO_TO_ELEMWISE(MO::TanhOp),
      MO_TO_KERNEL(MO::TransposeOp),
      MO_TO_KERNEL(MO::WhereOp),

      // These kernels we don't jit but we almost certainly never actually call them.
      MO_SHIM_ONLY(MO::BroadcastShapeOp, "mop.kernels.broadcast_shape"),
      MO_SHIM_ONLY(MO::ShapeOfOp, "mop.kernels.shape_of"),

      // Bookkeeping for gotchas -- handled specially elsewhere
      MO_NO_LOWER(MO::ConstantOp),
      MO_NO_LOWER(MO::RebindOp),
      MO_NO_LOWER(MO::Static::ReshapeOp),
      MO_NO_LOWER(MO::IndexToTensorOp),
      MO_NO_LOWER(MO::Shape::ToTensorOp),
      MO_NO_LOWER(MO::ReshapeOp),

      // Yield will never be lowered and is informational
      MO_NO_LOWER(MO::YieldOp),

      // Control flow
      MO_NO_LOWER(MO::IfOp),
      MO_NO_LOWER(MO::WhileOp),

      // Printing
      MO_NO_LOWER(MO::DebugTensorPrintOp),

      // These ops have no legal lowering and should assert on sight.
      MO_UNLOWERABLE_OP(MO::BroadcastToOp),
  };
  // clang-format on
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

#undef MO_TO_ELEMWISE
#undef MO_TO_KERNEL
#undef MO_NO_LOWER
#undef MO_UNLOWERABLE_OP
#undef MO_SHIM_ONLY
