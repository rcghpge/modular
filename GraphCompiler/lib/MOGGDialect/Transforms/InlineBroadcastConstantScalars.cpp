//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MODialect/MOOps.h"
#include "GraphCompiler/MODialect/MOTypes.h"
#include "GraphCompiler/MODialect/MOUtils.h"
#include "GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GraphCompiler/MOGGDialect/MOGGPasses.h"

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"

using namespace M;

namespace M::MOGG {

#define GEN_PASS_DEF_INLINEBROADCASTCONSTANTSCALARS
#include "GraphCompiler/MOGGDialect/MOGGPasses.h.inc"

namespace {

// A fully-broadcast compile-time constant scalar recovered from a
// `StaticBroadcastTo` view. This is the information common to both lambda
// shapes; each call site pairs it with how the value is consumed (see
// `FusibleBroadcastConstant` and `CapturedBroadcastConstant`).
struct BroadcastConstant {
  // The recovered scalar constant, typed to the source element type.
  TypedAttr value;
  // The element dtype (used to build the !mo.scalar result type).
  DType dtype;
  // The broadcast view op producing the scalar.
  CallViewOp view;
  // The kernel exec-body block argument feeding the broadcast view (the
  // external constant operand).
  BlockArgument sourceArg;
};

// If `view` is a `StaticBroadcastTo` of a single-element constant scalar passed
// in as a direct kernel operand, recover the scalar constant value. Returns
// std::nullopt when the pattern does not match.
static std::optional<BroadcastConstant>
recoverBroadcastConstant(KernelOp kernel, CallViewOp view) {
  // The broadcast was lowered to a generic `mogg.call.view` kernel, so we can't
  // dyn_cast to MO::Static::BroadcastToOp directly; the original MO op it
  // represents is recorded in the view's `moOpName` attribute, which we match.
  if (view.getMoOpName() != MO::Static::BroadcastToOp::getOperationName())
    return std::nullopt;

  // The broadcast source is the view's first input. We only handle the case
  // where it is a direct kernel operand (a constant passed into the kernel).
  if (view.getInputs().empty())
    return std::nullopt;
  auto sourceArg = dyn_cast<BlockArgument>(view.getInputs().front());
  if (!sourceArg || sourceArg.getOwner() != &kernel.getExecBody().front())
    return std::nullopt;

  Value external = kernel.getOperand(sourceArg.getArgNumber());
  auto srcTensor = dyn_cast<MO::TensorValue>(external);
  if (!srcTensor)
    return std::nullopt;
  MO::TensorType srcTy = srcTensor.getType();

  // Only fully-broadcast scalars: a single-element source guarantees the
  // broadcast view has all-zero strides. Partial (vector) broadcasts and
  // dynamically-shaped sources (getNumElements() == nullopt) are excluded.
  if (srcTy.getNumElements() != 1)
    return std::nullopt;

  // A host constant moved onto the device (`ops.constant(...).to(device)`)
  // reaches the kernel through an `mo.transfer`, which the identity chain below
  // does not see through. Transfers preserve value, dtype, and rank, and the
  // literal is inlined into the kernel body, so placement is irrelevant here.
  // FuseViews already fused this view in on the assumption that this pass
  // inlines it, so failing to look through leaves a zero-strided operand the
  // Trainium tile-fusion path cannot express.
  while (auto transfer = srcTensor.getDefiningOp<MO::TransferOp>())
    srcTensor = cast<MO::TensorValue>(transfer.getInput());

  // Recover the constant ops feeding the source (handles constant /
  // rebind / reshape chains). We require exactly one constant.
  SmallVector<MO::ConstantOp> constantOps =
      MO::tryGetConstantOpsOfRank(srcTensor, srcTy.getRank());
  if (constantOps.size() != 1)
    return std::nullopt;

  mlir::ElementsAttr elems = constantOps.front().getValueAttr();
  DType dtype = srcTy.getDtype();
  mlir::Type elemType = srcTy.getElementType();

  TypedAttr scalarAttr;
  if (isa<mlir::FloatType>(elemType)) {
    auto blob = MO::tryGetConstantBlob<double>(elems, dtype);
    if (!blob || blob->size() != 1)
      return std::nullopt;
    scalarAttr = mlir::FloatAttr::get(elemType, blob->front());
  } else if (isa<mlir::IntegerType>(elemType)) {
    auto blob = MO::tryGetConstantBlob<int64_t>(elems, dtype);
    if (!blob || blob->size() != 1)
      return std::nullopt;
    scalarAttr = mlir::IntegerAttr::get(elemType, blob->front());
  } else {
    return std::nullopt;
  }

  return BroadcastConstant{scalarAttr, dtype, view, sourceArg};
}

// A broadcast constant consumed as a fusible operand of an elementwise/input
// lambda (bound to a value arg at a given operand index).
struct FusibleBroadcastConstant {
  BroadcastConstant constant;
  // Index of the fusible operand within the BindLambdaOp.
  unsigned fusibleIdx;
  // The lambda block argument bound to this fusible operand.
  BlockArgument valueArg;
};

// A broadcast constant captured into an output (epilogue) lambda body and read
// via a `mogg.tensor.load`.
struct CapturedBroadcastConstant {
  BroadcastConstant constant;
  MOGG::LoadTensorOp load;
};

// If `fusible` (a BindLambdaOp operand of `kernel`) is a `StaticBroadcastTo`
// view of a single-element constant scalar, recover the match. Returns
// std::nullopt when the pattern does not match.
static std::optional<FusibleBroadcastConstant>
matchFusibleBroadcastConstant(KernelOp kernel, unsigned fusibleIdx,
                              Value fusible, BlockArgument valueArg) {
  auto view = fusible.getDefiningOp<CallViewOp>();
  if (!view)
    return std::nullopt;
  std::optional<BroadcastConstant> constant =
      recoverBroadcastConstant(kernel, view);
  if (!constant)
    return std::nullopt;
  return FusibleBroadcastConstant{*constant, fusibleIdx, valueArg};
}

// Erases broadcast views that became dead and any kernel inputs whose exec-body
// argument is now unused, so the constant buffer is never passed into the
// kernel. SimplifyKernelOperands only de-duplicates operands and there is no
// dead-kernel-input cleanup pass, so we remove them here. Views/sources are
// deduped (a constant may feed multiple views/loads). Order of input erasure is
// irrelevant: eraseInputsAt re-reads the (live) argument number, which MLIR
// keeps up to date as inputs are erased.
static void eraseDeadViewsAndInputs(KernelOp kernel,
                                    ArrayRef<BroadcastConstant> constants) {
  llvm::SmallSetVector<CallViewOp, 4> uniqueViews;
  llvm::SmallSetVector<BlockArgument, 4> uniqueSources;
  for (const BroadcastConstant &constant : constants) {
    uniqueViews.insert(constant.view);
    uniqueSources.insert(constant.sourceArg);
  }

  for (CallViewOp view : uniqueViews) {
    if (!view->use_empty())
      continue;
    Operation *shapeProducer = view.getInputs().size() > 1
                                   ? view.getInputs()[1].getDefiningOp()
                                   : nullptr;
    view->erase();
    // The shape operand (mogg.tensor.shape) is dead once the view is gone.
    if (shapeProducer && shapeProducer->use_empty())
      shapeProducer->erase();
  }

  for (BlockArgument arg : uniqueSources)
    if (arg.use_empty())
      kernel.eraseInputsAt(arg.getArgNumber());
}

struct InlineBroadcastConstantScalars
    : public impl::InlineBroadcastConstantScalarsBase<
          InlineBroadcastConstantScalars> {
  void runOnOperation() override {
    MO::GraphOp graph = getOperation();
    // Collect lambdas up front, then process: rewriting one lambda erases
    // views and kernel inputs, so we avoid mutating the IR while the walker is
    // still traversing. Walking BindLambdaOp visits each elementwise, input,
    // and output lambda exactly once (those are the only bind kinds).
    SmallVector<BindLambdaOp> binds;
    graph.walk([&](BindLambdaOp bind) { binds.push_back(bind); });
    for (BindLambdaOp bind : binds)
      processLambda(bind);
  }

  void processLambda(BindLambdaOp bind) {
    auto kernel = bind->getParentOfType<KernelOp>();
    assert(kernel && "BindLambdaOp must be nested in a KernelOp");

    // Elementwise and input (prologue) lambdas share the same shape: the
    // broadcast-constant appears as a fusible operand consumed via a value arg.
    // Output (epilogue) lambdas instead capture broadcasts directly into the
    // body and read them via `mogg.tensor.load`, which needs a separate
    // matcher.
    switch (bind.getBindKind()) {
    case BindLambdaKind::ElemwiseLambda:
    case BindLambdaKind::InputLambda:
      processFusibleOperands(kernel, bind);
      break;
    case BindLambdaKind::OutputLambda:
      processCapturedLoads(kernel, bind);
      break;
    }
  }

  // Inlines broadcast-constant scalars that appear as fusible operands of
  // `bind` (the elementwise/input lambda shape).
  void processFusibleOperands(KernelOp kernel, BindLambdaOp bind) {
    OperandRange fusibles = bind.getFusibleTensors();
    ArrayRef<BlockArgument> valueArgs = bind.getValueArgs();
    unsigned numFusibles = fusibles.size();

    // Collect all broadcast-constant-scalar fusible operands up front, before
    // mutating the IR (which would invalidate operand/argument indices).
    SmallVector<FusibleBroadcastConstant> candidates;
    for (unsigned i = 0; i < numFusibles; ++i) {
      if (auto match = matchFusibleBroadcastConstant(kernel, i, fusibles[i],
                                                     valueArgs[i]))
        candidates.push_back(*match);
    }
    if (candidates.empty())
      return;

    // Guard: keep at least one fusible tensor so the elementwise call still has
    // a real SIMD operand to infer the vector width from. If every operand is a
    // broadcast constant, leave the last one as a load (constant + constant is
    // normally folded upstream anyway).
    // TODO: Once the new fusion system lands it can emit an all-constant lambda
    // without any load at all, so this width-donor guard can be dropped.
    if (candidates.size() >= numFusibles) {
      candidates.pop_back();
      if (candidates.empty())
        return;
    }

    numInlinedScalars += candidates.size();

    OpBuilder builder(kernel.getContext());
    builder.setInsertionPoint(bind);

    // Create the constant scalars and rewire the elementwise call to use them.
    // This does not change operand/argument indices yet.
    for (FusibleBroadcastConstant &cand : candidates) {
      auto scalarTy =
          MO::ScalarType::get(kernel.getContext(), cand.constant.dtype);
      Value cst = MOGG::ConstantScalarOp::create(
          builder, cand.constant.view.getLoc(), scalarTy, cand.constant.value);
      cand.valueArg.replaceAllUsesWith(cst);
    }

    // Erase the fusible operands + their block arguments in descending index
    // order so earlier indices remain valid.
    SmallVector<unsigned> indices;
    for (FusibleBroadcastConstant &cand : candidates)
      indices.push_back(cand.fusibleIdx);
    llvm::sort(indices, std::greater<>());
    for (unsigned idx : indices) {
      bind->eraseOperand(idx);
      bind.getLambda().front().eraseArgument(idx);
    }

    // Erase the now-dead broadcast views and drop the orphaned kernel inputs.
    SmallVector<BroadcastConstant> constants;
    for (FusibleBroadcastConstant &cand : candidates)
      constants.push_back(cand.constant);
    eraseDeadViewsAndInputs(kernel, constants);
  }

  // Inlines broadcast-constant scalars that are captured into an output
  // (epilogue) lambda body and read via `mogg.tensor.load`. Unlike the fusible
  // operand shape, the broadcast view is referenced directly inside the body
  // rather than passed as a bind operand.
  void processCapturedLoads(KernelOp kernel, BindLambdaOp bind) {
    // Collect matching loads up front, before mutating the body.
    SmallVector<CapturedBroadcastConstant> hits;
    bind.getBody()->walk([&](MOGG::LoadTensorOp load) {
      auto view = load.getTensor().getDefiningOp<CallViewOp>();
      if (!view)
        return;
      if (std::optional<BroadcastConstant> constant =
              recoverBroadcastConstant(kernel, view))
        hits.push_back({*constant, load});
    });
    if (hits.empty())
      return;

    numInlinedScalars += hits.size();

    OpBuilder builder(kernel.getContext());
    builder.setInsertionPoint(bind);

    // Replace each captured load with a constant scalar. The load result feeds
    // an elementwise op whose other operand is the bound output value (always a
    // real, width-carrying load), so vector-width inference still has a donor.
    for (CapturedBroadcastConstant &hit : hits) {
      auto scalarTy =
          MO::ScalarType::get(kernel.getContext(), hit.constant.dtype);
      Value cst = MOGG::ConstantScalarOp::create(builder, hit.load.getLoc(),
                                                 scalarTy, hit.constant.value);
      hit.load.getOutputs().replaceAllUsesWith(cst);
      hit.load->erase();
    }

    SmallVector<BroadcastConstant> constants;
    for (CapturedBroadcastConstant &hit : hits)
      constants.push_back(hit.constant);
    eraseDeadViewsAndInputs(kernel, constants);
  }
};

} // namespace

} // namespace M::MOGG
