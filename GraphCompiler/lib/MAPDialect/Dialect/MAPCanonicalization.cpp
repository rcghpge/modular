//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MAPDialect/Dialect/MAPOps.h"
#include "GraphCompiler/MODialect/MODevices.h"
#include "GraphCompiler/MODialect/MOOps.h"
#include "GraphCompiler/MODialect/MOTypes.h"
#include "GraphCompiler/MODialect/MOUtils.h"
#include "GraphCompiler/MOGGDialect/Support/TypeTraits.h"
#include "Support/AssertStream.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace M::MAP;

namespace {

// Thin canonicalization wrappers over `dedupeKernelInputs` (see MAPOps.h):
// fusion can leave a kernel binding the same value several times in `inputs`,
// and the `$inputs` list becomes the lowered kernel's parameter list.
struct DedupeIterOpaqueInputs : OpRewritePattern<IterOpaqueOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(IterOpaqueOp op,
                                PatternRewriter &rewriter) const override {
    return success(dedupeKernelInputs(op, rewriter));
  }
};

struct DedupeIterForeachInputs : OpRewritePattern<IterForeachOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(IterForeachOp op,
                                PatternRewriter &rewriter) const override {
    return success(dedupeKernelInputs(op, rewriter));
  }
};

// Mirrors `DedupeIterOpaqueInputs`, one level further in: a capturing
// closure reaches its body through its own capture operand + shadowing
// block argument. Generic over the interface (one pattern, not one per
// closure type) since all three expose the same primitives. Also what
// `LoopInvariantViewMotion` relies on: it calls `addCapture` freely,
// expecting this to fold away any duplicate it creates.
struct DedupeCapturingClosureCaptures
    : OpInterfaceRewritePattern<CapturingFusionClosure> {
  using OpInterfaceRewritePattern::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(CapturingFusionClosure op,
                                PatternRewriter &rewriter) const override {
    Block::BlockArgListType captureArgs = op.getCaptureBlockArgs();

    llvm::DenseMap<Value, BlockArgument> seen;
    llvm::SmallVector<unsigned> duplicates;
    for (auto [idx, capture] : llvm::enumerate(op.getCaptures()))
      if (!seen.insert({capture, captureArgs[idx]}).second)
        duplicates.push_back(idx);
    if (duplicates.empty())
      return failure();

    for (unsigned idx : duplicates)
      rewriter.replaceAllUsesWith(captureArgs[idx],
                                  seen.lookup(op.getCaptures()[idx]));
    rewriter.modifyOpInPlace(op.getOperation(), [&] {
      for (unsigned idx : llvm::reverse(duplicates))
        op.eraseCaptureAt(idx);
    });
    return success();
  }
};

// Drops captures whose shadowed block arg has no use. Unlike
// `map.iter.opaque`'s own inputs below, a capture carries no graph-level
// scheduling edge of its own (that lives on the opaque's inputs), so
// dropping one outright is always safe -- what lets `LoopInvariantViewMotion`
// shed a capture a hoisted load no longer reads.
struct DropDeadCapturingClosureCaptures
    : OpInterfaceRewritePattern<CapturingFusionClosure> {
  using OpInterfaceRewritePattern::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(CapturingFusionClosure op,
                                PatternRewriter &rewriter) const override {
    Block::BlockArgListType captureArgs = op.getCaptureBlockArgs();
    llvm::SmallVector<unsigned> dead;
    for (auto [idx, arg] : llvm::enumerate(captureArgs))
      if (arg.use_empty())
        dead.push_back(idx);
    if (dead.empty())
      return failure();

    rewriter.modifyOpInPlace(op.getOperation(), [&] {
      for (unsigned idx : llvm::reverse(dead))
        op.eraseCaptureAt(idx);
    });
    return success();
  }
};

// The largest constant worth rematerializing inside a kernel body rather than
// passing at runtime. Kept in step with `MOGG::FuseConstants`, which applies
// the same budget to the pre-advanced-fusion lane.
constexpr int64_t kMaxConstantSize = 2048;

// Returns the `mo.constant` producing `val` when its value can be
// rematerialized inside a kernel body, else null.
M::MO::ConstantOp getFusableConstant(Value val) {
  auto cst = val.getDefiningOp<M::MO::ConstantOp>();
  if (!cst || !M::MO::isOnHost(cst))
    return {};

  ElementsAttr elems = cst.getValueAttr();
  if (elems.getNumElements() > kMaxConstantSize)
    return {};
  if (!M::MOGG::TypeTraits::isSupportedMOGGType(elems.getElementType()))
    return {};
  if (!isa<M::ArrayElementsAttr, DenseResourceElementsAttr>(elems))
    return {};

  return cst;
}

// Rematerializes small constant inputs inside the body and drops them from
// `inputs`, which is what the lowered kernel's parameter list is built from.
// `map.iter.opaque` is isolated from above, so the clone has to live in the
// body; from there plain SSA dominance carries it to the loop body and to any
// prologue/epilogue capture.
//
// `DropDeadIterOpaqueInputs` would collect the operand on its own, and the
// original `mo.constant` is `Pure`, so the driver DCEs it once its last use
// goes. Clones left behind by folding the same constant into several input
// slots are uniqued by the folder, since `mo.constant` is `ConstantLike`.
struct FuseConstantIterOpaqueInputs : OpRewritePattern<IterOpaqueOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(IterOpaqueOp op,
                                PatternRewriter &rewriter) const override {
    // Folding into a device kernel miscompiles it into illegal memory accesses
    // (GEX-3839).
    if (!M::MO::isOnHost(op.getOperation()))
      return failure();

    rewriter.setInsertionPointToStart(&op.getBody().front());
    llvm::SmallVector<unsigned> folded;
    for (auto [idx, input] : llvm::enumerate(op.getInputs())) {
      M::MO::ConstantOp cst = getFusableConstant(input);
      if (!cst)
        continue;
      // A `map.iter.opaque` body holds only MAP ops, so the value is carried
      // by a `map.constant` rather than a clone of the `mo.constant`.
      Value inBody = ConstantOp::create(rewriter, cst.getLoc(), cst.getType(),
                                        cst.getValueAttr());
      rewriter.replaceAllUsesWith(op.getInputBlockArgs()[idx], inBody);
      folded.push_back(idx);
    }
    if (folded.empty())
      return failure();

    // Erasing shifts the indices after it, so unwind back to front.
    rewriter.modifyOpInPlace(op, [&] {
      for (unsigned idx : llvm::reverse(folded))
        op.eraseInputsAt(idx);
    });
    return success();
  }
};

// Drops inputs whose shadowed block arg has no use, except chains: a chain
// input sequences the kernel against other memory ops at the graph level even
// when the body never consumes it, so dropping it would change scheduling.
struct DropDeadIterOpaqueInputs : OpRewritePattern<IterOpaqueOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(IterOpaqueOp op,
                                PatternRewriter &rewriter) const override {
    llvm::SmallVector<unsigned> dead;
    for (auto [idx, arg] : llvm::enumerate(op.getInputBlockArgs()))
      if (arg.use_empty() && !llvm::isa<M::MO::ChainType>(arg.getType()))
        dead.push_back(idx);
    if (dead.empty())
      return failure();

    rewriter.modifyOpInPlace(op, [&] {
      for (unsigned idx : llvm::reverse(dead))
        op.eraseInputsAt(idx);
    });
    return success();
  }
};

// Walks `tensor` out of MAP entirely, to the MO-level value the kernel
// ultimately reads: out of every capturing closure it is a capture-shadowing
// block argument of, past the hoisted stride views that only re-address
// someone else's storage, and finally out of the enclosing
// `map.iter.opaque`'s own inputs -- which is the boundary, since those
// operands are plain graph values. Returns null as soon as the chain leaves
// that shape: an output block argument, a value computed inside the kernel,
// an unrecognized producer.
static Value resolveMOSource(Value tensor) {
  while (true) {
    if (auto arg = dyn_cast<BlockArgument>(tensor)) {
      Operation *parent = arg.getOwner()->getParentOp();
      if (auto closure = dyn_cast_or_null<CapturingFusionClosure>(parent)) {
        // Null for a leading callback parameter, which shadows nothing
        // outside the closure.
        tensor = closure.getCaptureFor(arg);
        if (!tensor)
          return nullptr;
        continue;
      }
      if (auto opaque = dyn_cast_or_null<IterOpaqueOp>(parent)) {
        // Null for an output block arg, which shadows no operand.
        tensor = opaque.getInputFor(arg);
        if (!tensor)
          return nullptr;
        continue;
      }
      return nullptr;
    }
    Operation *def = tensor.getDefiningOp();
    if (auto view = dyn_cast_or_null<StrideViewOpInterface>(def)) {
      tensor = view.getViewedTensor();
      continue;
    }
    return tensor;
  }
}

// Reads the single element of `elems` as a scalar attribute of `elemType`.
// Float and integer are the only kinds with a scalar attribute to build (and a
// Mojo literal to emit), same as the legacy pass's own extraction.
static TypedAttr getSingleElementAttr(mlir::ElementsAttr elems, M::DType dtype,
                                      Type elemType) {
  if (isa<mlir::FloatType>(elemType)) {
    auto blob = M::MO::tryGetConstantBlob<double>(elems, dtype);
    if (!blob || blob->size() != 1)
      return nullptr;
    return mlir::FloatAttr::get(elemType, blob->front());
  }
  if (isa<mlir::IntegerType>(elemType)) {
    auto blob = M::MO::tryGetConstantBlob<int64_t>(elems, dtype);
    if (!blob || blob->size() != 1)
      return nullptr;
    return mlir::IntegerAttr::get(elemType, blob->front());
  }
  return nullptr;
}

// The constant elements `source` holds, or null if it holds none.
//
// Two spellings reach here. On host, `FuseConstantIterOpaqueInputs` has
// usually already folded the constant into the body as a `map.constant`; on a
// device it refuses to (GEX-3839), so the constant is still arriving as a
// kernel operand and `source` is the `mo.constant` behind it. `m_Constant`
// covers both without naming either -- both ops are `ConstantLike` and fold to
// their own `ElementsAttr` -- and only fails for the shapes that need the
// walk below.
static mlir::ElementsAttr getConstantElements(Value source) {
  mlir::ElementsAttr elems;
  if (matchPattern(source, m_Constant(&elems)))
    return elems;

  auto srcTensor = dyn_cast<M::MO::TensorValue>(source);
  if (!srcTensor)
    return nullptr;

  // A host constant moved onto the device (`ops.constant(...).to(device)`)
  // reaches the kernel through an `mo.transfer`, which the identity chain
  // `tryGetConstantOpsOfRank` walks does not see through. Transfers preserve
  // value, dtype and rank, and inlining a literal replaces a read rather than
  // creating anything device-resident, so placement is irrelevant here.
  size_t rank = srcTensor.getType().getRank();
  while (auto transfer = srcTensor.getDefiningOp<M::MO::TransferOp>())
    srcTensor = cast<M::MO::TensorValue>(transfer.getInput());

  // Handles the constant / rebind / concat / reshape chains; more than one
  // constant means a concatenation, which cannot be a single element.
  SmallVector<M::MO::ConstantOp> constants =
      M::MO::tryGetConstantOpsOfRank(srcTensor, rank);
  if (constants.size() != 1)
    return nullptr;
  return constants.front().getValueAttr();
}

// Replaces a load of a one-element compile-time constant with that constant's
// single element, inlined as a literal -- the MAP-lane equivalent of
// `MOGG::InlineBroadcastConstantScalars`. Strictly more general than that
// pass: keying off the source having exactly one element, rather than off an
// actual broadcast view, covers every spelling of "reads the same element
// every iteration" (`map.index.broadcast` on the index, a stride view on the
// tensor, or no view at all), because a one-element tensor yields the same
// element at every index and the load's own index therefore cannot matter.
//
// Rooted on the load rather than on `map.iter.opaque`'s inputs so it never
// mutates across the isolated-from-above boundary: it only READS the outer
// operand to identify the constant. The now-unused capture and kernel input
// are shed by `DropDeadCapturingClosureCaptures` / `DropDeadIterOpaqueInputs`,
// and the dead view and `mo.constant` by ordinary DCE.
struct InlineConstantScalarLoad : OpRewritePattern<LoadOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LoadOp op,
                                PatternRewriter &rewriter) const override {
    Value source = resolveMOSource(op.getTensor());
    if (!source)
      return failure();
    auto srcTy = dyn_cast<M::MO::TensorType>(source.getType());
    if (!srcTy)
      return failure();

    // One element is what makes the index irrelevant. It also excludes
    // dynamically-shaped sources, whose `getNumElements()` is nullopt.
    if (srcTy.getNumElements() != 1)
      return failure();

    mlir::ElementsAttr elems = getConstantElements(source);
    if (!elems)
      return failure();

    Type elemType = op.getResult().getType();
    TypedAttr scalar = getSingleElementAttr(elems, srcTy.getDtype(), elemType);
    if (!scalar)
      return failure();

    rewriter.replaceOpWithNewOp<ConstantScalarOp>(op, elemType, scalar);
    return success();
  }
};

} // namespace

StoreOp M::MAP::getUniqueStore(Block &body) {
  StoreOp store;
  for (Operation &op : body)
    if (auto candidate = dyn_cast<StoreOp>(&op)) {
      ASSERT_STREAM(!store, << "expected exactly one map.store in the body");
      store = candidate;
    }
  ASSERT_STREAM(store, << "expected a map.store in the body");
  return store;
}

// Fusion can leave a MAP kernel binding the same value several times in
// `inputs` (once per fused producer). The `$inputs` list becomes the lowered
// kernel's parameter list, so keep it minimal: rewire every duplicate's
// shadowed block arg onto the first occurrence and erase it. Uniquing also
// makes the redundant loads and computations in the body identical, so CSE can
// collapse them -- and re-exposes a shared producer as single-use so it fuses.
bool M::MAP::dedupeKernelInputs(Operation *kernel, RewriterBase &rewriter) {
  auto dedupe = [&](auto op,
                    llvm::function_ref<BlockArgument(unsigned)> inputArg) {
    llvm::DenseMap<Value, BlockArgument> seen;
    llvm::SmallVector<unsigned> duplicates;
    for (auto [idx, input] : llvm::enumerate(op.getInputs()))
      if (!seen.insert({input, inputArg(idx)}).second)
        duplicates.push_back(idx);
    if (duplicates.empty())
      return false;

    for (unsigned idx : duplicates)
      rewriter.replaceAllUsesWith(inputArg(idx),
                                  seen.lookup(op.getInputs()[idx]));
    rewriter.modifyOpInPlace(op, [&] {
      for (unsigned idx : llvm::reverse(duplicates))
        op.eraseInputsAt(idx);
    });
    return true;
  };

  if (auto foreachOp = dyn_cast<IterForeachOp>(kernel)) {
    // A foreach body's leading arguments are its inputs.
    Block &body = foreachOp.getBody().front();
    return dedupe(foreachOp,
                  [&](unsigned idx) { return body.getArgument(idx); });
  }
  if (auto opaqueOp = dyn_cast<IterOpaqueOp>(kernel))
    return dedupe(opaqueOp, [&](unsigned idx) {
      return opaqueOp.getInputBlockArgs()[idx];
    });
  return false;
}

void IterOpaqueOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                               MLIRContext *context) {
  results.add<DedupeIterOpaqueInputs, DropDeadIterOpaqueInputs,
              FuseConstantIterOpaqueInputs>(context);
}

void IterForeachOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                MLIRContext *context) {
  results.add<DedupeIterForeachInputs>(context);
}

void LoadOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                         MLIRContext *context) {
  results.add<InlineConstantScalarLoad>(context);
}

void IterForeachBodyOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                    MLIRContext *context) {
  results.add<DedupeCapturingClosureCaptures, DropDeadCapturingClosureCaptures>(
      context);
}

void IterPrologueOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                 MLIRContext *context) {
  results.add<DedupeCapturingClosureCaptures, DropDeadCapturingClosureCaptures>(
      context);
}

void IterEpilogueOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                 MLIRContext *context) {
  results.add<DedupeCapturingClosureCaptures, DropDeadCapturingClosureCaptures>(
      context);
}
