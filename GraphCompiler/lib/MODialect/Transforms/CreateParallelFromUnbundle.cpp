//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// mo-create-parallel-from-unbundle
//
// Walks each `mo.tensor.unbundle` and, when its N per-launch results each
// have exactly one user and those users form a structurally identical,
// mutually distinct consumer set, wraps those consumers in a fresh
// `mo.parallel` region placed downstream of the unbundle.
//
// The new parallel's primary input is a transient
// `bundle(unbundle(X)#0..N-1)`; `TensorBundleOp::fold` collapses it back
// to `X` during the post-pass canonicalize, restoring the SSA equality
// that `mo-merge-adjacent-parallels` matches on to merge the wrapped
// region into the producing parallel.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MODialect/MOOps.h"
#include "GraphCompiler/MODialect/Transforms/MOTransforms.h"
#include "Support/AssertStream.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "create-parallel-from-unbundle"

namespace M::MO {

#define GEN_PASS_DEF_CREATEPARALLELFROMUNBUNDLE
#include "GraphCompiler/MODialect/Transforms/MOTransforms.h.inc"

} // namespace M::MO

using namespace M;
using namespace M::MO;

namespace {

//===----------------------------------------------------------------------===//
// Data structures
//===----------------------------------------------------------------------===//

/// An unbundle whose N results each have exactly one user, whose user
/// set is structurally identical and mutually distinct, and which all
/// reference the unbundle at a single shared operand position
/// (`drivingOperandIdx`).
struct WrapCandidate {
  TensorUnbundleOp unbundleOp;
  /// One consumer op per launch, in launch-index order.
  SmallVector<Operation *> consumerOps;
  /// The operand position at which `consumerOps[i]` references
  /// `unbundleOp.getResult(i)`, identical across all launches.  This is
  /// the driving operand of the wrap.
  size_t drivingOperandIdx;
};

/// Pre-IR plan for the new parallel's inputs.
struct InputPlan {
  /// One outer-value list per bundle input to emit, in order.  Entry 0
  /// always covers the driving unbundle's results -- the transient
  /// `bundle(unbundle(X)#0..N-1)` that folds back to `X` during the
  /// post-pass canonicalize.  The driving operand uses this slot;
  /// subsequent entries cover non-driving operand positions.
  SmallVector<SmallVector<Value>> bundles;
  /// For each operand of the representative consumer, the index into
  /// `bundles` whose corresponding block argument supplies it.
  SmallVector<size_t> operandToBlockArg;
};

//===----------------------------------------------------------------------===//
// Predicates
//===----------------------------------------------------------------------===//

/// Returns true if \p op can be cloned into a parallel body: at least one
/// result, all results tensor-typed, no `!mo.buffer` or `!mo.chain`
/// operands, and no non-trivial memory effects.
static bool isAbsorbableConsumer(Operation *op) {
  if (op->getNumResults() == 0)
    return false;
  for (Value result : op->getResults())
    if (!isa<TensorType>(result.getType()))
      return false;
  for (Value operand : op->getOperands())
    if (isa<BufferType, ChainType>(operand.getType()))
      return false;
  if (op->hasTrait<OpTrait::HasRecursiveMemoryEffects>())
    return false;
  if (auto iface = dyn_cast<mlir::MemoryEffectOpInterface>(op))
    return iface.hasNoEffect();
  // MO elementwise / custom ops are pure but may not implement
  // MemoryEffectOpInterface.  The buffer/chain check above is the strict
  // guard for chain-bearing ops.
  return true;
}

/// Returns true if \p a and \p b have the same op name, operand and result
/// counts, and identical attribute dictionaries modulo the `device`
/// attribute (which legitimately varies per launch).
static bool opsMatch(Operation *a, Operation *b) {
  if (a->getName() != b->getName() ||
      a->getNumOperands() != b->getNumOperands() ||
      a->getNumResults() != b->getNumResults())
    return false;

  auto stripDevice = [](DictionaryAttr attrs) {
    SmallVector<NamedAttribute> filtered;
    filtered.reserve(attrs.size());
    for (NamedAttribute attr : attrs)
      if (attr.getName() != kOperationDeviceRefAttrName)
        filtered.push_back(attr);
    return DictionaryAttr::get(attrs.getContext(), filtered);
  };
  return stripDevice(a->getAttrDictionary()) ==
         stripDevice(b->getAttrDictionary());
}

/// Returns true if \p perLaunchValues can be packaged as one bundle input
/// to the parallel.  Requires shape and dtype to agree across launches,
/// and each value's device to match its corresponding entry in
/// \p launchDevices (the per-launch device refs of the driving unbundle's
/// source bundle).
static bool canBundlePerLaunch(ArrayRef<Value> perLaunchValues,
                               ArrayRef<DeviceRefAttr> launchDevices) {
  auto firstTy = dyn_cast<TensorType>(perLaunchValues[0].getType());
  if (!firstTy)
    return false;

  // Check that shapes and dtypes agree across launches.
  for (Value v : perLaunchValues.drop_front()) {
    auto ty = dyn_cast<TensorType>(v.getType());
    if (!ty || firstTy.getShapeAttr() != ty.getShapeAttr() ||
        firstTy.getDtype() != ty.getDtype())
      return false;
  }

  // Check that devices agree across launches.
  for (auto [i, v] : llvm::enumerate(perLaunchValues))
    if (cast<TensorType>(v.getType()).getDeviceRef() != launchDevices[i])
      return false;
  return true;
}

//===----------------------------------------------------------------------===//
// Candidate discovery
//===----------------------------------------------------------------------===//

/// Returns a wrap candidate seeded by \p unbundleOp, or std::nullopt if any
/// of these gates fail: every result has exactly one user, those users are
/// mutually distinct, the representative is absorbable, every other peer
/// is structurally identical to the representative, and every peer
/// references its launch's unbundle result at the same operand position
/// as the representative.
static std::optional<WrapCandidate> findCandidate(TensorUnbundleOp unbundleOp) {
  // At least one result.
  if (unbundleOp.getNumResults() == 0)
    return std::nullopt;
  // Every result has exactly one user.
  if (!llvm::all_of(unbundleOp.getResults(),
                    [](Value v) { return v.hasOneUse(); }))
    return std::nullopt;
  // Gather the consumers of the unbundle's results.
  auto consumers = llvm::map_to_vector(unbundleOp.getResults(), [](Value v) {
    return v.getUses().begin()->getOwner();
  });

  // Distinctness: a binary op consuming two unbundled elements (e.g.
  // `mo.add(u#0, u#1)`) would otherwise be nominated twice and we'd
  // clone/erase a single op instance more than once.
  llvm::SmallDenseSet<Operation *> unique;
  for (Operation *c : consumers)
    if (!unique.insert(c).second)
      return std::nullopt;

  // Pick a representative consumer that we can absorb into a parallel.
  Operation *rep = consumers[0];
  if (!isAbsorbableConsumer(rep))
    return std::nullopt;
  // All other consumers must be structurally identical to the representative.
  for (Operation *op : llvm::drop_begin(consumers))
    if (!opsMatch(rep, op))
      return std::nullopt;

  // Locate the driving operand position in the representative.  unbundle#0
  // has exactly one use (checked above), so it appears at exactly one
  // operand index of `rep`.
  OperandRange repOperands = rep->getOperands();
  auto it = llvm::find(repOperands, unbundleOp.getResult(0));
  ASSERT_STREAM(it != repOperands.end(),
                << "rep " << *rep << " is the unique user of "
                << unbundleOp.getResult(0) << " but does not reference it");
  size_t drivingOperandIdx = std::distance(repOperands.begin(), it);

  // Every peer must reference its own launch's unbundle result at the same
  // operand index.  Without this, cloning the rep once and substituting
  // block arguments by position would mis-route operands at runtime (e.g.
  // a commutative op matched in swapped operand order).
  for (auto [i, peer] : llvm::enumerate(llvm::drop_begin(consumers)))
    if (peer->getOperand(drivingOperandIdx) != unbundleOp.getResult(i + 1))
      return std::nullopt;

  return WrapCandidate{unbundleOp, std::move(consumers), drivingOperandIdx};
}

//===----------------------------------------------------------------------===//
// Input planning
//===----------------------------------------------------------------------===//

/// Plans the new parallel's inputs by walking the representative
/// consumer's operands.  The driving operand position
/// (`candidate.drivingOperandIdx`) is supplied by bundle #0 (the
/// unbundle's results); every other operand position gets its own
/// per-launch bundle.  Returns std::nullopt when:
///   * any non-driving operand is shared across all launches (would
///     require capturing an outer SSA value into the parallel body), or
///   * any non-driving operand can't be bundled (shape/dtype/device
///     mismatch).
static std::optional<InputPlan>
planInputs(WrapCandidate &candidate, ArrayRef<DeviceRefAttr> launchDevices) {
  size_t numLaunches = candidate.unbundleOp.getNumResults();
  size_t numOperands = candidate.consumerOps[0]->getNumOperands();

  InputPlan plan;
  plan.operandToBlockArg.assign(numOperands, 0);
  // Bundle #0 feeds the parallel from the driving unbundle's results;
  // the driving operand uses this slot.
  plan.bundles.emplace_back(candidate.unbundleOp.getResults().begin(),
                            candidate.unbundleOp.getResults().end());

  for (size_t idx = 0; idx < numOperands; ++idx) {
    if (idx == candidate.drivingOperandIdx)
      continue;

    SmallVector<Value> outerValues;
    outerValues.reserve(numLaunches);
    for (size_t launch = 0; launch < numLaunches; ++launch)
      outerValues.push_back(candidate.consumerOps[launch]->getOperand(idx));

    // TODO: Allow captured value from outside the parallel's body.
    if (llvm::all_equal(outerValues))
      return std::nullopt;
    if (!canBundlePerLaunch(outerValues, launchDevices))
      return std::nullopt;
    // Plan to create a new bundle input for these values.
    plan.bundles.push_back(std::move(outerValues));
    plan.operandToBlockArg[idx] = plan.bundles.size() - 1;
  }
  return plan;
}

//===----------------------------------------------------------------------===//
// Construction
//===----------------------------------------------------------------------===//

/// Builds a fresh mo.parallel downstream of \p candidate.unbundleOp
/// containing a single clone of the representative consumer, then routes
/// the original consumers' result users to the new parallel's per-launch
/// outputs and erases the consumers.
static void wrapCandidate(WrapCandidate &candidate, const InputPlan &plan,
                          OpBuilder &builder) {
  Location loc = candidate.unbundleOp.getLoc();
  builder.setInsertionPointAfter(candidate.unbundleOp);
  Operation *representative = candidate.consumerOps[0];

  // Emit a `mo.tensor.bundle` for each planned input.
  SmallVector<Value> inputs;
  for (ArrayRef<Value> values : plan.bundles) {
    SmallVector<Type> elemTypes =
        llvm::map_to_vector(values, [](Value v) { return v.getType(); });
    auto bundleTy = BundleType::get(builder.getContext(), elemTypes);
    inputs.push_back(
        TensorBundleOp::create(builder, loc, bundleTy, values).getResult());
  }

  // Create the output bundle.
  SmallVector<Type> outputElemTypes =
      llvm::map_to_vector(candidate.consumerOps, [](Operation *op) {
        return op->getResult(0).getType();
      });
  auto outputBundleTy = BundleType::get(builder.getContext(), outputElemTypes);

  // Create the parallel op.
  auto parallel =
      ParallelOp::create(builder, loc, inputs, TypeRange{outputBundleTy});

  // Body: one clone of the representative, with each operand remapped to
  // its assigned block argument.
  Block &body = parallel.getBodyRegion().front();
  IRMapping mapping;
  for (auto [i, blockArgIdx] : llvm::enumerate(plan.operandToBlockArg))
    mapping.map(representative->getOperand(i), body.getArgument(blockArgIdx));
  auto bodyBuilder = OpBuilder::atBlockBegin(&body);
  Operation *cloned = bodyBuilder.clone(*representative, mapping);
  YieldOp::create(bodyBuilder, loc, ValueRange{cloned->getResult(0)});

  // Route each consumer's sole result use to the corresponding per-launch
  // element of the new parallel's output bundle, then erase the consumer.
  builder.setInsertionPointAfter(parallel);
  auto newUnbundle = TensorUnbundleOp::create(builder, loc, outputElemTypes,
                                              parallel.getResult(0));
  for (auto [i, op] : llvm::enumerate(candidate.consumerOps)) {
    op->getResult(0).replaceAllUsesWith(newUnbundle.getResult(i));
    op->erase();
  }

  // Don't erase the original unbundle: bundle #0 still reads from it.
  // The transient `bundle(unbundle(X))` pair is folded away by the
  // post-pass canonicalize.
}

//===----------------------------------------------------------------------===//
// Pass driver
//===----------------------------------------------------------------------===//

/// Visits every `TensorUnbundleOp` reachable from \p root and tries to wrap
/// its consumer set.  Returns true if at least one wrap fired.
static bool runOneIteration(Operation *root, OpBuilder &builder) {
  // Gather tensor unbundle ops into a list.
  SmallVector<TensorUnbundleOp> unbundles;
  root->walk([&](TensorUnbundleOp op) { unbundles.push_back(op); });

  bool changed = false;
  for (TensorUnbundleOp unbundleOp : unbundles) {
    // Skip ops that have been erased by an earlier wrap this iteration.
    if (!unbundleOp.getOperation()->getParentOp())
      continue;
    // Find a candidate set of ops to wrap in a parallel region.
    auto candidate = findCandidate(unbundleOp);
    if (!candidate)
      continue;
    // Plan the parallel's inputs.
    auto inputBundleTy =
        cast<BundleType>(candidate->unbundleOp.getInput().getType());
    auto launchDevices = llvm::map_to_vector(
        inputBundleTy.getElementTypes(), [](Type t) -> DeviceRefAttr {
          return cast<TensorType>(t).getDeviceRef();
        });
    auto plan = planInputs(*candidate, launchDevices);
    if (!plan)
      continue;

    LLVM_DEBUG(llvm::dbgs() << "create-parallel-from-unbundle: wrapping "
                            << candidate->consumerOps.size() << " peer(s) of "
                            << candidate->unbundleOp << "\n");
    wrapCandidate(*candidate, *plan, builder);
    changed = true;
  }
  return changed;
}

struct CreateParallelFromUnbundlePass
    : public impl::CreateParallelFromUnbundleBase<
          CreateParallelFromUnbundlePass> {
  using CreateParallelFromUnbundleBase::CreateParallelFromUnbundleBase;

  void runOnOperation() override {
    OpBuilder builder{&getContext()};
    // Iterate to a fixed point.  Each wrap creates a fresh unbundle on the
    // new parallel's output, which may itself seed a downstream wrap.
    while (runOneIteration(getOperation(), builder)) {
    }
  }
};

} // namespace
