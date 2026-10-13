//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the MAX License:
// https://modular.com/legal/max-license
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//
//
// Merges two sibling foreaches over the same index space that share an input
// into one multi-output kernel, so the shared operand is loaded once. Unlike
// the other fusers this pairs siblings rather than a producer with a consumer.
// It also moves a consumer into its producer's epilogue when the producer's
// result has other readers, which output fusion leaves alone; one that reads a
// strided window of the result runs there under a `map.index.slice.select`.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MAPDialect/Dialect/MAPOps.h"
#include "GraphCompiler/MAPDialect/Support/OutputFusion.h"
#include "GraphCompiler/MAPDialect/Transforms/MAPTransforms.h"
#include "GraphCompiler/MODialect/MODevices.h"
#include "GraphCompiler/MODialect/MOOps.h"
#include "GraphCompiler/MOGGDialect/Support/FusionUtils.h"
#include "GraphCompiler/Support/DataflowUtils.h"

#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/DenseMap.h"

#include <tuple>

using namespace M;

namespace M::MAP {

#define GEN_PASS_DEF_FUSEHORIZONTAL
#include "GraphCompiler/MAPDialect/Transforms/MAPTransforms.h.inc"

namespace {

// The pair is rebuilt at `b`, so anything between the two that reads `a`'s
// results would end up ahead of its definition. A graph region allows that and
// the dependency walks already rule out a cycle, so this only keeps kernels in
// def-before-use order, conservatively. A reader inside a region (an `mo.if`
// branch) counts against the op that encloses it here, which is what actually
// moves relative to `b`.
bool resultsUnusedBefore(IterForeachOp a, IterForeachOp b) {
  Block *block = b->getBlock();
  return llvm::none_of(a->getUsers(), [&](Operation *user) {
    Operation *inBlock = block->findAncestorOpInBlock(*user);
    return inBlock && inBlock->isBeforeInBlock(b);
  });
}

// Whether `op`'s body loads from its input `in`.
bool loadsInput(IterForeachOp op, Value in) {
  Block &body = op.getBody().front();
  for (auto [i, operand] : llvm::enumerate(op.getInputs()))
    if (operand == in &&
        llvm::any_of(body.getArgument(i).getUsers(),
                     [](Operation *user) { return isa<LoadOp>(user); }))
      return true;
  return false;
}

// The kernels `fusible` can pair: same shape and device, both loading
// `loadedOperand`.
struct FusionCandidateGroup {
  Attribute shape;
  Attribute device;
  Value loadedOperand;

  std::tuple<Attribute, Attribute, Value> key() const {
    return {shape, device, loadedOperand};
  }
  bool operator==(const FusionCandidateGroup &other) const {
    return key() == other.key();
  }
};

// A kernel belongs to one group per operand it loads.
SmallVector<FusionCandidateGroup> fusionCandidateGroupsOf(IterForeachOp op) {
  SmallVector<FusionCandidateGroup> groups;
  for (Value in : op.getInputs()) {
    FusionCandidateGroup group{op.getShapeAttr(), MO::getDeviceOrDefault(op),
                               in};
    if (loadsInput(op, in) && !llvm::is_contained(groups, group))
      groups.push_back(group);
  }
  return groups;
}

bool fusible(IterForeachOp a, IterForeachOp b) {
  // 1) Check that both kernels have the same shape and device.
  if (a.getShapeAttr() != b.getShapeAttr())
    return false;
  if (MO::getDeviceOrDefault(a) != MO::getDeviceOrDefault(b))
    return false;

  // 2) Skip kernels with side effects (not supported for now).
  // A chain or a memory effect carries an ordering this pass doesn't model.
  if (a.getChain() || b.getChain() || a.getHasReadEffect() ||
      a.getHasWriteEffect() || b.getHasReadEffect() || b.getHasWriteEffect())
    return false;

  // A view chain becomes a zero-copy view after fusion, which a multi-result
  // kernel can't, so merging it would turn a free view into a copy.
  if (isRaisableViewChain(a) || isRaisableViewChain(b))
    return false;

  // 3) Only pair kernels that both load a shared operand.
  // Loading it once is the whole gain. An operand only read as slice bounds
  // (starts/stops/steps, shared by every window slice) saves nothing and would
  // steal the slice from the producer it could fuse into. Most pairs fail here,
  // so this runs before the dependency walks.
  if (!llvm::any_of(a.getInputs(), [&](Value in) {
        return llvm::is_contained(b.getInputs(), in) && loadsInput(a, in) &&
               loadsInput(b, in);
      }))
    return false;

  // 4) Check that no dependency between the 2 kernels makes the fused kernel
  // invalid.
  // The merged kernel takes `b`'s place, so nothing in between may read `a`'s
  // results, and neither kernel may depend on the other, even indirectly, or it
  // would consume its own result.
  if (!resultsUnusedBefore(a, b))
    return false;
  return !dataflowGraphDependsOn(b, a) && !dataflowGraphDependsOn(a, b);
}

// Whether a slice leaves a strided window of its input that no view can
// express -- the only kind worth a select in the producer's epilogue. A
// contiguous slice (every step 1, and at most one sliced dim ahead of the
// first non-unit one) is an offset view, better left to view fusion than paid
// for with a second store.
bool isStridedWindow(SliceSpecAttr spec) {
  MOSH::ShapeAttr in = spec.getSignature().getIn();
  MOSH::ShapeAttr out = spec.getSignature().getOut();
  ArrayRef<int64_t> steps = spec.getSteps().asArrayRef();
  if (llvm::any_of(steps, [](int64_t step) { return step != 1; }))
    return true;
  bool nonUnitDimSeen = false;
  for (unsigned d = 0, e = in.getRank(); d < e; ++d) {
    if (in.getValues()[d] != out.getValues()[d] && nonUnitDimSeen)
      return true;
    if (in.getIntDims()[d] != 1)
      nonUnitDimSeen = true;
  }
  return false;
}

// The slice through which a consumer reads the fused operand, if that is how
// it reads it: one load, through one `map.index.slice` of the loop index, in a
// body with one store at the loop index. The body then moves into a select,
// where the loop index is the slice's output index -- the consumer's own -- so
// elementwise work between the load and the store keeps its meaning there.
// Null when the read is not such a slice.
// TODO(slariau): also take loads of the consumer's other inputs at the loop
// index, which the select keeps in bounds; today they decline the merge.
SliceIndexOp matchSliceRead(BlockArgument fusionArg, BlockArgument loopIdx,
                            StoreOp store) {
  if (!fusionArg.hasOneUse())
    return nullptr;
  auto load = dyn_cast<LoadOp>(*fusionArg.getUsers().begin());
  if (!load)
    return nullptr;
  auto slice = load.getIndex().getDefiningOp<SliceIndexOp>();
  if (!slice || slice.getIndex() != loopIdx || !slice->hasOneUse())
    return nullptr;
  if (store.getIndex() != loopIdx)
    return nullptr;
  if (llvm::any_of(loopIdx.getUsers(), [&](Operation *user) {
        return user != slice.getOperation() && user != store.getOperation() &&
               !isa<CallElementwiseOp>(user);
      }))
    return nullptr;
  return slice;
}

// The producer's own output store and each consumer of that result write one
// element of the same index space, so they are siblings -- but one of them
// sits inside the producer, where `fuse` cannot reach it. So the merge goes
// the other way: the consumer moves into the producer's epilogue and stores
// beside the store already there, which is the one the producer still owes the
// readers fusion cannot absorb. Output fusion will not do this, since it takes
// a result with a single reader.
bool fuseEscapingReader(IterOpaqueOp &dst) {
  for (unsigned idx = 0, e = dst.getNumNonChainResults(); idx < e; ++idx) {
    Value res = dst->getResult(idx);
    if (!isa<MO::TensorType>(res.getType()))
      continue;

    IterForeachOp src;
    unsigned numConsumers = 0;
    bool escapes = false;
    for (OpOperand &use : res.getUses()) {
      auto consumer = dyn_cast<IterForeachOp>(use.getOwner());
      if (consumer && use.getOwner()->getBlock() == dst->getBlock()) {
        ++numConsumers;
        src = consumer;
        continue;
      }
      escapes = true;
    }
    // Without an escaping reader this is a plain producer/consumer edge that
    // output fusion already takes. More than one consumer waits for `fuse` to
    // merge them with each other first, since moving one in would strand the
    // other behind a producer it can no longer reach.
    if (!escapes || numConsumers != 1)
      continue;
    if (src.getChain() || src.getHasReadEffect() || src.getHasWriteEffect())
      continue;
    if (llvm::count(src.getInputs(), res) != 1)
      continue;
    if (llvm::any_of(src->getResultTypes(),
                     [](Type t) { return !isa<MO::TensorType>(t); }) ||
        llvm::any_of(dst->getResultTypes(),
                     [](Type t) { return !isa<MO::TensorType>(t); }))
      continue;

    IterEpilogueOp epilogueOp = dst.getEpilogueForResult(idx);
    if (!epilogueOp)
      continue;
    Block &epilogueBody = epilogueOp.getBody().front();
    SmallVector<StoreOp> epilogueStores = getOutputStores(epilogueBody);
    Block &srcBody = src.getBody().front();
    SmallVector<StoreOp> srcStores = getOutputStores(srcBody);
    if (epilogueStores.size() != 1 || srcStores.size() != src.getNumResults())
      continue;

    // The consumer must read the fused operand at its own loop index -- the
    // epilogue hands over one element, with no way to honor a transform --
    // or through a strided slice of it, which a select inverts: the consumer
    // then runs only where the slice selects the element, at its own index.
    unsigned fusionArgIdx =
        llvm::find(src.getInputs(), res) - src.getInputs().begin();
    BlockArgument fusionArg = srcBody.getArgument(fusionArgIdx);
    BlockArgument srcLoopIdx = srcBody.getArguments().back();
    SliceIndexOp slice;
    SliceSpecAttr sliceSpec;
    if (llvm::any_of(fusionArg.getUsers(), [&](Operation *user) {
          auto load = dyn_cast<LoadOp>(user);
          return !load || load.getIndex() != srcLoopIdx;
        })) {
      StoreOp epilogueStore = epilogueStores.front();
      if (srcStores.size() != 1 ||
          !isa<MO::TensorType>(epilogueStore.getTensor().getType()))
        continue;
      // The select would sit at the epilogue store's index, so the merge
      // checks the select's rules there. They reject a transformed index,
      // such as a fused reshape's, which can carry one stored vector across
      // rows of the slice's input.
      slice = matchSliceRead(fusionArg, srcLoopIdx, srcStores.front());
      if (!slice)
        continue;
      sliceSpec = slice.getSpec();
      if (failed(SliceSelectOp::verifySpecAt(sliceSpec,
                                             epilogueStore.getIndex())) ||
          !isStridedWindow(sliceSpec))
        continue;
    }
    // Another operand reaching back through `dst` would make the merge a cycle.
    if (llvm::any_of(src->getOperands(), [&](Value other) {
          return other != res && dataflowGraphDependsOn(other, dst);
        }))
      continue;

    IRRewriter rewriter(dst.getContext());
    unsigned numDpsBefore = dst.getNumNonChainResults();
    dst = growOutputs(dst, src->getResultTypes(), rewriter);
    epilogueOp = dst.getEpilogueForResult(idx);
    Block &body = epilogueOp.getBody().front();

    StoreOp epilogueStore = getOutputStores(body).front();
    Value val = epilogueStore.getValue();
    Value storeIdx = epilogueStore.getIndex();
    Location loc = epilogueStore.getLoc();

    // The fused operand's block arg is a tensor; `val` is already the scalar
    // `dst` produced for this element, so every load of it becomes `val`.
    for (Operation *user : llvm::to_vector(fusionArg.getUsers())) {
      auto load = cast<LoadOp>(user);
      load.getResult().replaceAllUsesWith(val);
      load.erase();
    }
    // The select takes over the slice, spec and all.
    if (slice)
      slice.erase();

    // Every other input the body still reads becomes a capture, threaded
    // through a new input of `dst` -- the epilogue is isolated from above. A
    // slice's bounds tensors are no longer read, since the select carries them
    // statically.
    for (auto [i, input] : llvm::enumerate(src.getInputs())) {
      if (i == fusionArgIdx || srcBody.getArgument(i).use_empty())
        continue;
      unsigned dstInputIdx = dst.getInputs().size();
      dst.insertInputsAt(dstInputIdx, input);
      epilogueOp.getCapturesMutable().append(
          dst.getInputBlockArgs()[dstInputIdx]);
      srcBody.getArgument(i).replaceAllUsesWith(
          body.addArgument(input.getType(), input.getLoc()));
    }

    // The grown output block args are the consumer's destinations.
    SmallVector<Value> caps;
    Block::BlockArgListType outArgs = dst.getOutputBlockArgs();
    for (unsigned i = 0, n = src->getNumResults(); i < n; ++i) {
      BlockArgument outArg = outArgs[numDpsBefore + i];
      epilogueOp.getCapturesMutable().append(outArg);
      caps.push_back(body.addArgument(outArg.getType(), outArg.getLoc()));
    }

    SmallVector<Value> storedVals;
    for (StoreOp store : getOutputStores(srcBody))
      storedVals.push_back(store.getValue());
    for (StoreOp store : getOutputStores(srcBody))
      store.erase();

    // The producer's own store stays: it is what the escaping reader is owed.
    // Through a slice, the consumer's body moves into a select, which runs it
    // at the slice's index and only where the slice selects the element.
    Block *consumerBody = &body;
    Value consumerIdx = storeIdx;
    if (sliceSpec) {
      rewriter.setInsertionPointToEnd(&body);
      auto select = SliceSelectOp::create(rewriter, loc, sliceSpec, storeIdx);
      consumerBody =
          rewriter.createBlock(&select.getBody(), select.getBody().end(),
                               {storeIdx.getType()}, {loc});
      consumerIdx = consumerBody->getArgument(0);
    }
    srcLoopIdx.replaceAllUsesWith(consumerIdx);
    srcBody.eraseArguments(0, srcBody.getNumArguments());
    rewriter.inlineBlockBefore(&srcBody, consumerBody, consumerBody->end());
    rewriter.setInsertionPointToEnd(consumerBody);
    for (auto [cap, storedVal] : llvm::zip_equal(caps, storedVals))
      StoreOp::create(rewriter, loc, cap, consumerIdx, storedVal);

    SmallVector<Value> replacements;
    for (unsigned i = 0, n = src->getNumResults(); i < n; ++i)
      replacements.push_back(dst->getResult(numDpsBefore + i));
    src->replaceAllUsesWith(replacements);

    dst.setFusionInfoAttr(MOGG::createFusionInfo<MOGG::FusionEpilogueAttr>(
        dst.getContext(), {dst.getOperation(), src.getOperation()},
        {static_cast<uint32_t>(idx)}));
    dst->setLoc(
        mlir::FusedLoc::get(dst.getContext(), {src->getLoc(), dst->getLoc()}));
    src->erase();
    dedupeKernelInputs(dst, rewriter);
    return true;
  }
  return false;
}

void fuse(IterForeachOp a, IterForeachOp b) {
  // Merged inputs: `a`'s, then the ones only `b` carries. A shared operand
  // keeps `a`'s slot, which is what collapses the duplicate load.
  SmallVector<Value> inputs(a.getInputs());
  SmallVector<unsigned> slotForBInput;
  for (Value in : b.getInputs()) {
    auto *it = llvm::find(inputs, in);
    slotForBInput.push_back(std::distance(inputs.begin(), it));
    if (it == inputs.end())
      inputs.push_back(in);
  }

  SmallVector<Type> resultTypes(a->getResultTypes());
  llvm::append_range(resultTypes, b->getResultTypes());

  MLIRContext *ctx = a.getContext();
  Location loc = mlir::FusedLoc::get(ctx, {a.getLoc(), b.getLoc()});
  OpBuilder builder(b);
  // Deferred if either member is, as input fusion combines the launch policy.
  auto fused = IterForeachOp::create(
      builder, loc, resultTypes, a.getShapeAttr(), inputs,
      /*chain=*/Value(),
      /*fusionInfo=*/
      MOGG::createFusionInfo<MOGG::FusionHorizontalAttr>(
          ctx, {a.getOperation(), b.getOperation()}),
      /*hasReadEffect=*/false, /*hasWriteEffect=*/false,
      /*dispatch=*/
      a.getDispatch() == MOGG::DispatchKind::Deferred ||
              b.getDispatch() == MOGG::DispatchKind::Deferred
          ? MOGG::DispatchKind::Deferred
          : MOGG::DispatchKind::Immediate);

  // `device` and friends are discardable, so `create` does not carry them.
  // The pair shares its device; on any other clash `a`'s attribute wins.
  NamedAttrList attrs(b->getDiscardableAttrDictionary());
  for (NamedAttribute attr : a->getDiscardableAttrs())
    attrs.set(attr.getName(), attr.getValue());
  fused->setDiscardableAttrs(attrs.getDictionary(ctx));

  Block &aBody = a.getBody().front();
  Block *body = &fused.getBody().emplaceBlock();
  for (Value in : inputs)
    body->addArgument(in.getType(), loc);
  for (Type ty : resultTypes)
    body->addArgument(ty, loc);
  body->addArgument(aBody.getArguments().back().getType(), loc);

  OpBuilder bodyBuilder = OpBuilder::atBlockEnd(body);
  BlockArgument index = body->getArguments().back();
  unsigned outputBase = inputs.size();

  auto cloneBody = [&](IterForeachOp src, ArrayRef<unsigned> inputSlots,
                       unsigned firstOutput) {
    // Body args are [inputs..., outputs..., index], as MOToMAP builds them.
    Block &srcBody = src.getBody().front();
    unsigned numInputs = src.getInputs().size();
    IRMapping map;
    for (unsigned i = 0; i < numInputs; ++i)
      map.map(srcBody.getArgument(i), body->getArgument(inputSlots[i]));
    for (unsigned i = 0, e = src->getNumResults(); i < e; ++i)
      map.map(srcBody.getArgument(numInputs + i),
              body->getArgument(outputBase + firstOutput + i));
    map.map(srcBody.getArguments().back(), index);
    for (Operation &op : srcBody)
      bodyBuilder.clone(op, map);
  };

  SmallVector<unsigned> aSlots(
      llvm::to_vector(llvm::seq<unsigned>(0, a.getInputs().size())));
  cloneBody(a, aSlots, 0);
  cloneBody(b, slotForBInput, a->getNumResults());

  unsigned split = a->getNumResults();
  a.replaceAllUsesWith(fused.getResults().take_front(split));
  b.replaceAllUsesWith(fused.getResults().drop_front(split));
  b.erase();
  a.erase();
}

struct FuseHorizontalPass
    : public MAP::impl::FuseHorizontalBase<FuseHorizontalPass> {
  using Base::Base;
  FuseHorizontalPass(const FuseHorizontalPass &other) : Base(other) {}

  void runOnOperation() override {
    // Fusion stays within a block, so each block runs to its own fixpoint, as
    // in `fuseOutput`. Blocks are not fully independent: a select merge in a
    // branch drops the slice's bounds operands, which can remove the capture
    // an outer pair's cycle check went through, so what fuses can depend on
    // the order blocks are visited in. Every order gives valid IR. Unlike in
    // `fuseOutput`, a block is not sorted first: the greedy choices follow
    // block order.
    getOperation().getBody().walk([&](Block *block) { fuseBlock(*block); });
  }

  void fuseBlock(Block &block) {
    // Pairing can leave a producer one consumer for the merge, and a merge can
    // expose a new pair, so alternate until neither fuses anything.
    while (true) {
      bool pairedSiblings = fuseSiblings(block);
      bool mergedReaders = fuseEscapingReaders(block);
      if (!pairedSiblings && !mergedReaders)
        break;
    }
  }

  // Returns whether any pair was fused.
  bool fuseSiblings(Block &block) {
    bool fusedAny = false;
    // Fusing rebuilds both kernels, so the kernel list and groups are rebuilt
    // after each pair. The scan resumes at the merged pair's first kernel, not
    // at the top: a merged kernel reads everything both members read, so
    // every pair rejected before it is still rejected.
    unsigned start = 0;
    bool changed = true;
    while (changed) {
      changed = false;
      auto kernels = llvm::to_vector(block.getOps<IterForeachOp>());

      // A pair that shares no group fails `fusible`, so a kernel is tried
      // only against the later kernels of its own groups. They are still
      // visited in block order, which the greedy choice depends on.
      SmallVector<SmallVector<FusionCandidateGroup>> groupsOf;
      DenseMap<std::tuple<Attribute, Attribute, Value>, SmallVector<unsigned>>
          members;
      for (auto [idx, kernel] : llvm::enumerate(kernels)) {
        groupsOf.push_back(fusionCandidateGroupsOf(kernel));
        for (const FusionCandidateGroup &group : groupsOf.back())
          members[group.key()].push_back(idx);
      }

      for (unsigned i = start, e = kernels.size(); i < e && !changed; ++i) {
        SmallVector<unsigned> candidates;
        for (const FusionCandidateGroup &group : groupsOf[i])
          for (unsigned j : members.at(group.key()))
            if (j > i)
              candidates.push_back(j);
        llvm::sort(candidates);
        candidates.erase(llvm::unique(candidates), candidates.end());
        for (unsigned j : candidates) {
          IterForeachOp a = kernels[i];
          IterForeachOp b = kernels[j];
          if (!fusible(a, b))
            continue;
          fuse(a, b);
          ++numKernelsFused;
          // `a` is gone, so the next kernel in block order now sits at `i`.
          start = i;
          changed = true;
          fusedAny = true;
          break;
        }
      }
    }
    return fusedAny;
  }

  // Returns whether any consumer was merged.
  bool fuseEscapingReaders(Block &block) {
    bool mergedAny = false;
    // Merging a consumer into a producer's epilogue rebuilds the producer, so
    // the scan restarts after each one.
    bool merged = true;
    while (merged) {
      merged = false;
      auto opaques = llvm::to_vector(block.getOps<IterOpaqueOp>());
      for (IterOpaqueOp dst : opaques) {
        if (fuseEscapingReader(dst)) {
          ++numKernelsFused;
          merged = true;
          mergedAny = true;
          break;
        }
      }
    }
    return mergedAny;
  }

  Statistic numKernelsFused{this, "num-kernels-fused",
                            "Number of kernels fused"};
};

} // namespace

} // namespace M::MAP
