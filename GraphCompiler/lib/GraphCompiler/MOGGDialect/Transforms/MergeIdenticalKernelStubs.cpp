//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "AsyncRT/CompilerSupport/Context.h"
#include "GraphCompiler/GraphCompiler/MODialect/MOOps.h"
#include "GraphCompiler/GraphCompiler/MOGGDialect/MOGGDialect.h"
#include "GraphCompiler/GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GraphCompiler/GraphCompiler/MOGGDialect/MOGGPasses.h"
#include "KGEN/TransformUtils/SCCUtils.h"
#include "MLRT/GraphRT/GRTDialect/GRTOps.h"
#include "Support/ML/CommonAttributes.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/Hashing.h"

using namespace M;

namespace M::MOGG {

#define GEN_PASS_DEF_MERGEIDENTICALKERNELSTUBS
#include "GraphCompiler/GraphCompiler/MOGGDialect/MOGGPasses.h.inc"

} // namespace M::MOGG

namespace M::MOGG {

namespace {

// Instead of storing `KernelStubOp` directly, we can store it with its
// precomputed hash. This speeds up multiple insertions of the same stub op
// because the part of the code that inserts the op also has access to the hash.
struct KernelStubHashEntry {
  MOGG::KernelStubOp op;
  unsigned hash;

  // Will be called in isEqual only.
  bool operator==(const KernelStubHashEntry &other) const {
    return other.op == op;
  }
};

struct KernelStubOpEquivalenceInfo
    : public llvm::DenseMapInfo<KernelStubHashEntry> {

  // We first compute the hash of a node and store it with the entry in the
  // dense set. This is an optimization where subsequent insertions of the same
  // op will not rehash. We will be comparing two entries when the hash collides
  // though but this is less frequent than rehashing every time we try to insert
  // something.
  static unsigned getHashValue(const KernelStubHashEntry cStubOp) {
    return cStubOp.hash;
  }
  static bool isEqual(KernelStubHashEntry lhs, KernelStubHashEntry rhs) {
    if (lhs.op == rhs.op)
      return true;

    // Check if properties are equivalent modulo sym_name and fusionInfo
    // attributes.
    auto lhsProperties = lhs.op.getProperties();
    auto rhsProperties = rhs.op.getProperties();

    lhsProperties.sym_name = rhsProperties.sym_name = {};
    lhsProperties.fusionInfo = rhsProperties.fusionInfo = {};

    if (lhsProperties != rhsProperties)
      return false;

    // Check if discardable attributes are equivalent.
    DictionaryAttr lhsDictAttr = lhs.op->getRawDictionaryAttrs();
    DictionaryAttr rhsDictAttr = rhs.op->getRawDictionaryAttrs();
    if (lhsDictAttr.size() != rhsDictAttr.size())
      return false;

    for (auto [lhsAttr, rhsAttr] : llvm::zip_equal(lhsDictAttr, rhsDictAttr)) {
      if (lhsAttr.getName() != rhsAttr.getName())
        return false;

      if (lhsAttr.getValue() == rhsAttr.getValue())
        continue;

      auto attrName = lhsAttr.getName();
      if (attrName != MO::kOperationDeviceRefAttrName)
        return false;

      // Do not compare the device ID, just the label.
      if (cast<DeviceRefAttr>(lhsAttr.getValue()).getLabel() !=
          cast<DeviceRefAttr>(rhsAttr.getValue()).getLabel())
        return false;
    }

    return mlir::OperationEquivalence::isRegionEquivalentTo(
        &lhs.op.getBodyRegion(), &rhs.op.getBodyRegion(),
        /*flags=*/mlir::OperationEquivalence::Flags::IgnoreLocations);
  }
};

static unsigned getHashValue(const MOGG::KernelStubOp cStubOp) {
  llvm::hash_code hash = {};
  MOGG::KernelStubOp stubOp = const_cast<MOGG::KernelStubOp &>(cStubOp);

  auto properties = stubOp.getProperties();
  properties.sym_name = {};
  properties.fusionInfo = {};

  stubOp->getName().hashOpProperties(mlir::PropertyRef(
      stubOp->getName().getOpPropertiesTypeID(), &properties));

  for (NamedAttribute namedAttr : cStubOp->getDiscardableAttrDictionary()) {
    StringAttr attrName = namedAttr.getName();

    // Do not hash the device ID, just the label.
    if (attrName == MO::kOperationDeviceRefAttrName) {
      hash = llvm::hash_combine(
          hash, cast<DeviceRefAttr>(namedAttr.getValue()).getLabel());
      continue;
    }

    hash = llvm::hash_combine(hash, namedAttr);
  }

  stubOp.getBodyBlock().walk([&](Operation *op) {
    hash = llvm::hash_combine(
        hash,
        mlir::OperationEquivalence::computeHash(
            op, /*hashOperands=*/mlir::OperationEquivalence::ignoreHashValue,
            /*hashResults=*/mlir::OperationEquivalence::ignoreHashValue,
            /*flags=*/mlir::OperationEquivalence::Flags::IgnoreLocations));
  });

  return hash;
}

// The call graph edge is a `StubCallOpInterface`, implemented by both
// `mogg.call.stub` and `mogg.call.stub.async`. Using the interface as the
// edge type lets the generic `CallGraphBase::build` discover both synchronous
// and asynchronous stub calls as edges, so a caller stub is canonicalized
// before it is hashed for deduplication.
struct CallGraphNode : public KGEN::SCCNode<CallGraphNode, MOGG::KernelStubOp,
                                            MOGG::StubCallOpInterface> {
  using Base = KGEN::SCCNode<CallGraphNode, MOGG::KernelStubOp,
                             MOGG::StubCallOpInterface>;
  using Base::Base;

  unsigned hash;
};

struct CallStubOrCalleeName {
  MOGG::StubCallOpInterface callOp;
  StringAttr name;

  CallStubOrCalleeName(MOGG::StubCallOpInterface call)
      : callOp(call), name(call.getCalleeSymbol().getAttr()) {}
  CallStubOrCalleeName(StringAttr n) : callOp(nullptr), name(n) {}
};

struct CallGraph : public KGEN::SCCGraph<CallGraph, CallGraphNode,
                                         /*BottomUpRewrite=*/true> {
  bool doAnalysis(CallGraphNode *node) { return false; }
  // For all callsites inside the current node, rewrite them if you aren't
  // able to insert them in the hash table.
  void doRewrite(CallGraphNode *node);

  DenseSet<KernelStubHashEntry, KernelStubOpEquivalenceInfo> uniqueFuncSet;

  // A mapping that shows the renaming of kernels that aren't called by any
  // other kernel (top-level). This is necessary because the CallGraph will only
  // replace the ones that get called. At the very end in runOnOperation, we do
  // the replacement for the top-level kernels using this mapping.
  llvm::DenseMap<StringAttr, StringAttr> topLevelReplacements;

  // A map of replacement-replaced that will be used at the very end to
  // stabilize the replacements (execution order will determine the replacement
  // without this which is non deterministic).
  llvm::DenseMap<StringAttr, SmallVector<CallStubOrCalleeName>>
      reverseReplacements;

  // The mutex to protect uniqueFuncSet accesses.
  std::mutex setAccessMutex;
};

void CallGraph::doRewrite(CallGraphNode *node) {
  for (auto [caller, callee] : node->callsites) {
    MOGG::KernelStubOp replacementCallee = nullptr;
    {
      std::lock_guard<std::mutex> lock(setAccessMutex);
      replacementCallee =
          uniqueFuncSet.insert(KernelStubHashEntry{callee->func, callee->hash})
              .first->op;
      reverseReplacements[replacementCallee.getSymNameAttr()].emplace_back(
          caller);
    }

    if (replacementCallee != callee->func) {
      // Merge locations so source notes from all merged stubs survive.
      // This preserves Python call site metadata for error messages and
      // NVTX trace detail strings.
      replacementCallee->setLoc(mlir::FusedLoc::get(
          replacementCallee->getContext(),
          {replacementCallee->getLoc(), callee->func->getLoc()}));
      caller.setCalleeSymbol(
          FlatSymbolRefAttr::get(replacementCallee.getSymNameAttr()));
    }
  }

  node->hash = getHashValue(node->func);
  // Try to insert yourself if no CallGraphNode is calling you. If another top
  // level stub (that has no callers) hashes to the same value, store the
  // replacement in another map that will be used after the analysis to replace
  // the external calls (see mergeIdenticalKernelStubs below).
  if (node->callers.empty()) {
    MOGG::KernelStubOp replacementCallee = nullptr;
    std::lock_guard<std::mutex> lock(setAccessMutex);
    replacementCallee =
        uniqueFuncSet.insert(KernelStubHashEntry{node->func, node->hash})
            .first->op;
    if (replacementCallee != node->func) {
      replacementCallee->setLoc(mlir::FusedLoc::get(
          node->func->getContext(),
          {replacementCallee->getLoc(), node->func->getLoc()}));
    }
    reverseReplacements[replacementCallee.getSymNameAttr()].emplace_back(
        node->func.getSymNameAttr());
    topLevelReplacements[node->func.getSymNameAttr()] =
        replacementCallee.getSymNameAttr();
  }
}

void mergeIdenticalKernelStubs(ModuleOp module) {
  SymbolTable symTable(module);
  CallGraph cg;
  cg.build(module, symTable);
  cg.run(*loadContext(module.getContext())->get<AsyncRT::CPUDevice>());

  // At this point, the replacement is valid but non deterministic because the
  // analysis will pick the first candidate when deduplicating. The first
  // candidate is a function of the execution order. To fix this, we keep track
  // of all the stubs that got renamed and then sort them by name. Then, we pick
  // the first candidate, which makes the analysis deterministic.
  llvm::DenseMap<StringAttr, StringAttr> finalReplacements;
  for (auto &[replacement, replacedCallOps] : cg.reverseReplacements) {
    if (replacedCallOps.size() == 1)
      continue;

    auto finalCandidate =
        *llvm::min_element(replacedCallOps, [](CallStubOrCalleeName lhs,
                                               CallStubOrCalleeName rhs) {
          return lhs.name.compare(rhs.name) < 0;
        });
    finalReplacements[replacement] = finalCandidate.name;

    // Rewrite any in-graph callsite entries (CallStubOp / CallStubAsyncOp) to
    // point at the chosen canonical name. The list may also hold pure
    // StringAttr entries (callOp == nullptr) for top-level KernelStubOps
    // recorded on the `node->callers.empty()` path above; those have no
    // in-graph callsite to rewrite here and are separately handled by the
    // ExecuteCompiledOp/CallStubAsyncOp walk below via `topLevelReplacements` +
    // `finalReplacements`.
    for (auto &callOp : replacedCallOps)
      if (callOp.callOp)
        callOp.callOp.setCalleeSymbol(
            FlatSymbolRefAttr::get(finalCandidate.name));
  }

  // Rewrite external callers that are not call-graph nodes. Stub calls *inside
  // KernelStubOp bodies* are graph edges (already rewritten in doRewrite), but
  // two callers are not graph nodes and still need their symbol references
  // updated to the deduplicated (top-level) callee:
  //   - ExecuteCompiledOp in mgp.ModelOps (ModelOp -> region), and
  //   - StubCallOpInterface ops (mogg.call.stub{,.async}) inside pre-outline
  //     mogg.generic.compiled.region ops (before kernels are outlined into
  //     KernelStubOp bodies).
  // For stub calls that *are* graph edges, the callee has a graph caller and
  // so is never in `topLevelReplacements`, making this walk a no-op for them.
  module.walk([&](Operation *op) {
    StringAttr calleeName;
    if (auto execOp = dyn_cast<MOGG::ExecuteCompiledOp>(op))
      calleeName = execOp.getCalleeSymbolAttr().getAttr();
    else if (auto stubCall = dyn_cast<MOGG::StubCallOpInterface>(op))
      calleeName = stubCall.getCalleeSymbol().getAttr();
    else
      return;

    auto it = cg.topLevelReplacements.find(calleeName);
    if (it == cg.topLevelReplacements.end())
      return;

    auto replacement = finalReplacements.find(it->second);
    if (replacement == finalReplacements.end())
      return;

    auto newRef = mlir::FlatSymbolRefAttr::get(replacement->second);
    if (auto execOp = dyn_cast<MOGG::ExecuteCompiledOp>(op))
      execOp.setCalleeSymbolAttr(newRef);
    else
      cast<MOGG::StubCallOpInterface>(op).setCalleeSymbol(newRef);
  });
}

struct MergeIdenticalKernelStubs
    : public MOGG::impl::MergeIdenticalKernelStubsBase<
          MergeIdenticalKernelStubs> {
  using MergeIdenticalKernelStubsBase::MergeIdenticalKernelStubsBase;

  void runOnOperation() override { mergeIdenticalKernelStubs(getOperation()); }
};

} // namespace

} // namespace M::MOGG
