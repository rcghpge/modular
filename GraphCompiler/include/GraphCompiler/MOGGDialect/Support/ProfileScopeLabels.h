//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Shared location-walking helpers for the Graph profile scope NVTX
// labeling feature. `Graph.profile_scope(name, color=None)`
// (max/python/max/graph/graph.py) wraps the MLIR location of every op created
// inside its `with` block in a custom `ProfileScopeLocationAttr` named
// `profile_scope(name, color=...)`. Nesting one profile_scope inside
// another wraps progressively outward, so the *last*-entered (innermost)
// scope ends up as the outermost `ProfileScopeLocationAttr` layer on the
// op's location -- see graph.py's `_location()`. Every mechanism that
// uses those labels (the per-kernel suffix in
// `WithTraceOp::emitMojo`, and the in-region range names added in
// `AddTryExceptBlocks.cpp`) reads them through this shared set of functions.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MOGGDIALECT_SUPPORT_PROFILESCOPELABELS_H
#define GENERICML_GRAPHCOMPILER_MOGGDIALECT_SUPPORT_PROFILESCOPELABELS_H

#include "GraphCompiler/MOGGDialect/MOGGAttrs.h"
#include "Support/ADT/DenseStringMap.h"
#include "mlir/IR/Location.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <optional>
#include <string>

namespace M::MOGG {

/// One Graph profile_scope nesting layer.
/// Outermost scope first (e.g. `{"draft_forward", "inner_stage"}` for
/// `with profile_scope("draft_forward"): with profile_scope("inner_stage"):`).
using ProfileScopeLayer = llvm::SmallVector<std::string, 2>;

} // namespace M::MOGG

namespace M::MOGG {

/// A contiguous run of ops inside a region body that share the same
/// profile_scope layer set, as discovered by
/// `AddTryExceptBlocks::addProfileScopeTracing`.
struct ProfileScopeLayerSet {
  mlir::Operation *first = nullptr;
  mlir::Operation *last = nullptr;
  llvm::SmallVector<ProfileScopeLayer, 2> layers;
};

namespace detail {

/// Storage for the set of profile-scope layers discovered during a walk.
using ProfileScopeLayerSetVector = llvm::SmallSetVector<ProfileScopeLayer, 2>;

/// Forward declaration: the helpers below recurse through this function.
inline void walkProfileScopeLayer(mlir::Location loc,
                                  llvm::ArrayRef<std::string> prefix,
                                  ProfileScopeLayerSetVector &layers);

/// Recurses into each sub-location as a peer boundary: layers found below are
/// added directly without folding through any outer prefix (none is set here).
/// Used at the walk root and at the `FusedLoc`/`CallSiteLoc` nodes that turn
/// out to be peer boundaries.
inline void recurseLocPeers(llvm::ArrayRef<mlir::Location> subs,
                            ProfileScopeLayerSetVector &layers) {
  for (mlir::Location sub : subs)
    walkProfileScopeLayer(sub, {}, layers);
}

/// Folds each sub-location's sub-layers into the accumulated prefix: for every
/// layer found below, prepend `prefix`; if a sub-location produced no layers
/// at all, emit `prefix` alone -- the active scopes above still apply even when
/// nothing concrete was reachable below.
inline void foldLocsIntoPrefix(llvm::ArrayRef<mlir::Location> subs,
                               llvm::ArrayRef<std::string> prefix,
                               ProfileScopeLayerSetVector &layers) {
  for (mlir::Location sub : subs) {
    ProfileScopeLayerSetVector subLayers;
    walkProfileScopeLayer(sub, {}, subLayers);
    for (const ProfileScopeLayer &subLayer : subLayers) {
      ProfileScopeLayer combined;
      combined.reserve(prefix.size() + subLayer.size());
      combined.append(prefix.begin(), prefix.end());
      combined.append(subLayer.begin(), subLayer.end());
      layers.insert(std::move(combined));
    }
    if (subLayers.empty())
      layers.insert(ProfileScopeLayer(prefix.begin(), prefix.end()));
  }
}

/// Collect every reachable profile_scope nesting layer
/// 'prefix` is the sequence of scope labels already
/// accumulated from outer `ProfileScopeLocationAttr` wrappers, in
/// outermost-first order. When the walk reaches a leaf, `prefix` is emitted
/// as a completed layer (if non-empty).
///
/// `FusedLoc` and `CallSiteLoc` are treated as peer boundaries only when
/// `prefix` is empty: at the walk root, each sub-location is an independent
/// layer because a compiler pass (CSE, SequenceFusion,
/// MergeIdenticalKernelStubs, ...) may have merged locations from genuinely
/// different, non-nested `profile_scope()` calls. When `prefix` is non-empty,
/// the composite node is nested inside one or more active scopes (e.g.
/// `AttachPythonStackTraces` rewrote an `OpaqueLoc` leaf into a `FusedLoc`),
/// so the accumulated labels prefix every sub-layer discovered below.
inline void walkProfileScopeLayer(mlir::Location loc,
                                  llvm::ArrayRef<std::string> prefix,
                                  ProfileScopeLayerSetVector &layers) {
  if (auto profileLoc = mlir::dyn_cast<ProfileScopeLocationAttr>(loc)) {
    // `ProfileScopeLocationAttr` wraps innermost scope outermost in the
    // location tree, so the scope encountered here encloses the scopes in
    // `prefix`. Prepend it so the prefix stays outermost-first.
    ProfileScopeLayer nextPrefix;
    nextPrefix.reserve(prefix.size() + 1);
    nextPrefix.push_back(profileLoc.getName().str());
    nextPrefix.append(prefix.begin(), prefix.end());
    walkProfileScopeLayer(profileLoc.getChildLoc(), nextPrefix, layers);
    return;
  }
  if (auto fusedLoc = mlir::dyn_cast<mlir::FusedLoc>(loc)) {
    if (prefix.empty())
      recurseLocPeers(fusedLoc.getLocations(), layers);
    else
      foldLocsIntoPrefix(fusedLoc.getLocations(), prefix, layers);
    return;
  }
  if (auto callSiteLoc = mlir::dyn_cast<mlir::CallSiteLoc>(loc)) {
    if (prefix.empty()) {
      walkProfileScopeLayer(callSiteLoc.getCallee(), prefix, layers);
      walkProfileScopeLayer(callSiteLoc.getCaller(), prefix, layers);
    } else {
      foldLocsIntoPrefix({callSiteLoc.getCallee(), callSiteLoc.getCaller()},
                         prefix, layers);
    }
    return;
  }
  // FileLineColLoc / FileLineColRange / OpaqueLoc / UnknownLoc are leaves as
  // far as scope labels go.
  if (!prefix.empty())
    layers.insert(ProfileScopeLayer(prefix.begin(), prefix.end()));
}

inline void collectProfileScopeLayersImpl(mlir::Location loc,
                                          ProfileScopeLayerSetVector &layers) {
  walkProfileScopeLayer(loc, {}, layers);
}

/// Orders two peer layers for canonical sorting. Layers are peers -- an
/// *unordered* set discovered via whichever order a compiler pass happened to
/// list a `FusedLoc`'s inputs in, not anything scope-aware (see
/// `collectProfileScopeLayers`)
inline bool profileScopeLayerLess(const ProfileScopeLayer &a,
                                  const ProfileScopeLayer &b) {
  return std::lexicographical_compare(a.begin(), a.end(), b.begin(), b.end());
}

/// Collect every distinct color from a `ProfileScopeLocationAttr` with a
/// non-empty color field.
inline void
collectProfileScopeColorsImpl(mlir::Location loc,
                              llvm::SmallVectorImpl<std::string> &colors) {
  mlir::AttrTypeWalker walker;
  walker.addWalk([&](ProfileScopeLocationAttr profileLoc) {
    if (auto color = profileLoc.getColor()) {
      if (!llvm::is_contained(colors, color))
        colors.push_back(color.str());
    }
    return mlir::WalkResult::advance();
  });
  walker.walk(loc);
}

/// Join layers using `/` within one nesting layer (outermost
/// first), `:` between distinct (peer) layers.
inline void joinProfileScopeLayers(llvm::ArrayRef<ProfileScopeLayer> layers,
                                   llvm::raw_ostream &os) {
  llvm::interleave(
      layers,
      [&](const ProfileScopeLayer &layer) {
        llvm::interleave(
            layer, [&](const std::string &s) { os << s; }, [&] { os << '/'; });
      },
      [&] { os << ':'; });
}

} // namespace detail

/// Collect distinct nesting layers - deduplicated and canonically ordered.
///
/// A single layer represents scope nesting.  Different layers result from
/// compiler pass merges and are sorted for comparison and formatting.
inline llvm::SmallVector<ProfileScopeLayer, 2>
collectProfileScopeLayers(mlir::Location loc) {
  detail::ProfileScopeLayerSetVector layers;
  detail::collectProfileScopeLayersImpl(loc, layers);

  auto out = layers.takeVector();
  llvm::sort(out, detail::profileScopeLayerLess);
  return out;
}

/// Collects every distinct profile scope color in sorted order.
inline llvm::SmallVector<std::string, 2>
collectProfileScopeColors(mlir::Location loc) {
  llvm::SmallVector<std::string, 2> colors;
  detail::collectProfileScopeColorsImpl(loc, colors);
  llvm::sort(colors);
  return colors;
}

/// Conflicting colors (the range's ops disagree on which color to use, e.g.
/// two profile scope calls used the same scope name with different
/// colors, or two peer scopes merged into one layer set each requested a
/// different color) resolve to no color rather than arbitrarily picking
/// a color.
inline std::optional<std::string>
resolveProfileScopeColor(llvm::ArrayRef<std::string> colors) {
  if (colors.size() == 1)
    return colors.front();
  return std::nullopt;
}

/// Formats scope layers as `"[...]"`. `/` joins labels within one nesting
/// layer (outermost first); `:` joins distinct layers that reached the same
/// location through compiler merges.
/// `layers` is expected to already be in the canonical order
/// `collectProfileScopeLayers` returns, so a given peer set always prints the
/// same way.
/// Returns an empty string when `layers` is empty, so callers can
/// unconditionally prepend the result to a name without a branch.
/// For use as a prefix, there is space after the string, and as a suffix,
/// there is a space before the string.
inline std::string formatProfileScope(llvm::ArrayRef<ProfileScopeLayer> layers,
                                      bool asPrefix) {
  if (layers.empty())
    return "";
  std::string result;
  llvm::raw_string_ostream os(result);
  if (!asPrefix)
    os << ' ';
  os << '[';
  detail::joinProfileScopeLayers(layers, os);
  os << ']';
  if (asPrefix)
    os << ' ';
  return result;
}

} // namespace M::MOGG

#endif // GENERICML_GRAPHCOMPILER_MOGGDIALECT_SUPPORT_PROFILESCOPELABELS_H
