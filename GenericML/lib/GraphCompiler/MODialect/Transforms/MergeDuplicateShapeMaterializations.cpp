//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/MODialect/ShapeFunctions.h"
#include "GenericML/GraphCompiler/MODialect/Transforms/MOTransforms.h"
#include "GenericML/GraphCompiler/MOGGDialect/Support/KGENUtils.h"
#include "KGEN/KGENDialect/KGENParameters.h"
#include "llvm/ADT/STLExtras.h"

#include <unordered_map>

using namespace M;
using namespace MO;

#define DEBUG_TYPE "merge-duplicate-shape-materializations"

namespace std {
template <>
struct hash<::mlir::ValueRange> {
  std::size_t operator()(const ::mlir::ValueRange &s) const noexcept {
    return ::llvm::hash_combine_range(s.begin(), s.end());
  }
};

template <>
struct hash<std::pair<::llvm::StringRef, ::mlir::ValueRange>> {
  std::size_t operator()(const std::pair<::llvm::StringRef, ::mlir::ValueRange>
                             &s) const noexcept {
    return ::llvm::hash_combine(::llvm::hash_value(s.first),
                                std::hash<::mlir::ValueRange>{}(s.second));
  }
};
} // namespace std
namespace M::MO {

#define GEN_PASS_DEF_MERGEDUPLICATESHAPEMATERIALIZATIONS

#include "GenericML/GraphCompiler/MODialect/Transforms/MOTransforms.h.inc"

} // namespace M::MO

namespace {
/// A map that takes the operation name and its operands as a key. The value is
/// the primary operation and all of its defined parameters. The key is used to
/// identify operations that need to be merged with the primary operation, and
/// the parameters of the erased op are remapped to the parameters of the
/// primary operation.
using OpCacheTy = std::unordered_map<
    std::pair<llvm::StringRef, ValueRange>,
    std::pair<Operation *, SmallVector<KGEN::ParamDeclRefAttr>>>;

bool isOpDefiningOpForAllParams(Operation *op,
                                ArrayRef<KGEN::ParamDeclRefAttr> references,
                                const KGEN::ParameterUseDefGraph &useDefGraph) {

  SmallVector<KGEN::ParamDefinition> definitions;
  for (const auto &reference : references) {
    // We return failse if we can't find the definition or it doesn't match our
    // op
    if (auto it = useDefGraph.defs.find(reference.getName());
        it != useDefGraph.defs.end() && it->second.defOp == op)
      definitions.emplace_back(it->second);
    else
      return false;
  }
  return llvm::all_of(definitions, [&](const KGEN::ParamDefinition &def) {
    return isOpDefiningOpForAllParams(op, def.uses, useDefGraph);
  });
}
} // namespace

/// This pass identifies all shape materialization ops in a Graph op that
/// defines all its parameters. For all such operations, if there are multiple
/// operations found with the same operation name and operands, it erases all
/// but the first instance of the operation. It then remaps the parameters in
/// the erased ops to the primary instance's parameters. This is all to aid CSE,
/// since CSE can't automatically identify cases of otherwise equivalent
/// operations when they have different parameter references.
struct MergeDuplicateShapeMaterializationsPass
    : public MO::impl::MergeDuplicateShapeMaterializationsBase<
          MergeDuplicateShapeMaterializationsPass> {
  using Base::Base;

  void runOnOperation() override {
    MO::GraphOp graph = getOperation();
    auto context = &getContext();
    IRRewriter rewriter(context);

    OpCacheTy opCache;

    auto paramCache = KGEN::ParameterCollector::Analysis(graph);
    KGEN::ParameterUseDefGraph useDefGraph(graph.getRegion());
    useDefGraph.calculate(paramCache);

    llvm::DenseMap<StringAttr, TypedAttr> paramMappings;
    llvm::SmallVector<Operation *> toErase;
    for (const auto &opUse : useDefGraph.opUses) {
      Operation *op = opUse.first;
      const SmallVector<KGEN::ParamDeclRefAttr> &references = opUse.second;

      if (!isa<ShapeMaterialization>(op))
        continue;

      auto definesAllParams =
          isOpDefiningOpForAllParams(op, references, useDefGraph);

      if (!definesAllParams)
        continue;

      auto [it, inserted] = opCache.try_emplace(
          {op->getName().getStringRef(), ValueRange(op->getOperands())}, opUse);

      if (inserted)
        continue;

      for (auto [thisOpReference, primaryOpReference] :
           llvm::zip(references, it->second.second)) {
        paramMappings[thisOpReference.getName()] = primaryOpReference;
      }

      op->replaceAllUsesWith(it->second.first);
      toErase.emplace_back(op);
    }

    MOGG::remapTypesAndAttrs(graph, paramMappings);

    for (auto op : toErase)
      op->erase();
  }
};
