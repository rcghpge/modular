//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Fuses connected GenericCompiledRegion.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/Conversion/MOToPrimitives.h"
#include "GenericML/GraphCompiler/Interfaces/LowerableToMGPInterface.h"
#include "GenericML/GraphCompiler/MOGGDialect/Analysis/SequenceFusionAnalysis.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGDialect.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GraphRT/GRTDialect/GRTOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/TopologicalSortUtils.h"

using namespace M;

namespace M::MGP {

#define GEN_PASS_DEF_FUSEGENERICCOMPILEDREGIONS
#include "GenericML/GraphCompiler/Conversion/MOToPrimitives.h.inc"

} // namespace M::MGP

namespace M::MGP {

namespace {

// Fuses the provided operations. The resulting fused operation has the same
// output type as the types of valuesToReplace. This differs with standard
// sequence fusion in that we do not track outside uses (ie output values are
// values used elsewhere). The call site determines what values should be
// replaced via valuesToReplace. compiledRegionOpsToFuseTogether must be
// topologically sorted.
static void fuseCompiledRegionsAndReplaceUses(
    SmallVector<Operation *> &compiledRegionOpsToFuseTogether,
    ValueRange valuesToReplace) {
  // Gather all outside results and operands of all compiledRegions to be fused.
  llvm::SetVector<Value> allResults;
  llvm::SetVector<Value> allOperands;
  for (Operation *op : compiledRegionOpsToFuseTogether) {
    auto results = op->getResults();
    auto operands = op->getOperands();
    allResults.insert(results.begin(), results.end());
    allOperands.insert(operands.begin(), operands.end());
  }

  // FusedOperations = AllOperands - AllResults
  // This is because operands that belong to AllResults will be accessible from
  // within the compiledRegion. Otherwise, a cycle would be created.
  llvm::SetVector<Value> fusedCompiledRegionOperands = allOperands;
  fusedCompiledRegionOperands.set_subtract(allResults);

  // Create the output types which is just a 1:1 translation from Value -> Type.
  SmallVector<Type> fusedCompiledRegionResultTypes;
  for (Value result : valuesToReplace)
    fusedCompiledRegionResultTypes.push_back(result.getType());

  // Create the compiledRegion skeleton which is possible now that we have input
  // operands and output types.
  auto ctx = compiledRegionOpsToFuseTogether.front()->getContext();
  OpBuilder builder(ctx);
  Location loc = compiledRegionOpsToFuseTogether.back()->getLoc();
  builder.setInsertionPointAfter(compiledRegionOpsToFuseTogether.back());
  SmallVector<InOutSignatureAttr::InOutSemantics> operandSemantics;
  auto fusedCompiledRegionOp = builder.create<MOGG::GenericCompiledRegionOp>(
      loc, valuesToReplace.getType(), fusedCompiledRegionOperands.getArrayRef(),
      KGEN::ParamDeclArrayAttr::get(ctx, {}),
      InOutSignatureAttr::get(ctx, operandSemantics), BoolAttr{}, StringAttr{},
      BoolAttr{});
  mlir::Block &newBody = fusedCompiledRegionOp.getBody().emplaceBlock();
  builder.setInsertionPoint(&newBody, newBody.begin());

  // Create the compiledRegion arguments and map outside values to inside
  // arguments.
  mlir::DenseMap<Value, Value> outsideValueToInternalValue;
  for (Value operand : fusedCompiledRegionOperands) {
    auto newArg = newBody.addArgument(operand.getType(), loc);
    outsideValueToInternalValue[operand] = newArg;
  }

  mlir::IRMapping remapper;
  SmallVector<Value> valuesToOutput(valuesToReplace.size(), nullptr);
  for (auto opToFuse : compiledRegionOpsToFuseTogether) {
    auto compiledRegionToFuse = cast<MOGG::GenericCompiledRegionOp>(opToFuse);
    for (auto [operand, argument] :
         llvm::zip(opToFuse->getOperands(),
                   compiledRegionToFuse.getBody().front().getArguments())) {
      remapper.map(argument, outsideValueToInternalValue[operand]);
    }

    for (auto &internalOp : compiledRegionToFuse.getBody().front()) {
      // Actual fusion.
      auto newOp = builder.clone(internalOp, remapper);
      // If the cloned op is an output op, insert the old outside value into the
      // outsideValueToInternalValue map. This works because ops which use that
      // value will be fused after the current op (topologically sorted input).
      if (auto newOutputOp = dyn_cast<MOGG::OutputOp>(newOp)) {
        for (auto [result, internalValue] :
             llvm::zip(compiledRegionToFuse->getResults(),
                       newOutputOp.getOperands())) {
          outsideValueToInternalValue[result] = internalValue;

          for (auto [i, valueToReplace] : llvm::enumerate(valuesToReplace)) {
            if (valueToReplace == result) {
              GENERICML_ASSERT(!valuesToOutput[i]);
              valuesToOutput[i] = internalValue;
              break;
            }
          }
        }
        newOutputOp->erase();
      }
    }
  }

  builder.create<MOGG::OutputOp>(loc, valuesToOutput);

  // Use-def replacement.
  for (auto [oldValue, newValue] :
       llvm::zip(valuesToReplace, fusedCompiledRegionOp.getResults())) {
    oldValue.replaceAllUsesWith(newValue);
  }
}

static inline bool isEligibleForFusion(Operation *op) {
  return llvm::isa<MOGG::GenericCompiledRegionOp>(op);
}

static inline bool areCompatibleForFusion(Operation *lhs, Operation *rhs) {
  GENERICML_ASSERT(isEligibleForFusion(lhs) && isEligibleForFusion(rhs));
  return true;
}

// This pass will fuse *any* connected generic compiled region operation in the
// graph. It can afford to be greedy because it assumes that the content of
// generic compiled region operations are computationally trivial. If this
// changes, the pass must change or else it could be duplicating non trivial
// compute, in which case, a better, hollistic approach should be used.
// Example where being greedy could inefficient but it is fine for now:
// A--|
// |  |
// x  |
// |  |
// B <-
// Here, A and B wll be fused despite A not being erased after the pass. This is
// fine because we assume that A is computationally trivial.
struct FuseGenericCompiledRegions
    : public MGP::impl::FuseGenericCompiledRegionsBase<
          FuseGenericCompiledRegions> {
  using FuseGenericCompiledRegionsBase::FuseGenericCompiledRegionsBase;

  void runOnOperation() override {
    for (auto model : getOperation().getOps<MGP::ModelOp>()) {
      SmallVector<Operation *> compiledRegions;
      // Gather all compiled regions in the model.
      model.walk([&](MOGG::GenericCompiledRegionOp compiledRegion) {
        compiledRegions.push_back(compiledRegion);
      });
      if (compiledRegions.empty())
        return;

      // Sort them topologically.
      MO::ParametricTopologyAnalysis parametricTopologyAnalysis(model);
      parametricTopologyAnalysis.computeTopologicalSorting(compiledRegions);

      // Iterate in post order.
      for (auto compiledRegion : llvm::reverse(compiledRegions)) {
        // Operations which have been fused and have no other users can safely
        // be erased.
        if (llvm::all_of(compiledRegion->getUsers(),
                         [](auto user) { return isEligibleForFusion(user); })) {
          compiledRegion->erase();
          continue;
        }

        // Do a floodfill going up to gather all other regions that feed this op
        // (transitively). Takes care of blocks. Allowing duplication because of
        // the aforementioned assumptions on trivial compute.
        auto opsToFuse = MOGG::floodFillFrom<MOGG::Direction::Up, true>(
            {compiledRegion}, isEligibleForFusion, areCompatibleForFusion,
            parametricTopologyAnalysis);
        if (opsToFuse.size() == 1)
          continue;

        SmallVector<Operation *> sortedOpsToFuse(opsToFuse.begin(),
                                                 opsToFuse.end());
        parametricTopologyAnalysis.computeTopologicalSorting(sortedOpsToFuse);
        fuseCompiledRegionsAndReplaceUses(sortedOpsToFuse,
                                          compiledRegion->getResults());
        compiledRegion->erase();
      }
    }
  }
};

} // namespace

} // namespace M::MGP
