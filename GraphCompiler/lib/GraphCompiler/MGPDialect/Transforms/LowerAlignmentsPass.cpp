//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/GraphCompiler/MGPDialect/Transforms/MGPTransforms.h"
#include "MLRT/Driver/DeviceContext/DeviceContext.h"

#include "GraphCompiler/GraphCompiler/MGPDialect/Dialect/MGPOps.h"
#include "GraphCompiler/GraphCompiler/MGPDialect/Transforms/AsyncDataflowDependencies.h"
#include "GraphCompiler/GraphCompiler/MGPDialect/Transforms/Utils.h"
#include "GraphCompiler/GraphCompiler/MOGGDialect/MOGGOps.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "MLRT/GraphRT/GRTDialect/GRTChainInterface.h"
#include "MLRT/GraphRT/GRTDialect/GRTOps.h"
#include "MLRT/GraphRT/MEFDialect/MEFTypes.h"
#include "Support/AlignedAlloc.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"
#include <numeric>

using namespace mlir;

#define DEBUG_TYPE "mgp-lower-alignments-pass"

namespace M::MGP {

// Generate support code for LowerAlignmentsPass
#define GEN_PASS_DEF_LOWERALIGNMENTSPASS
#include "GraphCompiler/GraphCompiler/MGPDialect/Transforms/MGPTransforms.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// LowerAlignmentsPass
//===----------------------------------------------------------------------===//

class LowerAlignmentsPass
    : public impl::LowerAlignmentsPassBase<LowerAlignmentsPass> {
  using LowerAlignmentsPassBase::LowerAlignmentsPassBase;

  const int64_t kByteAlignment =
      1; // Means we can't infer anything about alignment

  int64_t getDefaultAlignment(DeviceRefAttr deviceRef) {
    return M::Driver::getPreferredAlignment(deviceRef.getLabel().str());
  }
  // Create a new alignment mapping valid for the interior of a block given
  // the alignmentMap of the scope of the parent Op.
  // Args:
  //      block - block to create alignmentMap for
  //      parentScopeAlignment - parent scope alignment map. Must contain the
  //      block
  llvm::DenseMap<Value, int64_t> createBlockArgAlignments(
      Block *block,
      const llvm::DenseMap<Value, int64_t> &parentScopeAlignments) {
    llvm::DenseMap<Value, int64_t> innerAlignments;
    for (auto arg : block->getArguments()) {
      // Populate default alignments for device types
      if (auto deviceType = dyn_cast<MGPHasDeviceTypeInterface>(arg.getType()))
        innerAlignments[arg] = getDefaultAlignment(deviceType.getDevice());
      else
        innerAlignments[arg] = M::Driver::Device::preferredAlignment("cpu");
      // If an outside operand exists and has scopedAlignments in the parent
      // scope we copy that for internal args
      if (block->getParentOp()->getNumOperands() > arg.getArgNumber()) {
        auto operand = block->getParentOp()->getOperand(arg.getArgNumber());
        if (auto it = parentScopeAlignments.find(operand);
            it != parentScopeAlignments.end())
          innerAlignments[arg] = it->second;
      }
    }
    return innerAlignments;
  }

  // Construct an argument dependency mapping for all values within a block.
  // If value relies on the input arg directly, or implicitly, it is placed in
  // the set.
  //    block (...) (arg0, arg1, arg2)
  //       val0 = ... arg0        val0 -> {arg0}
  //       val1 = ... arg1, arg2  val1 -> {arg1, arg2}
  //       val2 = ... val0, val1  val2 -> {arg0, arg1, arg2}
  llvm::DenseMap<Value, DenseSet<Value>> getValueArgDependencies(Block *block) {
    llvm::DenseMap<Value, DenseSet<Value>> results;
    llvm::DenseSet<Value> args;
    for (auto arg : block->getArguments())
      args.insert(arg);
    // Has to go in dependency order
    for (auto op : getDataflowOrder(block))
      for (auto res : op->getResults()) {
        auto resIt = results.insert({res, {}}).first;
        for (auto operand : op->getOperands())
          if (auto it = results.find(operand); it != results.end())
            resIt->second.insert(it->second.begin(), it->second.end());
          else if (args.contains(operand))
            resIt->second.insert(operand);
      }
    return results;
  }

  // Populate the alignments for all the placeholder ops contained by operation
  void
  populateAlignmentsForOp(Operation *op,
                          llvm::DenseMap<Value, int64_t> &parentScopeAlignments,
                          OpBuilder &builder) {
    for (auto &region : op->getRegions()) {
      for (auto &block : region.getBlocks()) {
        // Generate argument alignments the valueArg dependency
        auto argAlignments =
            createBlockArgAlignments(&block, parentScopeAlignments);
        auto valueArgDep = getValueArgDependencies(&block);

        auto replacePlaceholder =
            [&](MOGG::AlignmentPlaceholderOp placeholderOp,
                int64_t newAlignment) {
              builder.setInsertionPoint(placeholderOp);
              KGEN::ParamDeclareOp::create(builder, placeholderOp.getLoc(),
                                           placeholderOp.getParamDecl(),
                                           builder.getIndexAttr(newAlignment));
            };

        // Traverse and replace all placeholders
        op->walk([&](MOGG::AlignmentPlaceholderOp placeholderOp) {
          auto operand = placeholderOp.getOperand();
          auto it = valueArgDep.find(operand);
          assert(it != valueArgDep.end() &&
                 "valueArgDep must contain the set of arg it depends on (can "
                 "be empty)");
          if (it->second.empty()) {
            // If there are no argument it depends on, assume it is cpu (comes
            // from kgen input placeholder)
            replacePlaceholder(placeholderOp,
                               M::Driver::Device::preferredAlignment("cpu"));
          } else {
            // Placeholder operand depend on arguments - infer from args. Always
            // try to get GCD of alignment from inputs (2, 16, 8 will have GCD
            // of 2)
            int64_t alignment = argAlignments.at(*it->second.begin());
            for (auto arg : it->second)
              alignment = std::gcd(argAlignments.at(arg), alignment);
            replacePlaceholder(placeholderOp, alignment);
          }
          placeholderOp.erase();
        });
      }
    }
  }

  // Try to lower block in dataflow order and populate the alignments with
  // additional information
  void inferAndPopulateAlignments(Block *block,
                                  llvm::DenseMap<Value, int64_t> &alignmentMap,
                                  OpBuilder &builder) {
    // lambda to set alignmentMap from raw alignment
    auto setAlignmentFromRawAlignment = [&](uint64_t rawAlignment, Value value,
                                            DeviceRefAttr deviceRef) {
      // Priority order is alignment enforced by op > device
      if (rawAlignment != kUnknownSize)
        alignmentMap[value] = static_cast<int64_t>(rawAlignment);
      else
        alignmentMap[value] = getDefaultAlignment(deviceRef);
    };

    for (auto op : getDataflowOrder(block)) {
      if (auto bufferOp = dyn_cast<MGP::BufferAllocOp>(op)) {
        // Set from MGP Op
        setAlignmentFromRawAlignment(bufferOp.getRawAlign(),
                                     bufferOp.getOutBuffer(),
                                     bufferOp.getDevice());
      } else if (auto bufferOp = dyn_cast<MGP::BufferConstantOp>(op)) {
        // Set from MGP Op
        setAlignmentFromRawAlignment(
            bufferOp.getRawAlign(), bufferOp.getOutBuffer(),
            bufferOp.getResult().getType().getDevice());
      } else if (auto bufferOp = dyn_cast<MGP::BufferConstantExternalOp>(op)) {
        // Set from MGP Op
        setAlignmentFromRawAlignment(bufferOp.getBAlign(),
                                     bufferOp.getOutBuffer(),
                                     bufferOp.getDevice());
      } else if (auto bufferOp = dyn_cast<MGPBufferAliasOpInterface>(op)) {
        // Pass-through
        alignmentMap[bufferOp.getOutBuffer()] =
            alignmentMap.at(bufferOp.getInBuffer());
      } else if (auto bufferOp = dyn_cast<MGP::BufferCacheOp>(op)) {
        // Pass-through to caching system
      } else if (auto bufferOp = dyn_cast<MGP::BufferGetCachedOp>(op)) {
        // Pass-through from caching system
        alignmentMap[bufferOp.getOutBuffer()] =
            alignmentMap[bufferOp.getBBufferSlot()];
      } else if (auto extractBufferOp =
                     dyn_cast<MGP::TensorExtractBufferOp>(op)) {
        // Pass-through from Tensor Value if it is something we created.
        // Otherwise, assume it comes from a fallback so default to cpu
        if (auto it = alignmentMap.find(extractBufferOp.getTensor());
            it != alignmentMap.end())
          alignmentMap[extractBufferOp.getOutBuffer()] = it->second;
        else
          alignmentMap[extractBufferOp.getOutBuffer()] =
              M::Driver::Device::preferredAlignment("cpu");
      } else if (auto createTensorOp = dyn_cast<MGP::TensorCreateOp>(op)) {
        // Pass-through to Tensor Value
        alignmentMap[createTensorOp.getOutTensor()] =
            alignmentMap.at(createTensorOp.getBuffer());
      } else if (auto guardOp = dyn_cast<MGP::GuardOp>(op)) {
        // Pass-through
        for (auto [input, output] :
             llvm::zip(guardOp.getInputs(), guardOp.getResults()))
          alignmentMap[output] = alignmentMap.at(input);
      } else if (auto bufferOp = dyn_cast<MGP::BufferSliceOp>(op)) {
        // offsetAlignment can be inferred from static index op input. Otherwise
        // byteAlignment
        int64_t offsetAlignment = kByteAlignment;
        if (auto constOp =
                bufferOp.getOffset().getDefiningOp<mlir::index::ConstantOp>())
          offsetAlignment = constOp.getValue().getSExtValue();
        alignmentMap[bufferOp.getOutBuffer()] =
            std::gcd(alignmentMap.at(bufferOp.getInBuffer()), offsetAlignment);
      } else if (isa<MOGG::GenericCompiledRegionOp>(op)) {
        // Populate all alignments for GenericCompiledRegionOp
        populateAlignmentsForOp(op, alignmentMap, builder);
      } else if (op->getNumRegions()) {
        // Recursively populate for interior blocks
        for (auto &innerRegion : op->getRegions()) {
          for (auto &innerBlock : innerRegion.getBlocks()) {
            auto argAlignments =
                createBlockArgAlignments(&innerBlock, alignmentMap);
            inferAndPopulateAlignments(&innerBlock, argAlignments, builder);
          }
        }
      }
    }
  }

  // This populates alignment placeholders by doing a dataflow dependencies
  // analysis. Typical alignment assumes a default alignment which comes from
  // device except in cases when MGP introduces ops that can invalidate
  // alignment. In those cases, try to populate and keep track and populate
  // alignment Several main types of behaviour:
  //
  // Conditional invalidation of alignment
  //    conditionally from mgp.buffer.{split.borrow,slice}
  // Pass-through + read
  //    from mgp.buffer.{resize,unsize}
  //    from a side-load mechanism using mgp.buffer.{cache/get_cached pair}
  //    from mgp.tensor.{extract.buffer/create}
  // Set alignment
  //    from mgp.buffer.{alloc/constant/constant.external}
  // Populate/Use alignment
  //    from `GenericCompiledRegionOp`
  void runOnOperation() override {
    ModelOp modelOp = getOperation();
    llvm::DenseMap<Value, int64_t> alignmentMap;
    auto ctx = modelOp->getContext();
    OpBuilder builder(ctx);
    for (auto innerRegion : modelOp.getRegions()) {
      for (auto &innerBlock : innerRegion->getBlocks()) {
        auto argAlignments =
            createBlockArgAlignments(&innerBlock, alignmentMap);
        inferAndPopulateAlignments(&innerBlock, argAlignments, builder);
      }
    }
    LLVM_DEBUG(llvm::dbgs() << "LowerAlignmentsPass: done\n");
  }
};

} // namespace

} // namespace M::MGP
