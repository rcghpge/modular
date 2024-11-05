//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/MODialect/MOOps.h"
#include "GenericML/GraphCompiler/Transforms/MODialect/MOTransforms.h"
#include "mlir/IR/Visitors.h"
#include "llvm/ADT/DenseMap.h"

#define DEBUG_TYPE "validate-devices"

namespace M::MO {

#define GEN_PASS_DEF_VALIDATEDEVICESPASS
#include "GenericML/GraphCompiler/Transforms/MODialect/MOTransforms.h.inc"

//===----------------------------------------------------------------------===//
// ValidateDevicesPass
//===----------------------------------------------------------------------===//
//
// Verify that each input and result tensor uses data types that are valid for
// the assigned device architecture. This pass will report an error for only the
// first invalid combination it finds.

struct ValidateDevicesPass
    : public impl::ValidateDevicesPassBase<ValidateDevicesPass> {
  using ValidateDevicesPassBase::ValidateDevicesPassBase;

  // Add any new combinations of architecture and data type that are not
  // supported to this map. The existing logic in this pass should correctly
  // check for them and report errors.
  const llvm::DenseMap<llvm::Triple::ArchType, llvm::SmallVector<DType>>
      unsupportedArchDTypeCombinations{
          {llvm::Triple::ArchType::aarch64, {DType::bf16}},
      };

  bool isUnsupportedArchDTypeCombination(llvm::Triple::ArchType arch,
                                         DType dtype) {
    // Does this architecture have any unsupported data types?
    auto unsupportedArchDTypeCombination =
        unsupportedArchDTypeCombinations.find(arch);
    if (unsupportedArchDTypeCombination ==
        unsupportedArchDTypeCombinations.end())
      return false;

    // We found some, now let's see if the data type we are checking is in that
    // list.
    auto unsupportedDTypes = unsupportedArchDTypeCombination->getSecond();
    if (llvm::is_contained(unsupportedDTypes, dtype))
      return true;

    return false;
  }

  bool archHasAnyUnsupportedDTypes(llvm::Triple::ArchType arch) {
    return unsupportedArchDTypeCombinations.find(arch) !=
           unsupportedArchDTypeCombinations.end();
  }

  static std::string formatDeviceForUser(DeviceRefAttr device) {
    DeviceRef deviceRef;
    deviceRef.id = device.getId();
    deviceRef.label = device.getLabel();
    return deviceRef.toString();
  }

  static void emitErrorOnOp(Operation *op, DeviceRefAttr device,
                            StringRef dtype) {
    op->emitError() << "The " << dtype
                    << " data type is not supported on device '"
                    << formatDeviceForUser(device) << "'.";
  }

  LogicalResult process(DeviceRefAttr device, llvm::Triple::ArchType targetArch,
                        Block *block) {
    // Check the inputs first
    for (Value arg : block->getArguments()) {
      if (isa<MO::TensorType>(arg.getType())) {
        TensorType tensorType = dynCastFromTensorLike(arg.getType());
        LLVM_DEBUG(llvm::dbgs() << "Checking input '" << arg << "'\n");
        if (isUnsupportedArchDTypeCombination(targetArch,
                                              tensorType.getDtype())) {
          emitErrorOnOp(block->getParentOp(), device,
                        tensorType.getDtype().getAsString());
          return failure();
        }
      }
    }

    // Check the outputs now
    WalkResult result = graphOp->walk([&](Operation *op) {
      for (Value res : op->getResults()) {
        if (isa<MO::TensorType>(res.getType())) {
          TensorType tensorType = dynCastFromTensorLike(res.getType());
          LLVM_DEBUG(llvm::dbgs() << "Checking result '" << res << "'\n");
          if (isUnsupportedArchDTypeCombination(targetArch,
                                                tensorType.getDtype())) {
            emitErrorOnOp(op, device, tensorType.getDtype().getAsString());
            return WalkResult::interrupt();
          }
        }
      }
      return WalkResult::advance();
    });

    if (result.wasInterrupted())
      return failure();

    return success();
  }

  //===--------------------------------------------------------------------===//
  // Entry point
  //===--------------------------------------------------------------------===//

  void runOnOperation() override {
    graphOp = getOperation();

    TargetInfoAttr targetInfo = graphOp.getDeviceTargetInfo();
    if (!targetInfo) {
      LLVM_DEBUG({
        llvm::dbgs() << "The graph op has no target information, "
                        "ValidateDevicesPass will do nothing.\n";
      });
      return;
    }

    llvm::Triple::ArchType targetArch = targetInfo.getTriple().getArch();
    if (!archHasAnyUnsupportedDTypes(targetArch)) {
      LLVM_DEBUG({
        llvm::dbgs() << "There are no unsupported dtypes for: '"
                     << targetInfo.getTriple().getTriple()
                     << "' ValidateDevicesPass has no work to do.\n";
      });
      return;
    }

    LLVM_DEBUG({
      llvm::dbgs() << "Running ValidateDevicesPass for target '"
                   << targetInfo.getTriple().getTriple() << "'\n";
    });

    if (failed(process(graphOp.getDeviceRef(), targetArch,
                       graphOp.getBodyBlock()))) {
      return signalPassFailure();
    }
  }

  GraphOp graphOp;
};

} // namespace M::MO
