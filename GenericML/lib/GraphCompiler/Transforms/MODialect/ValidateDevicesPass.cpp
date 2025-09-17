//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/MODialect/MOOps.h"
#include "GenericML/GraphCompiler/Transforms/MODialect/MOTransforms.h"
#include "mlir/IR/Visitors.h"
#include "llvm/ADT/DenseMap.h"

#include "Utils.h"

#define DEBUG_TYPE "validate-devices"

namespace M::MO {

#define GEN_PASS_DEF_VALIDATEDEVICESPASS
#include "GenericML/GraphCompiler/Transforms/MODialect/MOTransforms.h.inc"

}; // namespace M::MO

using namespace M;
using namespace MO;

namespace {

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

  bool isSupportedArchDTypeCombination(llvm::Triple::ArchType arch,
                                       DType dtype) {
    // Return True if the architecture does not support the data type.
    // On CPUs, we currently do not support the float8* type.
    // On ARM systems, we do not support bfloat16 or float16.
    // Note that CPU also does not truly support bf16 or f16, but are permitted
    // for casting purposes.
    if (arch == llvm::Triple::ArchType::x86_64 &&
        (dtype == DType::bf16 || dtype == DType::f16))
      return true;
    if (llvm::is_contained(
            {llvm::Triple::ArchType::x86_64, llvm::Triple::ArchType::aarch64},
            arch) &&
        dtype.isFloat() && dtype.getWidthInBits() < 32) {
      return false;
    }

    return true;
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

  LogicalResult process(llvm::Triple::ArchType targetArch, Block *block) {
    // Check the inputs first
    for (Value arg : block->getArguments()) {
      if (isa<MO::TensorType>(arg.getType())) {
        TensorType tensorType = dynCastFromTensorLike(arg.getType());
        LLVM_DEBUG(llvm::dbgs() << "Checking input '" << arg << "'\n");
        if (!isSupportedArchDTypeCombination(targetArch,
                                             tensorType.getDtype())) {
          emitErrorOnOp(block->getParentOp(), tensorType.getDeviceRef(),
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
          if (!isSupportedArchDTypeCombination(targetArch,
                                               tensorType.getDtype())) {
            emitErrorOnOp(op, tensorType.getDeviceRef(),
                          tensorType.getDtype().getAsString());
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

    LLVM_DEBUG({
      llvm::dbgs() << "defaultDeviceTargetTriple: '"
                   << defaultDeviceTargetTriple << "'\n";
      llvm::dbgs() << "defaultDeviceTargetArch: '" << defaultDeviceTargetArch
                   << "'\n";
    });

    if (!defaultDeviceTargetTriple.empty() &&
        !defaultDeviceTargetArch.empty()) {
      TargetInfo targetInfo{llvm::Triple(defaultDeviceTargetTriple),
                            defaultDeviceTargetArch};
      graphOp.setDeviceTargetInfo(
          fromRuntimeTargetInfo(graphOp.getContext(), targetInfo));
    }

    TargetInfoAttr targetInfo = graphOp.getDeviceTargetInfo();
    if (!targetInfo) {
      LLVM_DEBUG({
        llvm::dbgs() << "The graph op has no target information, "
                        "ValidateDevicesPass will do nothing.\n";
      });
      return;
    }

    LLVM_DEBUG({
      llvm::dbgs() << "Running ValidateDevicesPass for target '"
                   << targetInfo.getTriple().getTriple() << "'\n";
    });

    llvm::Triple::ArchType targetArch = targetInfo.getTriple().getArch();
    if (failed(process(targetArch, graphOp.getBodyBlock()))) {
      return signalPassFailure();
    }
  }

  GraphOp graphOp;
};

} // namespace
