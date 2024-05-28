//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/MGPTransforms/MGPTransforms.h"

#include "GenericML/MGPDialect/MGPOps.h"
#include "Support/AssertStream.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

#define DEBUG_TYPE "mgp-prepare-device-primitives"

namespace M::MGP {

#define GEN_PASS_DEF_PREPAREDEVICEPRIMITIVESPASS
#include "GenericML/MGPTransforms/MGPTransforms.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// DevicePrimitivesPass
//===----------------------------------------------------------------------===//

class PrepareDevicePrimitivesPass
    : public impl::PrepareDevicePrimitivesPassBase<
          PrepareDevicePrimitivesPass> {
  void runOnOperation() override {
    ModelOp model = getOperation();
    OpBuilder builder(model.getContext());
    model.walk([&](Operation *origOp) {
      if (auto hasDeviceOp = dyn_cast<MGPHasDeviceOpInterface>(origOp)) {
        GENERICML_ASSERT(!origOp->hasSuccessors());
        GENERICML_ASSERT(origOp->getNumRegions() == 0);
        builder.setInsertionPoint(origOp);
        std::string newOpName = (Twine(origOp->getName().getStringRef()) + "." +
                                 hasDeviceOp.getDevice().getLabel())
                                    .str();
        LLVM_DEBUG(llvm::dbgs() << "renaming " << origOp->getName() << " to "
                                << newOpName << "\n");

        NamedAttrList newAttrs;
        for (auto namedAttr : origOp->getAttrDictionary()) {
          if (auto deviceRefAttr =
                  dyn_cast<DeviceRefAttr>(namedAttr.getValue())) {
            std::string newAttrName =
                (Twine(namedAttr.getName().getValue()) + "Id").str();
            LLVM_DEBUG(llvm::dbgs() << "renaming " << namedAttr.getName()
                                    << " to " << newAttrName << "\n");
            Type ui64 = builder.getType<IntegerType>(64, IntegerType::Unsigned);
            APInt apint(ui64.getIntOrFloatBitWidth(), deviceRefAttr.getId(),
                        ui64.isSignedInteger());
            newAttrs.set(newAttrName, IntegerAttr::get(ui64, apint));
          } else {
            newAttrs.set(namedAttr.getName(), namedAttr.getValue());
          }
        }

        Operation *newOp = Operation::create(
            /*location=*/origOp->getLoc(),
            /*name=*/OperationName(newOpName, origOp->getContext()),
            /*resultTypes=*/origOp->getResults().getTypes(),
            /*operands=*/origOp->getOperands(),
            /*attributes=*/std::move(newAttrs),
            /*properties=*/origOp->getPropertiesStorage(),
            /*successors=*/{},
            /*numRegions=*/0);
        builder.setInsertionPoint(origOp);
        builder.insert(newOp);
        origOp->replaceAllUsesWith(newOp->getResults());
        origOp->erase();
      }
    });
  }
};

} // namespace
} // namespace M::MGP
