//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/GraphCompiler/MODialect/MODevices.h"
#include "GraphCompiler/GraphCompiler/MODialect/MOOps.h"
#include "GraphCompiler/GraphCompiler/MODialect/Transforms/MOTransforms.h"
#include "GraphCompiler/GraphCompiler/MOGGDialect/Analysis/MojoLibraryAnalysis.h"
#include "GraphCompiler/GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GraphCompiler/GraphCompiler/MOGGDialect/Support/MOToMOGGLowering.h"
#include "Support/MDialect/MAttrs.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

namespace M::MO {

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#define GEN_PASS_DEF_NANCHECKPASS
#include "GraphCompiler/GraphCompiler/MODialect/Transforms/MOTransforms.h.inc"

} // namespace M::MO

using namespace M;
using namespace M::MO;

namespace {

struct NanCheckPass : public MOGG::MojoLibraryAnalysisPass<
                          impl::NanCheckPassBase<NanCheckPass>> {
  using Base::Base;

  /// Returns a human readable label corresponding to value.
  std::string valueToLabel(Value value) {
    std::string str;
    llvm::raw_string_ostream os(str);

    if (auto op = value.getDefiningOp())
      os << " " << op->getName() << " ";

    value.printAsOperand(os, mlir::OpPrintingFlags().printGenericOpForm());

    return str;
  }

  /// Returns true if the value is a floating-point tensor.
  bool isFloatTensor(Value value) {
    auto tensorType = dyn_cast<MO::TensorType>(value.getType());
    if (!tensorType)
      return false;
    return tensorType.getDtype().isFloat();
  }

  /// Lower an mo.custom op to an MOGG kernel in-place. Returns the kernel
  /// or signals pass failure.
  MOGG::KernelOp
  lowerCustomOpToMOGG(MO::CustomOp customOp, OpBuilder &builder,
                      MOGG::MojoLibraryAnalysis &mojoLibraryAnalysis) {
    OpBuilder insertPoint{customOp};
    MOGG::MOGGKernelBuilder kernelBuilder(customOp, insertPoint,
                                          /*forceSync=*/false,
                                          mojoLibraryAnalysis);
    auto kernelOrErr = kernelBuilder.lowerToKernel();
    if (kernelOrErr.isError()) {
      customOp->emitOpError("[NanCheckPass] ")
          << llvm::Twine(kernelOrErr.getError());
      return nullptr;
    }
    auto moggKernel = kernelOrErr.takeValue();
    customOp->replaceAllUsesWith(moggKernel->getResults());
    customOp->erase();
    return moggKernel;
  }

  void runOnOperation() override {
    GraphOp graphOp = getOperation();
    OpBuilder builder(graphOp->getContext());

    auto *ctx = builder.getContext();
    auto chainType = MO::ChainType::get(ctx);
    auto hostDevice = M::getHostDevice(ctx);

    // Single-element int32 tensor types for NaN/Inf counts.
    auto emptyMetadata = builder.getDictionaryAttr({});
    auto countTypeHost = MO::TensorType::get(
        ctx, ArrayRef<int64_t>{1}, M::DType::si32, hostDevice, emptyMetadata);
    assert(countTypeHost && "Failed to create count tensor type");

    // Collect all MOGG kernel ops (compiled regions).
    SmallVector<MOGG::KernelOp> kernelOps;
    graphOp->walk([&kernelOps](MOGG::KernelOp op) { kernelOps.push_back(op); });

    if (kernelOps.empty())
      return;

    auto analysisOr = getLibraryAnalysis();
    if (analysisOr.isError()) {
      emitError(
          graphOp.getLoc(),
          llvm::formatv("[NanCheckPass] Failed to load Mojo analysis: {0}",
                        analysisOr.getError()));
      return signalPassFailure();
    }
    MOGG::MojoLibraryAnalysis &mojoLibraryAnalysis = *analysisOr.takeValue();

    // We'll create the initial chain lazily inside the loop to avoid
    // issues with insertion point ordering.
    Value currentChain;

    // Insert NaN/Inf checks after each compiled region's float outputs.
    // Architecture:
    //   1. nan_check_count: read-only reduction → scalar counts (on device)
    //   2. mo.transfer: move counts to host (chain provides ordering)
    //   3. nan_check_raise: read host counts, raise if > 0 (chain for DCE)
    // The original tensor flows unchanged — no copy, no replaceAllUsesExcept.
    for (auto kernelOp : kernelOps) {
      builder.setInsertionPointAfter(kernelOp);
      for (Value res : kernelOp.getResults()) {
        if (!isFloatTensor(res))
          continue;

        auto tensorType = cast<MO::TensorType>(res.getType());
        auto device = getDeviceOrDefaultAny(kernelOp);

        // Single-element int32 on the tensor's device for count outputs.
        auto countTypeOnDevice = MO::TensorType::get(
            ctx, ArrayRef<int64_t>{1}, M::DType::si32, device, emptyMetadata);

        // Metadata for the error message (static, known at compile time).
        std::string label = valueToLabel(res);
        std::string typeStr;
        {
          llvm::raw_string_ostream os(typeStr);
          os << tensorType;
        }

        // 1. nan_check_count: input tensor → (nan_count, inf_count) on device
        auto countOp = MO::CustomOp::create(
            builder, kernelOp.getLoc(),
            /*resultTypes=*/TypeRange{countTypeOnDevice, countTypeOnDevice},
            /*operands=*/ValueRange{res},
            /*symbol=*/builder.getStringAttr("nan_check_count"),
            /*device=*/device,
            /*parameters=*/builder.getDictionaryAttr({}));

        Value nanCount = countOp.getResult(0);
        Value infCount = countOp.getResult(1);

        // 2. mo.transfer: move scalar counts to host
        if (!currentChain) {
          currentChain = MO::ChainCreateOp::create(builder, kernelOp.getLoc(),
                                                   chainType, ValueRange{})
                             .getResult();
          assert(currentChain && "currentChain is null after creation");
        }

        // Transfer counts to host if they're on a different device.
        // On CPU, counts are already on host — skip the transfer.
        Value nanCountHost = nanCount;
        Value infCountHost = infCount;
        bool needsTransfer = (device != hostDevice);
        if (needsTransfer) {
          auto nanTransfer = MO::TransferOp::create(
              builder, kernelOp.getLoc(), cast<TensorValue>(nanCount),
              hostDevice, cast<ChainValue>(currentChain),
              /*alwaysElideSameDeviceCopy=*/true);
          nanCountHost = nanTransfer.getResult();
          currentChain = nanTransfer.getOutChain();

          auto infTransfer = MO::TransferOp::create(
              builder, kernelOp.getLoc(), cast<TensorValue>(infCount),
              hostDevice, cast<ChainValue>(currentChain),
              /*alwaysElideSameDeviceCopy=*/true);
          infCountHost = infTransfer.getResult();
          currentChain = infTransfer.getOutChain();
        }

        // 3. nan_check_raise: read host counts, raise if > 0
        auto raiseOp = MO::CustomOp::create(
            builder, kernelOp.getLoc(),
            /*resultTypes=*/TypeRange{chainType},
            /*operands=*/ValueRange{nanCountHost, infCountHost, currentChain},
            /*symbol=*/builder.getStringAttr("nan_check_raise"),
            /*device=*/hostDevice,
            /*parameters=*/
            builder.getDictionaryAttr(
                {NamedAttribute(builder.getStringAttr("label"),
                                builder.getStringAttr(label)),
                 NamedAttribute(builder.getStringAttr("type_str"),
                                builder.getStringAttr(typeStr))}));
        currentChain = raiseOp.getResult(0);

        // Lower nan_check_count to MOGG kernel.
        if (!lowerCustomOpToMOGG(countOp, builder, mojoLibraryAnalysis))
          return signalPassFailure();

        // Lower nan_check_raise to MOGG kernel, then mark it as having
        // write side effects so DCE preserves it even with unused results.
        auto raiseKernel =
            lowerCustomOpToMOGG(raiseOp, builder, mojoLibraryAnalysis);
        if (!raiseKernel)
          return signalPassFailure();
        raiseKernel.setHasWriteEffect(true);

        // Update currentChain to point to the MOGG kernel's chain result
        // (the raiseOp was erased during lowering, invalidating the old ref).
        for (auto result : raiseKernel->getResults()) {
          if (isa<MO::ChainType>(result.getType())) {
            currentChain = result;
            break;
          }
        }
      }
    }
    // Note: currentChain (the final raise's chain output) is unused, but the
    // raise kernel survives DCE because we set hasWriteEffect=true above.
  }
};

} // namespace

std::unique_ptr<Pass> M::MO::createNanCheckPass(
    std::shared_ptr<MOGG::MojoLibraryAnalysis> libraryAnalysis) {
  return std::make_unique<NanCheckPass>(std::move(libraryAnalysis));
}
