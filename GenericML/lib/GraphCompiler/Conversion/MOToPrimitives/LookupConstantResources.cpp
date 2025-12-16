//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/Conversion/MOToPrimitives.h"
#include "GenericML/GraphCompiler/MOGGDialect/Support/FusionUtils.h"

using namespace M;

namespace M::MGP {

#define GEN_PASS_DEF_LOOKUPCONSTANTRESOURCES
#include "GenericML/GraphCompiler/Conversion/MOToPrimitives.h.inc"

} // namespace M::MGP

namespace M::MGP {

namespace {

struct LookupConstantResources
    : public MGP::impl::LookupConstantResourcesBase<LookupConstantResources> {
  using LookupConstantResourcesBase::LookupConstantResourcesBase;

  void runOnOperation() override {
    if (!MOGG::useExperimentalMojoPath())
      return;

    auto kernelStubOp = getOperation();
    auto constantResources = kernelStubOp.getConstantResources();
    auto ctx = kernelStubOp->getContext();
    bool wasInterrupted =
        kernelStubOp
            ->walk([&](BufferConstantOp bufferConstantOp) {
              auto elementsAttr = bufferConstantOp.getAValue();
              auto it = llvm::find(constantResources, elementsAttr);
              if (it == constantResources.end()) {
                bufferConstantOp->emitError(
                    "The constant resource can't be found in the kernel");
                signalPassFailure();
                return mlir::WalkResult::interrupt();
              }
              uint64_t resourceIndex =
                  std::distance(constantResources.begin(), it);

              OpBuilder builder(bufferConstantOp);
              auto newConstantResource = ResourceConstantOp::create(
                  builder, bufferConstantOp->getLoc(),
                  bufferConstantOp.getType(),
                  IntegerAttr::get(IndexType::get(ctx), resourceIndex));
              bufferConstantOp->replaceAllUsesWith(newConstantResource);
              bufferConstantOp->erase();

              return mlir::WalkResult::advance();
            })
            .wasInterrupted();

    if (wasInterrupted)
      return;

    // Reset the constant resource attribute because it is no longer needed.
    // Note that we need to maintain an array of the right size because this is
    // what KernelStubOp::emitMojo uses to map the resource array elements to
    // MojoValue (it only cares about the number of resources, not what they are
    // so a UnitAttr is fine here).
    SmallVector<Attribute> dummyResources(constantResources.size(),
                                          UnitAttr::get(ctx));
    kernelStubOp.setConstantResourcesAttr(
        OpBuilder(kernelStubOp).getArrayAttr(dummyResources));
  }
};

} // namespace
} // namespace M::MGP
