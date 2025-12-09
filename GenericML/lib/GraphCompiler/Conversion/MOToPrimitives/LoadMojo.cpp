//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/Conversion/MOToPrimitives.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGOps.h"

using namespace M;

namespace M::MGP {

#define GEN_PASS_DEF_LOADMOJO
#include "GenericML/GraphCompiler/Conversion/MOToPrimitives.h.inc"

} // namespace M::MGP

namespace M::MGP {

struct LoadMojo
    : public MOGG::MojoLibraryAnalysisPass<impl::LoadMojoBase<LoadMojo>> {
  using Base::Base;

  void runOnOperation() override {
    auto moduleOp = getOperation();
    static constexpr llvm::StringLiteral mojoResourceName = "mojo";
    auto resourceManager = DenseResourceElementsHandle::getManagerInterface(
        moduleOp->getContext());
    auto mojoTextResource =
        resourceManager.getBlobManager().lookup(mojoResourceName);
    if (!mojoTextResource) {
      moduleOp->emitError(
          "Can't load Mojo before emitting Mojo or after JIT compilation");
      return;
    }

    auto memBufferOr = toModularErrorOr(
        llvm::MemoryBuffer::getFile(this->path, /*IsText=*/true,
                                    /*RequiresNullTerminator=*/false));
    if (memBufferOr.isError()) {
      moduleOp->emitError(llvm::formatv("Could not load the Mojo file {} : {}",
                                        this->path, memBufferOr.getError()));
      signalPassFailure();
      return;
    }

    auto buffer = memBufferOr->get()->getBuffer();
    auto blob = mlir::HeapAsmResourceBlob::allocateAndCopyInferAlign<char>(
        {buffer.data(), buffer.size()});
    auto &blobManager = resourceManager.getBlobManager();
    blobManager.update(mojoResourceName, std::move(blob));
  }
};

} // namespace M::MGP
