//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/Conversion/MOToPrimitives.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GenericML/MGPDialect/MGPOps.h"

#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/IR/Threading.h"

using namespace M;

namespace M::MGP {

#define GEN_PASS_DEF_RAISEBACKINDEXCONSTANT
#include "GenericML/GraphCompiler/Conversion/MOToPrimitives.h.inc"

} // namespace M::MGP

namespace M::MGP {

namespace {

struct RaiseBackIndexConstant
    : public MGP::impl::RaiseBackIndexConstantBase<RaiseBackIndexConstant> {
  using RaiseBackIndexConstantBase::RaiseBackIndexConstantBase;
  /// Stubs that have only one caller were not deduplicated. Therefore, all the
  /// index.constant that were turned into mgp.buffer.constant +
  /// mgp.buffer.to_index should be raised back into index.constant to reduce
  /// the quantity of text that is emitted. Note that it is not required to
  /// update the constant resource attributes because they will just no longer
  /// be used.
  void runOnOperation() override {
    auto moduleOp = getOperation();
    // Build a map of callee -> callers. Not using SymbolTable because it is
    // very slow to call it on a ModuleOp. Instead, we iterate through the model
    // ops in order to skip the thousands of KernelStubOp.
    llvm::DenseMap<StringAttr, SmallVector<MOGG::ExecuteCompiledOp>> symbolUses;
    for (auto &&modelOp : moduleOp.getOps<MGP::ModelOp>()) {
      modelOp.walk([&](MOGG::ExecuteCompiledOp callOp) {
        symbolUses[callOp.getCalleeSymbolAttr().getAttr()].push_back(callOp);
      });
    }

    auto stubsRange = moduleOp.getOps<MOGG::KernelStubOp>();
    SmallVector<MOGG::KernelStubOp> kernels(stubsRange.begin(),
                                            stubsRange.end());
    mlir::parallelForEach(
        moduleOp->getContext(), kernels, [&](MOGG::KernelStubOp kernelStub) {
          if (kernelStub.getAbi() != MOGG::ABIKind::MGP)
            return;

          auto it = symbolUses.find(kernelStub.getSymNameAttr());
          if (it == symbolUses.end())
            return;

          // If there isn't exactly one caller, ignore because we don't want to
          // inline a constant that could possibly be different for different
          // callers (would have to look into their constant resources which is
          // not worth it).
          if (it->second.size() != 1)
            return;

          auto resourceArr = it->second.front().getConstantResourcesAttr();

          OpBuilder builder(kernelStub);
          kernelStub.walk([&](MGP::BufferToIndexOp bufferToIndexOp) {
            auto resourceConstantOp =
                bufferToIndexOp.getInBuffer()
                    .getDefiningOp<MGP::ResourceConstantOp>();
            if (!resourceConstantOp)
              return;

            uint64_t slotIndex =
                resourceConstantOp.getResourceSlot().getZExtValue();
            auto resource = dyn_cast<AlignedBytesAttr>(resourceArr[slotIndex]);
            // The assumption is that index.constant was lowered to
            // AlignedBytesAttr. If that changes, this part of the code has to
            // change to.
            if (!resource)
              return;

            if (resource.size() != sizeof(int64_t))
              return;

            int64_t value =
                *reinterpret_cast<const int64_t *>(resource.getData().data());
            builder.setInsertionPoint(bufferToIndexOp);
            auto indexConstantOp = mlir::index::ConstantOp::create(
                builder, resourceConstantOp->getLoc(), value);
            bufferToIndexOp->replaceAllUsesWith(indexConstantOp);
          });
        });
  }
};

} // namespace
} // namespace M::MGP
