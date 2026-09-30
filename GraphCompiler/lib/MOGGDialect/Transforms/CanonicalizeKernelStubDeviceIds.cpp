//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GraphCompiler/MOGGDialect/MOGGPasses.h"
#include "Support/MDialect/MAttrs.h"

#include "mlir/IR/AttrTypeSubElements.h"

using namespace M;

namespace M::MOGG {

#define GEN_PASS_DEF_CANONICALIZEKERNELSTUBDEVICEIDS
#include "GraphCompiler/MOGGDialect/MOGGPasses.h.inc"

namespace {

// The canonical device id every id inside a kernel stub is rewritten to. It
// must be 0, not an out-of-range sentinel: some ops carry an
// `InferTypeOpInterface` whose inferred result device is `DeviceRefAttr::
// getDefault` (id 0, e.g. `MGP::BufferConstantOp`), and this runs as a real
// (re-verified) pass, so a non-zero id on such an op would fail type
// verification. Id 0 is the default those ops already infer, and the emitted
// Mojo reads only the device label, never the id.
constexpr uint64_t kCanonicalDeviceId = 0;

struct CanonicalizeKernelStubDeviceIds
    : public impl::CanonicalizeKernelStubDeviceIdsBase<
          CanonicalizeKernelStubDeviceIds> {
  using CanonicalizeKernelStubDeviceIdsBase::
      CanonicalizeKernelStubDeviceIdsBase;

  void runOnOperation() override {
    mlir::ModuleOp moduleOp = getOperation();
    mlir::MLIRContext *ctx = moduleOp.getContext();

    // The device id lives on the `#M.device_ref` carried by the `!mgp.tensor`
    // types the stub operates on (and by a few op attributes). Walk both attrs
    // and types so every id inside a stub collapses to the same sentinel.
    mlir::AttrTypeReplacer replacer;
    replacer.addReplacement([ctx](DeviceRefAttr deviceRef) -> DeviceRefAttr {
      if (deviceRef.getId() == kCanonicalDeviceId)
        return deviceRef;
      return DeviceRefAttr::get(ctx, deviceRef.getLabel(), kCanonicalDeviceId);
    });

    for (KernelStubOp kernelStub :
         llvm::make_early_inc_range(moduleOp.getOps<KernelStubOp>()))
      replacer.recursivelyReplaceElementsIn(kernelStub, /*replaceAttrs=*/true,
                                            /*replaceLocs=*/false,
                                            /*replaceTypes=*/true);
  }
};

} // namespace

} // namespace M::MOGG
