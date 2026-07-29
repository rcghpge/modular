//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Outlines each `mogg._kernel` body into a module-level `mogg.kernel.stub`
// (labeled with the `MAP` ABI) and replaces the kernel body with a single
// `mogg.call.stub` forwarding the kernel's `!mogg.pointer_` / `!mogg.dim_` /
// `!mogg.context` block arguments. The `mogg._kernel` container survives: it
// keeps the `!mo.tensor` interface that `MOGGToMGP` later lowers.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/Conversion/MOToMAP/MOToMAP.h"
#include "GraphCompiler/MODialect/MODevices.h"
#include "GraphCompiler/MODialect/MODialect.h"
#include "GraphCompiler/MOGGDialect/MOGGDialect.h"
#include "GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GraphCompiler/MOGGDialect/MOGGPasses.h"
#include "GraphCompiler/MOGGDialect/Support/FusionUtils.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/Twine.h"

using namespace mlir;
using namespace M;

namespace M::MOGG {

#define GEN_PASS_DEF_OUTLINEKERNELSINTOFUNCTIONS
#include "GraphCompiler/MOGGDialect/MOGGPasses.h.inc"

namespace {

// Outlines one `mogg._kernel`: its body moves into `stub`; the kernel's body is
// rebuilt as a single `mogg.call.stub` forwarding fresh block args to `stub`.
void outlineKernel(OpBuilder &builder, MOGG::KernelRegionOp kernel,
                   uint64_t stubNumber) {
  MLIRContext *ctx = kernel.getContext();
  Location loc = kernel.getLoc();

  // The body block args (input ptrs, output ptrs, dims, context) become the
  // stub's function signature; outputs are written through pointers, so the
  // stub returns nothing.
  Block &body = kernel.getBody().front();
  SmallVector<Type> argTypes(body.getArgumentTypes());
  auto functionType = TypeAttr::get(FunctionType::get(ctx, argTypes, {}));
  auto stubName = StringAttr::get(ctx, ("map_stub_" + Twine(stubNumber)).str());

  // Forward the fusion-tree provenance `MAPToMOGG` threaded onto `mogg._kernel`
  // from the source `map.iter.opaque`; fall back to a fresh leaf only if it's
  // absent (e.g. a kernel region built by hand in a test).
  MOGG::FusionInfoAttr fusionInfo = kernel.getFusionInfoAttr();
  if (!fusionInfo)
    fusionInfo = MOGG::FusionInfoAttr::get(ctx, MOGG::createFusionLeaf(kernel));

  // Place the stub just after the kernel's top-level ancestor for readable,
  // deterministic output.
  Operation *topAncestor = kernel;
  while (topAncestor->getParentOp() &&
         !isa<ModuleOp>(topAncestor->getParentOp()))
    topAncestor = topAncestor->getParentOp();
  builder.setInsertionPointAfter(topAncestor);

  auto stub = MOGG::KernelStubOp::create(
      builder, loc, stubName, functionType, fusionInfo,
      /*isAsync=*/nullptr, /*constantResources=*/ArrayAttr::get(ctx, {}),
      MOGG::ABIKindAttr::get(ctx, MOGG::ABIKind::MAP),
      /*trait_stub_buffer_capture=*/nullptr, /*trait_stub_arg_bounds=*/nullptr);
  stub.setDeviceRef(MO::getDeviceOrDefaultAny(kernel));

  // Move the body (block args + codegen ops) into the stub, then cap it with a
  // `mogg.output` terminator (the stub returns nothing).
  stub.getBody().takeBody(kernel.getBody());
  builder.setInsertionPointToEnd(&stub.getBody().front());
  MOGG::OutputOp::create(builder, loc, ValueRange{});

  // Rebuild the kernel body as a single `mogg.call.stub` forwarding fresh block
  // args (same types) to the stub.
  Block *callBody = builder.createBlock(&kernel.getBody());
  SmallVector<Value> forwarded;
  for (Type t : argTypes)
    forwarded.push_back(callBody->addArgument(t, loc));
  builder.setInsertionPointToEnd(callBody);
  MOGG::CallStubOp::create(builder, loc, /*resultTypes=*/TypeRange{},
                           FlatSymbolRefAttr::get(stubName), forwarded,
                           /*affinity_device_id=*/nullptr);
}

struct OutlineKernelsIntoFunctionsPass
    : public MOGG::impl::OutlineKernelsIntoFunctionsBase<
          OutlineKernelsIntoFunctionsPass> {
  using Base::Base;

  void runOnOperation() override {
    if (!MO::useAdvancedFusion())
      return;

    OpBuilder builder(&getContext());
    uint64_t stubNumber = 0;
    SmallVector<MOGG::KernelRegionOp> kernels;
    getOperation().walk([&](MOGG::KernelRegionOp k) {
      kernels.push_back(k);
      // No need to recurse inside a `mogg._kernel`.
      return WalkResult::skip();
    });
    for (MOGG::KernelRegionOp kernel : kernels)
      outlineKernel(builder, kernel, stubNumber++);
  }
};

} // namespace

} // namespace M::MOGG
