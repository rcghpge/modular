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

#include "GraphCompiler/MODialect/MODevices.h"
#include "GraphCompiler/MODialect/MODialect.h"
#include "GraphCompiler/MOGGDialect/MOGGDialect.h"
#include "GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GraphCompiler/MOGGDialect/MOGGPasses.h"
#include "GraphCompiler/MOGGDialect/Support/FusionUtils.h"
#include "mlir/Analysis/SymbolTableAnalysis.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"

using namespace mlir;
using namespace M;

namespace M::MOGG {

#define GEN_PASS_DEF_OUTLINEKERNELSINTOFUNCTIONS
#include "GraphCompiler/MOGGDialect/MOGGPasses.h.inc"

namespace {

// Outlines one `mogg._kernel`: its body moves into `stub`; the kernel's body is
// rebuilt as a single `mogg.call.stub` forwarding fresh block args to `stub`.
void outlineKernel(OpBuilder &builder, SymbolTable &symtab,
                   MOGG::KernelRegionOp kernel) {
  MLIRContext *ctx = kernel.getContext();
  Location loc = kernel.getLoc();

  // The body block args (input ptrs, output ptrs, dims, context) become the
  // stub's function signature; outputs are written through pointers, so the
  // stub returns nothing.
  Block &body = kernel.getBody().front();
  SmallVector<Type> argTypes(body.getArgumentTypes());

  // A `mogg.buffer.constant` stays behind at the call site: `MOGGToMGP` turns
  // it into the real `mgp.buffer.constant` in the region it builds there,
  // because resources reach the runtime through the `void**` of the MGP-ABI
  // entry point, which this MAP-ABI stub does not have. Each one becomes a
  // trailing stub argument here, while the signature is still being built --
  // growing it afterwards would mean patching every call site too.
  SmallVector<MOGG::BufferConstantOp> constants;
  body.walk([&](MOGG::BufferConstantOp cst) { constants.push_back(cst); });

  SmallVector<Type> stubArgTypes(argTypes);
  stubArgTypes.append(constants.size(), MOGG::PointerType::get(ctx));
  auto functionType = TypeAttr::get(FunctionType::get(ctx, stubArgTypes, {}));

  // Forward the fusion-tree provenance `MAPToMOGG` threaded onto `mogg._kernel`
  // from the source `map.iter.opaque`; fall back to a fresh leaf only if it's
  // absent (e.g. a kernel region built by hand in a test).
  MOGG::FusionInfoAttr fusionInfo = kernel.getFusionInfoAttr();
  if (!fusionInfo)
    fusionInfo = MOGG::FusionInfoAttr::get(ctx, MOGG::createFusionLeaf(kernel));

  // Name the stub after the ops it fuses, e.g. "relu_add_map"; the symbol table
  // appends a numeric suffix below when two kernels fuse the same ops.
  auto stubName = MOGG::getKernelStubName(fusionInfo, MOGG::ABIKind::MAP);

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

  // `create` already linked the stub into the module; registering it here
  // resolves any name collision and returns the unique name for the call site.
  stubName = symtab.insert(stub);

  // Move the body (block args + codegen ops) into the stub, then cap it with a
  // `mogg.output` terminator (the stub returns nothing).
  stub.getBody().takeBody(kernel.getBody());
  Block &stubBody = stub.getBody().front();
  // The constants came along with the body; rewire their uses onto the trailing
  // arguments reserved above, leaving the ops themselves unused and free to
  // move back to the call site.
  for (MOGG::BufferConstantOp cst : constants)
    cst.getPointer().replaceAllUsesWith(
        stubBody.addArgument(MOGG::PointerType::get(ctx), cst.getLoc()));
  builder.setInsertionPointToEnd(&stubBody);
  MOGG::OutputOp::create(builder, loc, ValueRange{});

  // Rebuild the kernel body as a single `mogg.call.stub` forwarding fresh block
  // args (same types) to the stub, preceded by the constants it hands over.
  Block *callBody = builder.createBlock(&kernel.getBody());
  SmallVector<Value> forwarded;
  for (Type t : argTypes)
    forwarded.push_back(callBody->addArgument(t, loc));
  builder.setInsertionPointToEnd(callBody);
  for (MOGG::BufferConstantOp cst : constants) {
    cst->moveBefore(callBody, callBody->end());
    forwarded.push_back(cst.getPointer());
  }
  MOGG::CallStubOp::create(builder, loc, /*resultTypes=*/TypeRange{},
                           FlatSymbolRefAttr::get(stubName), forwarded,
                           /*affinity_device_id=*/nullptr);
}

struct OutlineKernelsIntoFunctionsPass
    : public MOGG::impl::OutlineKernelsIntoFunctionsBase<
          OutlineKernelsIntoFunctionsPass> {
  using Base::Base;

  void runOnOperation() override {
    OpBuilder builder(&getContext());
    // Register outlined stubs through the module's cached `SymbolTableAnalysis`
    // so their names stay collision-free and the table stays consistent for
    // downstream consumers.
    SymbolTable &symtab =
        getAnalysis<mlir::SymbolTableAnalysis>().getTopLevelSymbolTable();
    SmallVector<MOGG::KernelRegionOp> kernels;
    getOperation().walk([&](MOGG::KernelRegionOp k) {
      kernels.push_back(k);
      // No need to recurse inside a `mogg._kernel`.
      return WalkResult::skip();
    });
    for (MOGG::KernelRegionOp kernel : kernels)
      outlineKernel(builder, symtab, kernel);
  }
};

} // namespace

} // namespace M::MOGG
