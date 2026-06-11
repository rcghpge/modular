//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/GraphCompiler/MOGGDialect/Analysis/Support.h"
#include "GraphCompiler/GraphCompiler/MOGGDialect/Analysis/KernelDeclAdaptor.h"
#include "GraphCompiler/GraphCompiler/MOGGDialect/Analysis/MojoRegistry.h"
#include "GraphCompiler/GraphCompiler/MOGGDialect/MOGGOps.h"
#include "KGEN/KGENDialect/KGENOps.h"

namespace M::MOGG {

int64_t getNumDPSStyleOutputs(KGEN::GeneratorOp mojoCode) {
  auto count = mojoCode->getAttr(MOGG::kMoggNumDpsOutputs);
  [[maybe_unused]] bool isExecuteFunc =
      mojoCode->hasAttr(MOGG::kMoggExecuteFunctionLabel);

  if (!count) {
    ASSERT_STREAM(!isExecuteFunc,
                  << "The kernel's entry point must have number of dps "
                     "operands explicitly labeled. Kernel is "
                  << mojoCode.getName());

    return 0;
  }

  ASSERT_STREAM(isExecuteFunc, << "Only DPS operands are supported in the "
                                  "kernel's entry point. Kernel is "
                               << mojoCode.getName());
  return cast<IntegerAttr>(count).getInt();
}

/// Returns the name of the given `op` is registered under in mojo.
MojoOperationInfo getMojoOperationInfo(Operation *op) {
  // Custom operations call the symbol and function directly.
  if (auto customOp = dyn_cast<MO::CustomOp>(op))
    return {customOp.getSymbol(), MOGG::kMoggExecuteFunctionLabel};

  // Layout transforms map to a hardcoded op name based on the inferred layouts
  if (auto layoutTransformOp = dyn_cast<MO::LayoutTransformOp>(op)) {
    auto startLayout = layoutTransformOp.getInput().getType().getLayoutAttr();
    auto endLayout = layoutTransformOp.getResult().getType().getLayoutAttr();

    StringAttr opName = StringAttr::get(
        op->getContext(), llvm::Twine("layout_transform_") +
                              startLayout.getFormat().strref() + "_to_" +
                              endLayout.getFormat().strref());
    return {opName, MOGG::kMoggExecuteFunctionLabel};
  }

  // Shape invocations call the shape functions
  if (auto shapeInvoke = dyn_cast<MO::InvokeShapeFuncOp>(op))
    return {shapeInvoke.getShapeFuncName(), MOGG::kMoggShapeFunctionLabel};

  // Else we register 1-1 with the official name of the operation.
  return {op->getName().getStringRef(), MOGG::kMoggExecuteFunctionLabel};
}

ErrorOr<KernelDeclAdaptor> getMojoKernelAdaptor(const MojoRegistry &registry,
                                                Operation *op) {
  MojoOperationInfo info = getMojoOperationInfo(op);

  if (!registry.containsKernel(info.operationName)) {
    return Error(
        llvm::formatv("no kernel registered for '{}'", info.operationName)
            .str());
  }
  KernelDeclOp kernelDecl = registry.getResolvedKernel(info.operationName);
  KernelDeclAdaptor adaptor(kernelDecl);

  if (!adaptor.executeFunction.has_value() &&
      !adaptor.elementwiseFunction.has_value()) {
    return Error(
        llvm::formatv(
            "Could not find an execute or elementwise function for '{}'",
            info.operationName)
            .str());
  }

  return adaptor;
}

} // namespace M::MOGG
