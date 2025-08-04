//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/MOGGDialect/Analysis/Support.h"
#include "GenericML/GraphCompiler/MOGGDialect/Analysis/MojoLibraryAnalysis.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GenericML/GraphCompiler/MOGGDialect/Support/TypeManager.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/LITDialect/LITUtils.h"
#include "KGEN/MOGGPreElab/MOGGPreElabDecorators.h"

namespace M::MOGG {

int64_t getNumDPSStyleOutputs(KGEN::GeneratorOp mojoCode) {
  auto count = mojoCode->getAttr(KGEN::MOGGPreElab::kMOGGNumDPSOutputs);
  [[maybe_unused]] bool isExecuteFunc =
      mojoCode->hasAttr(KGEN::MOGGPreElab::kMOGGExecuteFunctionLabel);

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

namespace {
// Records the operation (e.g. mo.add) and extensibility function e.g. @shape,
// @execute, etc.
struct MojoOperationInfo {
  StringRef operationName, functionLabel;
};

} // namespace

/// Returns the name of the given `op` is registered under in mojo.
static MojoOperationInfo getMojoOperationInfo(Operation *op) {
  // Custom operations call the symbol and function directly.
  if (auto customOp = dyn_cast<MO::CustomOp>(op))
    return {customOp.getSymbol(), KGEN::MOGGPreElab::kMOGGExecuteFunctionLabel};

  // Layout transforms map to a hardcoded op name based on the inferred layouts
  if (auto layoutTransformOp = dyn_cast<MO::LayoutTransformOp>(op)) {
    auto startLayout = layoutTransformOp.getInput().getType().getLayoutAttr();
    auto endLayout = layoutTransformOp.getResult().getType().getLayoutAttr();

    StringAttr opName = StringAttr::get(
        op->getContext(), llvm::Twine("layout_transform_") +
                              startLayout.getFormat().strref() + "_to_" +
                              endLayout.getFormat().strref());
    return {opName, KGEN::MOGGPreElab::kMOGGExecuteFunctionLabel};
  }

  // Shape invocations call the shape functions
  if (auto shapeInvoke = dyn_cast<MO::InvokeShapeFuncOp>(op))
    return {shapeInvoke.getShapeFuncName(),
            KGEN::MOGGPreElab::kMOGGShapeFunctionLabel};

  // Else we register 1-1 with the official name of the operation.
  return {op->getName().getStringRef(),
          KGEN::MOGGPreElab::kMOGGExecuteFunctionLabel};
}

ErrorOr<KGEN::GeneratorOp>
getMojoKernelTarget(const MOGG::MojoLibraryAnalysis &mojoLibraryAnalysis,
                    StringRef mojoOperationName, StringRef functionLabel) {
  auto generatorOp = mojoLibraryAnalysis.getRegisteredMojoFunctionFor(
      mojoOperationName, functionLabel);

  if (!generatorOp)
    return Error(Twine("Could not find a mojo kernel registered for ") +
                 mojoOperationName + " with function " + functionLabel);
  return generatorOp;
}

ErrorOr<KGEN::GeneratorOp>
getMojoKernelTarget(const MOGG::MojoLibraryAnalysis &mojoLibraryAnalysis,
                    Operation *op) {
  MojoOperationInfo info = getMojoOperationInfo(op);
  return getMojoKernelTarget(mojoLibraryAnalysis, info.operationName,
                             info.functionLabel);
}

ErrorOr<KernelAdaptor>
getMojoKernelAdaptor(const MOGG::MojoLibraryAnalysis &mojoLibraryAnalysis,
                     Operation *op) {
  // Find the kernel from the library.
  MojoOperationInfo info = getMojoOperationInfo(op);
  ErrorOr<KGEN::GeneratorOp> maybeExecuteFunc = getMojoKernelTarget(
      mojoLibraryAnalysis, info.operationName, info.functionLabel);
  KGEN::GeneratorOp executeFunc =
      maybeExecuteFunc.isError() ? nullptr : *maybeExecuteFunc;
  KGEN::GeneratorOp elemtwiseFunc =
      mojoLibraryAnalysis.getRegisteredMojoElementwiseFunctionFor(
          info.operationName);
  // We either need an execute or an elementwise function.
  if (!executeFunc && !elemtwiseFunc) {
    return maybeExecuteFunc.takeError();
  }

  return KernelAdaptor("", /*execute=*/executeFunc, /*shape=*/nullptr,
                       /*elementwise=*/elemtwiseFunc);
}

} // namespace M::MOGG
