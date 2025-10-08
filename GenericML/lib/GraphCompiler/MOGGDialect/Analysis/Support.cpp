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

template <>
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

template <>
ErrorOr<KGEN::LIT::FnOp>
getMojoKernelTarget(const MOGG::MojoLibraryAnalysis &mojoLibraryAnalysis,
                    StringRef mojoOperationName, StringRef functionLabel) {
  auto generatorOp = mojoLibraryAnalysis.getRegisteredLITMojoFunctionFor(
      mojoOperationName, functionLabel);

  if (!generatorOp)
    return Error(Twine("Could not find a mojo kernel registered for ") +
                 mojoOperationName + " with function " + functionLabel);
  return generatorOp;
}

template <class FuncT>
ErrorOr<FuncT>
getMojoKernelTarget(const MOGG::MojoLibraryAnalysis &mojoLibraryAnalysis,
                    Operation *op) {
  MojoOperationInfo info = getMojoOperationInfo(op);
  return getMojoKernelTarget<FuncT>(mojoLibraryAnalysis, info.operationName,
                                    info.functionLabel);
}

template <class FuncT>
static FuncT
getElementwiseFunc(const MOGG::MojoLibraryAnalysis &mojoLibraryAnalysis,
                   StringRef name) {
  if constexpr (std::is_same_v<FuncT, KGEN::GeneratorOp>) {
    return mojoLibraryAnalysis.getRegisteredMojoElementwiseFunctionFor(name);
  } else if constexpr (std::is_same_v<FuncT, KGEN::LIT::FnOp>) {
    return mojoLibraryAnalysis.getRegisteredLITMojoElementwiseFunctionFor(name);
  } else {
    llvm_unreachable("Invalid type");
    return nullptr;
  }
}

template <class FuncT>
ErrorOr<KGEN::MOGGPreElab::MojoKernelAdaptor<FuncT>>
getMojoKernelAdaptor(const MOGG::MojoLibraryAnalysis &mojoLibraryAnalysis,
                     Operation *op) {
  // Find the kernel from the library.
  MojoOperationInfo info = getMojoOperationInfo(op);
  ErrorOr<FuncT> maybeExecuteFunc = getMojoKernelTarget<FuncT>(
      mojoLibraryAnalysis, info.operationName, info.functionLabel);
  FuncT executeFunc = maybeExecuteFunc.isError() ? nullptr : *maybeExecuteFunc;
  FuncT elemtwiseFunc =
      getElementwiseFunc<FuncT>(mojoLibraryAnalysis, info.operationName);
  // We either need an execute or an elementwise function.
  if (!executeFunc && !elemtwiseFunc) {
    return maybeExecuteFunc.takeError();
  }

  return KGEN::MOGGPreElab::MojoKernelAdaptor<FuncT>(
      "", /*execute=*/executeFunc, /*shape=*/nullptr,
      /*elementwise=*/elemtwiseFunc);
}

template ErrorOr<KGEN::GeneratorOp> getMojoKernelTarget<KGEN::GeneratorOp>(
    const MOGG::MojoLibraryAnalysis &mojoLibraryAnalysis, Operation *op);
template ErrorOr<KGEN::LIT::FnOp> getMojoKernelTarget<KGEN::LIT::FnOp>(
    const MOGG::MojoLibraryAnalysis &mojoLibraryAnalysis, Operation *op);

template ErrorOr<KGEN::MOGGPreElab::MojoKernelAdaptor<KGEN::GeneratorOp>>
getMojoKernelAdaptor<KGEN::GeneratorOp>(
    const MOGG::MojoLibraryAnalysis &mojoLibraryAnalysis, Operation *op);
template ErrorOr<KGEN::MOGGPreElab::MojoKernelAdaptor<KGEN::LIT::FnOp>>
getMojoKernelAdaptor<KGEN::LIT::FnOp>(
    const MOGG::MojoLibraryAnalysis &mojoLibraryAnalysis, Operation *op);

} // namespace M::MOGG
