//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/MOGGDialect/MOGGPasses.h"
#include "GenericML/GraphCompiler/Version.h"
#include "KGEN/KGENDialect/KGENCompilationContext.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "KGEN/ToolCommon/CompilationOptions.h"
#include "MLRT/AsyncRT/CompilerSupport/Context.h"
#include "Support/AssertStream.h"
#include "Support/MArchTarget/MArchTargetMinimal.h"
#include "llvm/Support/TargetSelect.h"

using namespace M;
using namespace M::MOGG;

void M::MOGG::initLLVM() {
  static std::once_flag initLLVMFlag;
  std::call_once(initLLVMFlag, []() {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmParser();
    llvm::InitializeNativeTargetAsmPrinter();

    llvm::InitializeAllTargets();
    llvm::InitializeAllTargetInfos();
    llvm::InitializeAllTargetMCs();
    llvm::InitializeAllAsmParsers();
    llvm::InitializeAllAsmPrinters();
  });
}

KGEN::CompilationOptions M::MOGG::createCompilationOptions(
    TargetInfoAttr target, KernelBuildType kernelBuildType, bool debugKGENStubs,
    StringRef saveTempsPrefix) {

  KGEN::CompilationOptions compilationOptions;
  compilationOptions.targetTriple = target.getTripleStr();
  compilationOptions.targetCpu = target.getArch();
  compilationOptions.targetFeatures = target.getFeatures();
  compilationOptions.targetAccelerator =
      M::Driver::Device::getAcceleratorArchOrEmpty();
  compilationOptions.enableLLVMPerFunctionSplitting = false;

  if (kernelBuildType == KernelBuildType::kDebug)
    compilationOptions.optimizationLevel = 0;

  if (debugKGENStubs || hasDebuginfo(kernelBuildType))
    compilationOptions.debugLevel = KGEN::CompilationOptions::kFullDebugInfo;

  if (!saveTempsPrefix.empty())
    compilationOptions.setSaveTemps(saveTempsPrefix.str());

  return compilationOptions;
}

ErrorOrSuccess M::MOGG::graphCompilerCacheKey(llvm::raw_ostream &keyBuf,
                                              MLIRContext *ctx,
                                              KernelBuildType buildType,
                                              bool debug,
                                              llvm::StringRef saveTempsPrefix) {
  ASSERT_STREAM(ctx != nullptr, "context must be non-null");

  // Full target information for the host, including target accelerators.
  initLLVM();

  ctx->loadDialect<M::MDialect>();

  ErrorOr<TargetInfoAttr> targetOr = getTargetInfoFor(
      ctx, llvm::sys::getDefaultTargetTriple(), llvm::sys::getHostCPUName(),
      getHostCPUFeatures(), /*tuneCpu=*/"",
      /*acceleratorArch=*/M::Driver::Device::getAcceleratorArchOrEmpty());

  if (targetOr.isError())
    return targetOr.takeError();

  TargetInfoAttr target = targetOr.takeValue();

  // Compilation target information
  createCompilationOptions(target, buildType, debug, saveTempsPrefix)
      .print(keyBuf);

  // Full version infomration for the graph compiler.
  keyBuf << MO::getGraphCompilerVersionString();

  return success();
}

void M::MOGG::GraphCompilerCacheKey::print(llvm::raw_ostream &os) {
  [[maybe_unused]] auto result = graphCompilerCacheKey(
      os, ctx, this->buildType, this->debug, this->saveTempsPrefix);

  ASSERT_STREAM(!result.isError(), << "unable to extract host information: "
                                   << result.takeError());
}
