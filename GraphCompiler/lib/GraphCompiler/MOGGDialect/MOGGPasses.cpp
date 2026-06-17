//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/GraphCompiler/MOGGDialect/MOGGPasses.h"
#include "GraphCompiler/GraphCompiler/MOGGDialect/Support/KGENStubCompiler.h"
#include "GraphCompiler/GraphCompiler/Version.h"
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
  compilationOptions.targetAccelerator = target.getAcceleratorArch();
  compilationOptions.enableLLVMPerFunctionSplitting = false;

  if (kernelBuildType == KernelBuildType::kDebug)
    compilationOptions.optimizationLevel = 0;

  if (debugKGENStubs || hasDebuginfo(kernelBuildType))
    compilationOptions.debugLevel = KGEN::CompilationOptions::kFullDebugInfo;

  if (!saveTempsPrefix.empty())
    compilationOptions.setSaveTemps(saveTempsPrefix.str());

  return compilationOptions;
}

ErrorOr<TargetInfoAttr> M::MOGG::getHostTargetInfoAttr(MLIRContext *ctx) {
  ASSERT_STREAM(ctx != nullptr, "context must be non-null");

  // Resolve through getTargetInfo (the same chokepoint codegen uses) so the
  // key can never disagree with codegen about the target.
  initLLVM();

  ctx->loadDialect<M::MDialect>();

  return getTargetInfo(ctx);
}

ErrorOrSuccess M::MOGG::graphCompilerCacheKey(llvm::raw_ostream &keyBuf,
                                              TargetInfoAttr target,
                                              KernelBuildType buildType,
                                              bool debug,
                                              llvm::StringRef saveTempsPrefix) {
  // CompilationOptions::print() emits targetAccelerator but not the CPU
  // triple/cpu/features, so fold them in explicitly; else a host-warmed cache
  // falsely hits under a different CPU target (e.g. an x86-64-v3 artifact on a
  // v2 host, giving an illegal instruction).
  keyBuf << "target(" << target.getTripleStr() << ';' << target.getArch() << ';'
         << target.getFeatures() << ')';

  // Compilation target information
  createCompilationOptions(target, buildType, debug, saveTempsPrefix)
      .print(keyBuf);

  // Full version information for the graph compiler.
  keyBuf << MO::getGraphCompilerVersionString();

  return success();
}

void M::MOGG::GraphCompilerCacheKey::print(llvm::raw_ostream &os) {
  auto targetOr = getHostTargetInfoAttr(ctx);
  ASSERT_STREAM(!targetOr.isError(), << "unable to extract host information: "
                                     << targetOr.takeError());

  [[maybe_unused]] auto result =
      graphCompilerCacheKey(os, targetOr.takeValue(), this->buildType,
                            this->debug, this->saveTempsPrefix);

  ASSERT_STREAM(!result.isError(), << "unable to extract host information: "
                                   << result.takeError());
}
