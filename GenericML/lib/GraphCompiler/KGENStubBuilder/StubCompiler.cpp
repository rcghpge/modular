//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/KGENStubBuilder/StubCompiler.h"

#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"

#include "LLCL/Runtime/Runtime.h"

#include "GenericML/GraphCompiler/MOGGDialect/MOGGOps.h"
#include "KGEN/CLOptions.h"
#include "KGEN/KGENCompiler.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "KGEN/LowerToObject.h"
#include "Support/Host.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/TargetParser/Host.h"

namespace M::MOGG::Stub {

LogicalResult StubCompiler::elaborateKGEN() {
  // Create a runtime for KGEN to evaluate the kernels.
  LLCL::Runtime runtime(
      LLCL::createLeakCheckAllocator(LLCL::createMallocAllocator()),
      LLCL::createThreadPoolWorkQueue(
          /*numThreads=*/
          M::getHostMachineInfo()->numPhysicalCores));

  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmParser();
  llvm::InitializeNativeTargetAsmPrinter();

  ErrorOr<TargetInfoAttr> target = KGEN::getTargetInfoFor(
      kgenModule->getContext(), llvm::sys::getDefaultTargetTriple(),
      llvm::sys::getHostCPUName(), getHostCPUFeatures());
  if (target.isError())
    return mlir::emitError(kgenModule.getLoc(), target.getError());

  BuildInfoAttr buildInfo =
      BuildInfoAttr::getForCurrentBuild(kgenModule->getContext());

  mlir::PassManager pm(kgenModule->getContext());
  KGEN::populateElaborateModulePasses(pm, runtime, *target, buildInfo,
                                      {shouldSearch});

  if (!saveTempsPrefix.empty()) {
    std::string outPath = saveTempsPrefix + ".kgen.prelaboration.mlir";
    auto outFile = mlir::openOutputFile(outPath);
    if (!outFile)
      return failure();

    outFile->os() << kgenModule;
    outFile->keep();
  }

  auto res = pm.run(kgenModule);

  if (!saveTempsPrefix.empty()) {
    std::string outPath = saveTempsPrefix + ".kgen.elaborated.mlir";
    auto outFile = mlir::openOutputFile(outPath);
    if (!outFile)
      return failure();
    outFile->os() << kgenModule;
    outFile->keep();
  }

  return res;
}

ErrorOrSuccess StubCompiler::compileToLLVM(mlir::ArrayAttr combinedSymbolName) {
  LLCL::Runtime runtime(
      LLCL::createLeakCheckAllocator(LLCL::createMallocAllocator()),
      LLCL::createThreadPoolWorkQueue(
          /*numThreads=*/
          M::getHostMachineInfo()->numPhysicalCores));
  mlir::PassManager mgr(ctx);

  // Compile to LLVM.
  mlir::SymbolTable symTable{kgenModule};
  KGEN::CompilationOptions compilationOptions;

  if (!saveTempsPrefix.empty())
    compilationOptions.setSaveTemps(saveTempsPrefix);

  auto compilerOr = KGEN::ObjectCompiler::create(runtime, mgr, ".kgen_cache",
                                                 compilationOptions);
  if (compilerOr.isError())
    return compilerOr.takeError();
  llvm::MapVector<StringAttr, KGEN::ExportedSymbol> exportedSymbols =
      KGEN::getExportedSymbols(kgenModule);

  KGEN::ObjectCompiler compiler = compilerOr.takeValue();

  TargetInfoAttr target = getTargetInfo(kgenModule);

  auto standaloneOr = compiler.produceStandaloneArchiveAttr(
      symTable, exportedSymbols, target, true);
  if (failed(standaloneOr))
    return Error("Failed to create archive attribute");
  ElementsAttr archiveAttr = *standaloneOr;

  // Add the kernel as a constant.
  moggBuilder.create<MOGG::LoadJitKernelOp>(loc, combinedSymbolName,
                                            archiveAttr);
  return success();
}

ErrorOrSuccess StubCompiler::compile(mlir::ArrayAttr combinedSymbolName) {
  LogicalResult ok = elaborateKGEN();
  if (ok.failed())
    return Error("KGEN elaboration failed");
  return compileToLLVM(combinedSymbolName);
}

} // namespace M::MOGG::Stub
