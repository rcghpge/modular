//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/KGENStubBuilder/StubCompiler.h"

#include "mlir/Pass/PassManager.h"

#include "LLCL/Runtime/Runtime.h"

#include "KGEN/KGENCompiler.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/LowerToObject.h"

#include "GenericML/MOGGDialect/MOGGOps.h"

namespace M::MOGG::Stub {

LogicalResult StubCompiler::elaborateKGEN() {
  // Create a runtime for KGEN to evaluate the kernels.
  LLCL::Runtime runtime(
      LLCL::createLeakCheckAllocator(LLCL::createMallocAllocator()),
      LLCL::createSingleThreadWorkQueue());

  SmallVector<std::string> includedFiles{};

  // Only option is to turn on search, no include paths as we don't include
  // files in the stub currently.
  KGEN::ElaborateGeneratorsOptions options;
  options.shouldDoSearch = shouldSearch;

  mlir::PassManager pm(kgenModule->getContext());
  KGEN::elaborateModule(pm, runtime, options, includedFiles);

  return pm.run(kgenModule);
}

ErrorOrSuccess StubCompiler::compileToLLVM(StringRef symbolName) {
  LLCL::Runtime runtime(
      LLCL::createLeakCheckAllocator(LLCL::createMallocAllocator()),
      LLCL::createSingleThreadWorkQueue());

  // Compile to LLVM.
  mlir::SymbolTable symTable{kgenModule};
  KGEN::CompilationOptions compilationOptions;
  auto compiler = KGEN::ObjectCompiler::create(runtime, ".kgen_cache", symTable,
                                               compilationOptions);
  ErrorOr<TargetInfoAttr> target = KGEN::getHostTargetInfo(ctx);
  if (target.isError())
    return target.takeError();

  auto standaloneOr =
      compiler->produceStandaloneObjectAttr(target.takeValue(), true);
  if (failed(standaloneOr))
    return Error("Failed to create object attribute");

  auto objectAttr = std::move(*standaloneOr);

  // Add the kernel as a constant.
  moggBuilder.create<MOGG::LoadJitKernelOp>(
      loc, moggBuilder.getStringAttr(symbolName), objectAttr);

  return success();
}

ErrorOrSuccess StubCompiler::compile(StringRef symbolName) {
  LogicalResult ok = elaborateKGEN();
  if (ok.failed())
    return Error("KGEN elaboration failed");

  return compileToLLVM(symbolName);
}

} // namespace M::MOGG::Stub
