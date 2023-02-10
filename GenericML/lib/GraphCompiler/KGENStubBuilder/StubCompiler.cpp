//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/KGENStubBuilder/StubCompiler.h"

#include "mlir/Pass/PassManager.h"

#include "LLCL/Runtime/Runtime.h"

#include "KGEN/KGENCompiler.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/LowerToObject.h"

#include "GenericML/GraphCompiler/MOGGDialect/MOGGOps.h"

namespace M::MOGG::Stub {

LogicalResult StubCompiler::elaborateKGEN() {
  // Create a runtime for KGEN to evaluate the kernels.
  LLCL::Runtime runtime(
      LLCL::createLeakCheckAllocator(LLCL::createMallocAllocator()),
      LLCL::createSingleThreadWorkQueue());

  mlir::PassManager pm(kgenModule->getContext());
  KGEN::elaborateModule(pm, runtime, {shouldSearch});

  return pm.run(kgenModule);
}

ErrorOrSuccess StubCompiler::compileToLLVM(mlir::ArrayAttr combinedSymbolName) {
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
  moggBuilder.create<MOGG::LoadJitKernelOp>(loc, combinedSymbolName,
                                            objectAttr);
  return success();
}

ErrorOrSuccess StubCompiler::compile(mlir::ArrayAttr combinedSymbolName) {
  LogicalResult ok = elaborateKGEN();
  if (ok.failed())
    return Error("KGEN elaboration failed");
  return compileToLLVM(combinedSymbolName);
}

} // namespace M::MOGG::Stub
