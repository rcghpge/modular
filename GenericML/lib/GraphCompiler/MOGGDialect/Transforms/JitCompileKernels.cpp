//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/MOGGDialect/Support/MOGGCompiler.h"
#include "GenericML/GraphCompiler/MOGGDialect/Support/MojoTargetFunctions.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Threading.h"
#include "mlir/Pass/PassManager.h"

#include "GenericML/GraphCompiler/MODialect/MOOps.h"
#include "GenericML/GraphCompiler/MODialect/MOTypes.h"
#include "GenericML/GraphCompiler/MOGGDialect/Analysis/MojoLibraryAnalysis.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGPasses.h"
#include "GenericML/MGPDialect/MGPOps.h"

#include "Runtime/Tensor/Tensor.h"

#include "KGEN/Compiler/KGENCompiler.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/Support/CompilerProfiling.h"
#include "KGEN/ToolCommon/KGENPasses.h"
#include "mlir/IR/IRMapping.h"

#include <GenericML/GraphCompiler/MOGGDialect/Support/MOGGCompiler.h>
using namespace M;

namespace M::MOGG {

#define GEN_PASS_DEF_JITCOMPILEKERNELS
#include "GenericML/GraphCompiler/MOGGDialect/MOGGPasses.h.inc"

namespace {

// Convert all possible operations directly to the available operation.
struct JitCompileKernels
    : public impl::JitCompileKernelsBase<JitCompileKernels> {
  using JitCompileKernelsBase::JitCompileKernelsBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();

    KGEN::CompilerTimeTraceScope traceScope("jit-compile-kernels",
                                            module->getName().getStringRef());

    auto &analysis = getAnalysis<MOGG::LoadMojoAnalysisPass>();
    ErrorOr<MOGG::MojoLibraryAnalysis *> errOr =
        analysis.initIfNeeded(ctx, kernelLibraryPaths);
    if (errOr.isError())
      return signalPassFailure();
    MOGG::MojoLibraryAnalysis &mojoLibraryAnalysis = *errOr.takeValue();

    for (auto model : module.getOps<MGP::ModelOp>()) {
      auto errOr = runPass(model, mojoLibraryAnalysis);
      if (errOr.isError()) {
        signalPassFailure();
        return;
      }
    }

    // Strip all KGEN forward decls. There is now nothing they can be lowered
    // to.
    module.walk([](KGEN::ExternGeneratorOp op) { op.erase(); });
    module.walk([](KGEN::GeneratorOp gen) { gen.erase(); });
  }

  CompileDiagnosticOrSuccess
  runPass(MGP::ModelOp graph, MOGG::MojoLibraryAnalysis &mojoLibraryAnalysis) {
    KGEN::CompilerTimeTraceScope traceScope("jit-compile-kernels-run-on-pass",
                                            graph->getName().getStringRef());

    KernelBuildType bt = KernelBuildType::kRelease;

    // For each (virtual) device used by the graph...
    auto err = compileToBinary(
        graph, mojoLibraryAnalysis, dumpStub.getValue(),
        createMOGGReproducers.getValue(), saveTempPrefix.getValue(),
        minimumCpuAlignment.getValue(), timeMOGGCompile.getValue(),
        dryRun.getValue(), debugKGENStubs.getValue(), bt);

    if (err.isError())
      return err.takeError();
    return success();
  }
};

} // namespace

} // namespace M::MOGG
