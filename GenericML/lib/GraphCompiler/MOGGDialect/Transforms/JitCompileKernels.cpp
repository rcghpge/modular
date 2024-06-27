//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/MOGGDialect/Support/MOGGCompiler.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Threading.h"
#include "mlir/Pass/PassManager.h"

#include "GenericML/GraphCompiler/MODialect/MOOps.h"
#include "GenericML/GraphCompiler/MODialect/MOTypes.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGPasses.h"
#include "GenericML/GraphCompiler/MOGGDialect/Support/NameMangling.h"
#include "GenericML/MGPDialect/MGPOps.h"

#include "ArraySupport/Tensor.h"

#include "KGEN/Compiler/KGENCompiler.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/ToolCommon/KGENPasses.h"

#include "mlir/IR/IRMapping.h"

#include <GenericML/GraphCompiler/MOGGDialect/Support/MOGGCompiler.h>
using namespace M;

namespace M::MOGG {

#define GEN_PASS_DEF_JITCOMPILEKERNELS
#include "GenericML/GraphCompiler/MOGGDialect/MOGGPasses.h.inc"

namespace {

// Convert all possible operations directly to the avaliable operation.
struct JitCompileKernels
    : public impl::JitCompileKernelsBase<JitCompileKernels> {
  using JitCompileKernelsBase::JitCompileKernelsBase;

  void runOnOperation() override {
    mlir::ModuleOp m = getOperation();
    MLIRContext *ctx = m.getContext();

    auto &analysis = getAnalysis<MOGG::LoadMojoAnalysisPass>();
    ErrorOr<MOGG::MojoLibraryAnalysis *> errOr =
        analysis.initIfNeeded(ctx, kernelLibraryPaths, true);
    if (errOr.isError())
      return signalPassFailure();
    MOGG::MojoLibraryAnalysis &mojoLibraryAnalysis = *errOr.takeValue();

    // Run the jit required passes.
    mlir::PassManager pm(ctx);
    pm.addNestedPass<MO::GraphOp>(createInlineLambdas());
    if (failed(pm.run(m)))
      return signalPassFailure();

    // Check if any ops are graph ops so we know if we are the last compile.
    bool isGraph = !m.getOps<MO::GraphOp>().empty();

    auto func = [&](auto graph) {
      auto errOr = this->runPass(graph, mojoLibraryAnalysis);
      if (errOr.isError())
        return this->signalPassFailure();
    };

    auto range = m.getOps<MO::GraphOp>();
    mlir::parallelForEach(ctx, range.begin(), range.end(), func);

    auto modelRange = m.getOps<MGP::ModelOp>();
    mlir::parallelForEach(ctx, modelRange.begin(), modelRange.end(), func);

    // Strip all KGEN forward decls. There is now nothing they can be lowered
    // to. We only do this on the second compile (MGP models). This will be
    // removed once we only have one compile stage.
    if (!isGraph)
      m.walk([](KGEN::ExternGeneratorOp op) { op.erase(); });
  }

  template <typename OpContainer>
  CompileDiagnosticOrSuccess
  runPass(OpContainer graph, MOGG::MojoLibraryAnalysis &mojoLibraryAnalysis) {
    // We can optionally override
    DenseMap<Operation *, std::string> unusedOverrideMap;

    // For each (virtual) device used by the graph...
    auto err = compileToBinary(
        graph, mojoLibraryAnalysis, unusedOverrideMap,
        skipJitCompileMiddle.getValue(), dumpStub.getValue(),
        createMOGGReproducers.getValue(), useSearch.getValue(),
        saveTempPrefix.getValue(), minimumCpuAlignment.getValue(),
        splitBinaryCompile.getValue(), timeMOGGCompile.getValue());

    if (err.isError())
      return err.takeError();
    return success();
  }
};

} // namespace

} // namespace M::MOGG
