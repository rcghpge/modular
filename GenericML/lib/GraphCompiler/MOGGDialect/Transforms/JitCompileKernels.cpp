//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/MOGGSupport/MOGGCompiler.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassManager.h"

#include "GenericML/GraphCompiler/MODialect/MOOps.h"
#include "GenericML/GraphCompiler/MODialect/MOTypes.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGPasses.h"
#include "GenericML/GraphCompiler/MOGGSupport/NameMangling.h"
#include "GenericML/Support/Tensor.h"

#include "KGEN/Compiler/KGENCompiler.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/ToolCommon/KGENPasses.h"

#include "mlir/IR/IRMapping.h"

#include <GenericML/GraphCompiler/MOGGSupport/MOGGCompiler.h>
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

    auto mojoLibraryAnalysisOrErr =
        MOGG::MojoLibraryAnalysis::get(ctx, extraLibPaths);
    assert(!mojoLibraryAnalysisOrErr.isError());
    MOGG::MojoLibraryAnalysis mojoLibraryAnalysis =
        mojoLibraryAnalysisOrErr.takeValue();

    // Don't expose any extra stuff in the library.
    mojoLibraryAnalysis.markAllFunctionsAsNotExported();

    // Run the jit required passes.
    mlir::PassManager pm(ctx);
    pm.addNestedPass<MO::GraphOp>(createInlineLambdas());
    if (failed(pm.run(m)))
      return signalPassFailure();

    for (auto graph : m.getOps<MO::GraphOp>()) {
      auto errOr = this->runPass(graph, mojoLibraryAnalysis);
      if (errOr.isError()) {
        graph.emitError() << "[JitCompileKernels] " << errOr.getError();
        return signalPassFailure();
      }
    }

    // Strip all KGEN forward decls. There is now nothing they can be lowered
    // to.
    m.walk([](KGEN::ExternGeneratorOp op) { op.erase(); });
  }

  ErrorOrSuccess runPass(MO::GraphOp graph,
                         MOGG::MojoLibraryAnalysis &mojoLibraryAnalysis) {
    ErrorOr<DeviceSpecCollectionAttr> deviceSpecsOr =
        MO::getDeviceSpecsOrDefault(graph);
    if (deviceSpecsOr)
      return deviceSpecsOr.takeError();
    DeviceSpecAttr hostDeviceSpec = deviceSpecsOr->getHostDeviceSpec();

    // For each (virtual) device used by the graph...
    for (DeviceSpecAttr deviceSpec : deviceSpecsOr->getDevices()) {
      auto err = compileToBinary(
          graph, mojoLibraryAnalysis, deviceSpec, hostDeviceSpec,
          dumpStub.getValue(), createMOGGReproducers.getValue(),
          useSearch.getValue(), saveTempPrefix.getValue(),
          minimumCpuAlignment.getValue());

      if (err.isError())
        return err.takeError();
    }
    return success();
  }
};

} // namespace

} // namespace M::MOGG
