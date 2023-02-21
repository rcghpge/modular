//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/GraphCompiler.h"
#include "GenericML/GraphCompiler/Passes.h"

#include "GenericML/GraphCompiler/Conversion/MOToMOGG.h"
#include "GenericML/GraphCompiler/Conversion/MOToPrimitives.h"
#include "GenericML/GraphCompiler/MODialect/MODialect.h"
#include "GenericML/GraphCompiler/MODialect/MOTransforms.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGDialect.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGPasses.h"
#include "GenericML/MGPDialect/MGPTransforms.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"

#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

// All the KGEN dependent passes
#include "Cache/CacheDialect/CacheDialect.h"
#include "KGEN/InitAllDialects.h"
#include "Support/DebugInfoDialect/IR/DebugInfoDialect.h"
#include "Support/HLCFDialect/HLCFDialect.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

namespace M::MO {

void registerDialectsAndPasses(mlir::DialectRegistry &registry) {
  registry.insert<MO::MODialect, MOGG::MOGGDialect,
                  // The dialects needed for KGEN.
                  KGEN::KGENDialect, DebugInfo::DebugInfoDialect,
                  Cache::CacheDialect, HLCF::HLCFDialect,
                  mlir::index::IndexDialect, mlir::LLVM::LLVMDialect>();

  // The elaborator requires LLVM lowering to run the generated functions.
  mlir::registerLLVMDialectTranslation(registry);

  MO::registerMOToMOGG();
  MO::registerMOTransformsPasses();
  MGP::registerMOToPrimitives();
  MGP::registerMGPTransformsPasses();
  MOGG::registerPasses();

  mlir::registerCSEPass();
  mlir::registerSCCPPass();
  mlir::registerCanonicalizerPass();
}

void buildMoToMGPPipeline(mlir::PassManager &pm, const PassOptions &opts) {
  pm.addPass(mlir::createCSEPass());

  MO::MOToMOGGOptions moToMoggOpts;
  moToMoggOpts.useJit = opts.useJit;
  pm.addPass(MO::createMOToMOGG(moToMoggOpts));
  pm.addPass(mlir::createCSEPass());

  if (opts.useJit) {
    // If we are jitting then we need to compile the jit kernels as well.
    MOGG::JitCompileKernelsOptions jitOpts;
    jitOpts.useSearch = opts.useKgenSearch;
    pm.addPass(MOGG::createJitCompileKernels(jitOpts));

    if (opts.fuseJitKernels) {
      pm.addPass(MOGG::createFuseElementwise());
      pm.addPass(mlir::createCSEPass());
    }
  }

  pm.addPass(MGP::createMOToPrimitives(opts.moToPrimOpts));

  pm.addPass(mlir::createCSEPass());
  pm.addPass(MGP::createPeepholeTensorExtract());
}

} // namespace M::MO
