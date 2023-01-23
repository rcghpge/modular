//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/Conversion/MOToMOGG.h"
#include "GenericML/Conversion/MOToPrimitives.h"
#include "GenericML/MODialect/MODialect.h"
#include "GenericML/MODialect/MOTransforms.h"
#include "GenericML/MOGGDialect/MOGGDialect.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"

#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

// All the KGEN dependent passes
#include "Cache/CacheDialect/CacheDialect.h"
#include "KGEN/InitAllDialects.h"
#include "Support/DebugInfoDialect/IR/DebugInfoDialect.h"
#include "Support/HLCFDialect/HLCFDialect.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

using namespace M;

int main(int argc, char **argv) {
  DialectRegistry registry;
  registry.insert<MO::MODialect, MOGG::MOGGDialect,

                  // The dialects needed for KGEN.
                  DebugInfo::DebugInfoDialect, Cache::CacheDialect,
                  HLCF::HLCFDialect, mlir::index::IndexDialect,
                  mlir::LLVM::LLVMDialect, mlir::scf::SCFDialect>();

  // The elaborator requires LLVM lowering to run the generated functions.
  mlir::registerLLVMDialectTranslation(registry);

  MO::registerMOToMOGG();
  MO::registerMOTransformsPasses();
  MGP::registerMOToPrimitives();

  mlir::registerCSEPass();
  mlir::registerSCCPPass();
  mlir::registerCanonicalizerPass();

  return failed(
      mlir::MlirOptMain(argc, argv, "MO dialect test driver", registry));
}
