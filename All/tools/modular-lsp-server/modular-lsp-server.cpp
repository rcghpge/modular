//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Provides an LSP server for all the modular dialects.
//
//===----------------------------------------------------------------------===//

#include "Cache/CacheDialect/CacheDialect.h"
#include "Faux/Dialect/FauxDialect.h"
#include "GenericML/GraphCompiler/MODialect/MODialect.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGDialect.h"
#include "GenericML/MGPDialect/MGPDialect.h"
#include "GenericML/MLASPDialect/MLASPDialect.h"
#include "GraphRT/GRTDialect/GRTDialect.h"
#include "GraphRT/MEFDialect/MEFDialect.h"
#include "KGEN/HLCFDialect/HLCFDialect.h"
#include "KGEN/InitAllDialects.h"
#include "Support/DebugInfoDialect/IR/DebugInfoDialect.h"
#include "Support/MDialect/MDialect.h"
#include "TFLite/MTFLDialect/MTFLDialect.h"
#include "TFLite/TFLPDialect/TFLPDialect.h"
#include "TensorFlow/InitAllDialects.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MLProgram/IR/MLProgram.h"
#include "mlir/Dialect/Quant/QuantOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Tools/mlir-lsp-server/MlirLspServerMain.h"

using namespace mlir;
using namespace M;

int main(int argc, char **argv) {
  DialectRegistry registry;
  registry.insert<
      // Add the MLIR Dialects.
      arith::ArithDialect, cf::ControlFlowDialect, func::FuncDialect,
      index::IndexDialect, LLVM::LLVMDialect, quant::QuantizationDialect,
      // Add the Modular Dialects.
      MEF::MEFDialect, Cache::CacheDialect, DebugInfo::DebugInfoDialect,
      Faux::FauxDialect, GRT::GRTDialect, HLCF::HLCFDialect, MDialect,
      MGP::MGPDialect, MLASP::MLASPDialect, MO::MODialect, MOGG::MOGGDialect,
      TFL::MTFLDialect, TFLP::TFLPDialect>();
  // Add the KGEN Dialects.
  registerAllKGENDialects(registry);
  TF::registerAllDialects(registry);
  return failed(MlirLspServerMain(argc, argv, registry));
}
