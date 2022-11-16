//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Provides an LSP server for all the modular dialects.
//
//===----------------------------------------------------------------------===//

#include "Faux/Dialect/FauxDialect.h"
#include "GenericML/MOPDialect/MOPDialect.h"
#include "GraphRT/BEFDialect/BEFDialect.h"
#include "GraphRT/GRTDialect/GRTDialect.h"
#include "KGEN/InitAllDialects.h"
#include "Support/CacheDialect/CacheDialect.h"
#include "Support/HLCFDialect/HLCFDialect.h"
#include "Support/MDialect/MDialect.h"
#include "TFLite/MLASPDialect/MLASPDialect.h"
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
#include "mlir/Dialect/SCF/IR/SCF.h"
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
      scf::SCFDialect,
      // Add the Modular Dialects.
      BEF::BEFDialect, Cache::CacheDialect, Faux::FauxDialect, GRT::GRTDialect,
      HLCF::HLCFDialect, MDialect, TFL::MLASPDialect, TFL::MTFLDialect,
      TFLP::TFLPDialect, MOP::MOPDialect>();
  // Add the KGEN Dialects.
  registerAllKGENDialects(registry);
  TF::registerAllDialects(registry);
  return failed(MlirLspServerMain(argc, argv, registry));
}
