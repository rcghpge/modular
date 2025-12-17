//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Provides an LSP server for all the modular dialects.
//
//===----------------------------------------------------------------------===//

#include "Cache/CachedTransform.h"
#include "GenericML/GraphCompiler/MODialect/MODialect.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGDialect.h"
#include "GenericML/GraphCompiler/MOSHDialect/MOSHDialect.h"
#include "GenericML/GraphCompiler/RMODialect/RMODialect.h"
#include "GenericML/MGPDialect/MGPDialect.h"
#include "KGEN/ToolCommon/InitAllDialects.h"
#include "MLRT/GraphRT/GRTDialect/GRTDialect.h"
#include "MLRT/GraphRT/MEFDialect/MEFDialect.h"
#include "Support/DebugInfoDialect/IR/DebugInfoDialect.h"
#include "Support/MDialect/MDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MLProgram/IR/MLProgram.h"
#include "mlir/Dialect/Quant/IR/Quant.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Tools/mlir-lsp-server/MlirLspServerMain.h"

using namespace mlir;
using namespace M;

int main(int argc, char **argv) {
  DialectRegistry registry;
  registry.insert<
      // Add the MLIR Dialects.
      arith::ArithDialect, cf::ControlFlowDialect, func::FuncDialect,
      index::IndexDialect, LLVM::LLVMDialect, quant::QuantDialect,
      mlir::ub::UBDialect,
      // Add the Modular Dialects.
      MEF::MEFDialect, DebugInfo::DebugInfoDialect, GRT::GRTDialect, MDialect,
      MGP::MGPDialect, RMO::RMODialect, MOSH::MOSHDialect, MO::MODialect,
      MOGG::MOGGDialect>();
  // Add the KGEN Dialects.
  registerAllKGENDialects(registry);
  return failed(MlirLspServerMain(argc, argv, registry));
}
