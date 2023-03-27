//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_KGENSTUBBUILDER_STUBCOMPILER_H
#define GENERICML_GRAPHCOMPILER_KGENSTUBBUILDER_STUBCOMPILER_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"

#include "GenericML/GraphCompiler/MODialect/MOOps.h"
#include "Support/ForwardDecls.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/LLVMForwardDecls.h"

namespace M::MOGG::Stub {

/// Helpers to compile the KGEN stubs to binary. This splits the compile to
/// binary stuff from the KGEN stub creation process.
class StubCompiler {
public:
  StubCompiler(mlir::ModuleOp kgenStubs, MO::GraphOp graph, bool shouldSearch,
               std::string saveTempsPrefix)
      : ctx(graph.getContext()), kgenModule(kgenStubs), loc(graph.getLoc()),
        moggBuilder(&graph.getBody().front(), graph.getBody().front().begin()),
        saveTempsPrefix(saveTempsPrefix), shouldSearch(shouldSearch) {}

  /// Compile to binary object. Tell the jit all of the given symbols as symbols
  /// within it that can be executed at runtime.
  ErrorOrSuccess compile(mlir::ArrayAttr combinedSymbolName);

private:
  /// Run the KGEN elaboration passes on the module.
  LogicalResult elaborateKGEN();

  /// Lower to an binary object.
  ErrorOrSuccess compileToLLVM(mlir::ArrayAttr combinedSymbolName);

  MLIRContext *ctx;

  /// The KGEN module with all the stubs in it.
  mlir::ModuleOp kgenModule;

  /// Location to use.
  mlir::Location loc;

  /// The "mogg" side builder. I.E for adding ops to the main MOGG graph rather
  /// than the kgen stub.
  mlir::OpBuilder moggBuilder;

  /// If set to non empty string emit temporary IR and assembly outputs to the
  /// given prefix. Currently emits KGEN before and after elaboration, llvm
  /// before and after optimization, and the final assembly.
  std::string saveTempsPrefix;

  /// Should tell KGEN to search.
  bool shouldSearch;
};

} // namespace M::MOGG::Stub

#endif // GENERICML_GRAPHCOMPILER_KGENSTUBBUILDER_STUBCOMPILER_H
