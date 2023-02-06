//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_KGENSTUBBUILDER_STUBCOMPILER_H
#define GENERICML_KGENSTUBBUILDER_STUBCOMPILER_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"

#include "GenericML/MODialect/MOOps.h"
#include "Support/ForwardDecls.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/LLVMForwardDecls.h"

namespace M::MOGG::Stub {

/// Helpers to compile the KGEN stubs to binary. This splits the compile to
/// binary stuff from the KGEN stub creation process.
class StubCompiler {
public:
  StubCompiler(mlir::ModuleOp kgenStubs, MO::GraphOp graph, bool shouldSearch)
      : ctx(graph.getContext()), kgenModule(kgenStubs), loc(graph.getLoc()),
        moggBuilder(&graph.getBody().front(), graph.getBody().front().begin()),
        shouldSearch(shouldSearch) {}

  /// Compile to binary and create an operation to execute this operation.
  ErrorOrSuccess compile(StringRef symbolName);

private:
  /// Run the KGEN elaboration passes on the module.
  LogicalResult elaborateKGEN();

  /// Lower to an binary object.
  ErrorOrSuccess compileToLLVM(StringRef symbolName);

  MLIRContext *ctx;

  /// The KGEN module with all the stubs in it.
  mlir::ModuleOp kgenModule;

  /// Location to use.
  mlir::Location loc;

  /// The "mogg" side builder. I.E for adding ops to the main MOGG graph rather
  /// than the kgen stub.
  mlir::OpBuilder moggBuilder;

  // Should tell KGEN to search.
  bool shouldSearch;
};

} // namespace M::MOGG::Stub

#endif // GENERICML_KGENSTUBBUILDER_STUBCOMPILER_H