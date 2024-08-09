//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MOGGSUPPORT_KGENSTUBCOMPILER_H
#define GENERICML_GRAPHCOMPILER_MOGGSUPPORT_KGENSTUBCOMPILER_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"

#include "GenericML/GraphCompiler/MODialect/MOOps.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGPasses.h"
#include "Support/ForwardDecls.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/LLVMForwardDecls.h"

namespace M::MOGG::Stub {

/// Helpers to compile the KGEN stubs to binary. This splits the compile to
/// binary stuff from the KGEN stub creation process.
class StubCompiler {
public:
  StubCompiler(mlir::ModuleOp kgenStubs, Operation *graph, bool shouldSearch,
               std::string saveTempsPrefix, KernelBuildType kernelBuildType)
      : ctx(graph->getContext()), kgenModule(kgenStubs), loc(graph->getLoc()),
        saveTempsPrefix(std::move(saveTempsPrefix)), shouldSearch(shouldSearch),
        kernelBuildType(kernelBuildType) {}

  /// Compile to binary object. Tell the jit all of the given symbols as symbols
  /// within it that can be executed at runtime.
  ErrorOr<MOGG::LoadCompiledKernelOp>
  compile(mlir::ArrayAttr combinedSymbolName, bool enableTiming = false,
          bool debugKGENStubs = false);

  /// Run the KGEN elaboration passes on the module.
  LogicalResult elaborateKGEN(bool enableTiming = false, bool superOpt = false,
                              bool debugKGENStubs = false);

private:
  /// Lower to an binary object.
  ErrorOr<MOGG::LoadCompiledKernelOp>
  compileToLLVM(mlir::ArrayAttr combinedSymbolName, bool enableTiming,
                bool debugKGENStubs);

  MLIRContext *ctx;

  /// The KGEN module with all the stubs in it.
  mlir::ModuleOp kgenModule;

  /// Location to use.
  mlir::Location loc;

  /// If set to non empty string emit temporary IR and assembly outputs to the
  /// given prefix. Currently emits KGEN before and after elaboration, llvm
  /// before and after optimization, and the final assembly.
  std::string saveTempsPrefix;

  /// Should tell KGEN to search.
  bool shouldSearch;

  /// How should kernels be compiled. If debuginfo is included in kernels then
  /// it will be added to the stubs.
  KernelBuildType kernelBuildType;
};

} // namespace M::MOGG::Stub

#endif // GENERICML_GRAPHCOMPILER_MOGGSUPPORT_KGENSTUBCOMPILER_H
