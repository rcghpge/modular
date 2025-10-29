//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MOGGSUPPORT_KGENSTUBCOMPILER_H
#define GENERICML_GRAPHCOMPILER_MOGGSUPPORT_KGENSTUBCOMPILER_H

#include "mlir/IR/BuiltinAttributes.h"

#include "GenericML/GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGPasses.h"
#include "Support/Buffer.h"
#include "Support/LLVMForwardDecls.h"

namespace M::MOGG {

using CompiledKernels = SmallVector<OwningOpRef<MOGG::LoadCompiledKernelOp>>;

/// Compile to binary object. Tell the jit all of the given symbols as symbols
/// within it that can be executed at runtime.
CompileDiagnosticOr<SmallVector<MOGG::LoadCompiledKernelOp>> compileStubs(
    OwningOpRef<ModuleOp> kgenStubs, Operation *graph,
    std::string saveTempsPrefix, KernelBuildType kernelBuildType,
    const llvm::MapVector<mlir::StringAttr, KGEN::GeneratorOp> &symbolsAdded,
    bool enableTiming = false, bool debugKGENStubs = false);

DenseResourceElementsAttr createArchiveAttr(MLIRContext *ctx, BufferRef buffer,
                                            StringRef name);
ErrorOr<TargetInfoAttr> getTargetInfo(MLIRContext *ctx);

} // namespace M::MOGG

#endif // GENERICML_GRAPHCOMPILER_MOGGSUPPORT_KGENSTUBCOMPILER_H
