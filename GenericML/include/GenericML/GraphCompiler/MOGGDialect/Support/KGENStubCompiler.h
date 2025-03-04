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

namespace M::MOGG {

CompileDiagnosticOr<MOGG::LoadCompiledKernelOp>
compileStubs(OwningOpRef<ModuleOp> kgenStubs, Operation *graph,
             std::string saveTempsPrefix, KernelBuildType kernelBuildType,
             mlir::ArrayAttr combinedSymbolName, bool enableTiming = false,
             bool debugKGENStubs = false);

} // namespace M::MOGG

#endif // GENERICML_GRAPHCOMPILER_MOGGSUPPORT_KGENSTUBCOMPILER_H
