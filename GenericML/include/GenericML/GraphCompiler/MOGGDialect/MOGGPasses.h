//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MOGGDIALECT_TRANSFORMS_MOGGPASSES_H
#define GENERICML_GRAPHCOMPILER_MOGGDIALECT_TRANSFORMS_MOGGPASSES_H

#include "Support/AlignedAlloc.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/LLVMForwardDecls.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassOptions.h"

// Need to include the dialects we use, which includes many KGEN ones.
#include "Cache/CacheDialect/CacheDialect.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGOps.h"
#include "KGEN/HLCFDialect/HLCFDialect.h"
#include "KGEN/InitAllDialects.h"
#include "Support/DebugInfoDialect/IR/DebugInfoDialect.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

//===----------------------------------------------------------------------===//
// Forward Declarations
//===----------------------------------------------------------------------===//

namespace M::MOGG {

// Return the mojo registry, adding it if needed.
FailureOr<MojoRegisterOp>
getOrLoadMojoLibrary(ModuleOp theModule, const std::string &extraLibPaths);

// Return true if the MO op has the given parameter.
bool kernelHasInputParameter(MOGG::MojoRegisterOp opRegister,
                             ModuleOp kgenModule, StringRef moOpName,
                             StringRef parameterName);

// Returns true if the kernel is an elementwise kernel.
bool kernelIsElementwise(MOGG::MojoRegisterOp opRegister, ModuleOp kgenModule,
                         StringRef moOpName);

// Returns true if the kernel is expecting to take indices as last arg.
bool kernelTakesIndices(MOGG::MojoRegisterOp opRegister, ModuleOp kgenModule,
                        StringRef moOpName);

// Returns true if the kernel is a view operation.
bool kernelIsViewOp(MOGG::MojoRegisterOp opRegister, ModuleOp kgenModule,
                    StringRef moOpName);

// Returns the KGEN signature of the kernel.
KGEN::SignatureType kernelGetSignature(MOGG::MojoRegisterOp opRegister,
                                       ModuleOp kgenModule, StringRef moOpName);

// Return the KGEN module containing the MOGG bindings.
ErrorOr<OwningOpRef<mlir::ModuleOp>>
loadMojoBindings(MLIRContext *ctx, const std::string &extraLibPaths);

//===----------------------------------------------------------------------===//
// Generated Pass Classes and Registration
//===----------------------------------------------------------------------===//

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "GenericML/GraphCompiler/MOGGDialect/MOGGPasses.h.inc"

} // namespace M::MOGG

#endif // GENERICML_GRAPHCOMPILER_MOGGDIALECT_TRANSFORMS_MOGGPASSES_H
