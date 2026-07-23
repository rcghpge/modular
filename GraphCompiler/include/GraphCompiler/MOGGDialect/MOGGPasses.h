//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MOGGDIALECT_TRANSFORMS_MOGGPASSES_H
#define GENERICML_GRAPHCOMPILER_MOGGDIALECT_TRANSFORMS_MOGGPASSES_H

#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassOptions.h"

#include "KGEN/HLCFDialect/HLCFDialect.h"
#include "KGEN/LITDialect/LITAttrs.h"
#include "KGEN/LITDialect/LITDialect.h"
#include "KGEN/POPDialect/POPDialect.h"
#include "KGEN/ToolCommon/CompilationOptions.h"
#include "Support/AlignedAlloc.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/LLVMForwardDecls.h"

//===----------------------------------------------------------------------===//
// Forward Declarations
//===----------------------------------------------------------------------===//

namespace M::MO {
class GraphOp;
} // namespace M::MO

namespace M::MOGG {
class MojoRegistry;
} // namespace M::MOGG

namespace M::MOGG {

/// How to build kernels. Release is optimized with no debug information,
/// relwithdebinfo is optimized with debug information, debug is not optimized
/// with debug information.
enum class KernelBuildType {
  kRelease,
  kRelWithDebInfo,
  kDebug,
};

/// Does the kernel build type have debuginfo.
inline bool hasDebuginfo(KernelBuildType kernelBuildType) {
  return kernelBuildType == KernelBuildType::kRelWithDebInfo ||
         kernelBuildType == KernelBuildType::kDebug;
}

/// Build type as a string.
inline std::string buildTypeString(KernelBuildType bt) {
  if (bt == KernelBuildType::kRelease)
    return "release";
  if (bt == KernelBuildType::kRelWithDebInfo)
    return "relwithdebinfo";
  if (bt == KernelBuildType::kDebug)
    return "debug";
  llvm_unreachable("All cases should be covered");
}

/// Provide one-time initialization of LLVM in a thread safe manner.
void initLLVM();

/// Probe the host (target triple, CPU, features, accelerator arch) and build
/// the `TargetInfoAttr` that feeds the cache key. Callers pass the result
/// explicitly so key computation performs no hardware probing of its own and
/// one resolution can be reused across key computations.
///
/// When `acceleratorArchOverride`/`acceleratorTripleOverride` describe a GPU
/// target, the accelerator arch is taken from them instead of probing the host
/// GPU. The cache key and codegen must resolve the target the same way, so any
/// caller that overrides the codegen target must pass the identical override
/// here.
ErrorOr<TargetInfoAttr>
getHostTargetInfoAttr(mlir::MLIRContext *ctx,
                      llvm::StringRef acceleratorArchOverride = "",
                      llvm::StringRef acceleratorTripleOverride = "");

KGEN::CompilationOptions
createCompilationOptions(TargetInfoAttr target, KernelBuildType kernelBuildType,
                         bool debugKGENStubs, StringRef saveTempsPrefix);

/// Write all compiler configuration information used by the graph compiler to
/// the ostream to be used as a cache key. `target` is typically the host
/// (see `getHostTargetInfoAttr`).
///
/// This includes information like the host CPU, the target accelerators,
/// the optimization level, and versioning information for the graph compiler.
ErrorOrSuccess
graphCompilerCacheKey(llvm::raw_ostream &keyBuf, TargetInfoAttr target,
                      KernelBuildType buildType = KernelBuildType::kRelease,
                      bool debug = false, llvm::StringRef saveTempsPrefix = "");

struct GraphCompilerCacheKey {
  mlir::MLIRContext *ctx;

  KernelBuildType buildType = KernelBuildType::kRelease;

  bool debug = false;

  std::string saveTempsPrefix = "";

  void print(llvm::raw_ostream &os);
};

//===----------------------------------------------------------------------===//
// Generated Pass Classes and Registration
//===----------------------------------------------------------------------===//

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "GraphCompiler/MOGGDialect/MOGGPasses.h.inc"

void mergeIdenticalKernelStubs(mlir::ModuleOp module);

/// Lift inline `index.constant` ops feeding `mgp.buffer.get_cached`'s slot
/// operand from kernel-stub bodies into new kernel arguments. See GEX-3685.
void liftBufferCacheSlots(mlir::ModuleOp module);

/// Perform one iteration of reshape-into-epilogue fusion on the graph, fusing
/// SplitDim/MergeDim/AddSingletonDim ops into kernel output lambdas.
/// Returns the number of views fused. Shared by FuseReshapesIntoEpilogues and
/// FuseEpilogues.
size_t fuseReshapesIntoEpiloguesStep(MO::GraphOp graph,
                                     const MojoRegistry &registry);

} // namespace M::MOGG

#endif // GENERICML_GRAPHCOMPILER_MOGGDIALECT_TRANSFORMS_MOGGPASSES_H
