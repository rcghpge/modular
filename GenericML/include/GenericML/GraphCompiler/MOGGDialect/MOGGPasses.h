//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MOGGDIALECT_TRANSFORMS_MOGGPASSES_H
#define GENERICML_GRAPHCOMPILER_MOGGDIALECT_TRANSFORMS_MOGGPASSES_H

#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassOptions.h"

#include "KGEN/CustomDialect/CustomDialect.h"
#include "KGEN/HLCFDialect/HLCFDialect.h"
#include "KGEN/LITDialect/LITAttrs.h"
#include "KGEN/LITDialect/LITDialect.h"
#include "KGEN/POPDialect/POPDialect.h"
#include "Support/AlignedAlloc.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/LLVMForwardDecls.h"

//===----------------------------------------------------------------------===//
// Forward Declarations
//===----------------------------------------------------------------------===//

namespace M::MOGG {

/// How to build kernels. Release is optimized with no debug information,
/// relwithdebinfo is optimimized with debug information, debug is not optimized
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

//===----------------------------------------------------------------------===//
// Generated Pass Classes and Registration
//===----------------------------------------------------------------------===//

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "GenericML/GraphCompiler/MOGGDialect/MOGGPasses.h.inc"

} // namespace M::MOGG

#endif // GENERICML_GRAPHCOMPILER_MOGGDIALECT_TRANSFORMS_MOGGPASSES_H
