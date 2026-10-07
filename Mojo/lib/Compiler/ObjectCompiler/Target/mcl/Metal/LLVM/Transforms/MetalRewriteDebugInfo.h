//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Metal Rewrite DebugInfo pass - Transform LLVM DebugInfo to Metal intrinsics
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_COMPILER_LLVMIR_TRANSFORMS_METALREWRITEDEBUGINFO_H
#define KGEN_COMPILER_LLVMIR_TRANSFORMS_METALREWRITEDEBUGINFO_H

#include "llvm/IR/PassManager.h"

namespace llvm {
class Module;
class ModulePass;
} // namespace llvm

namespace M::KGEN {

/// The special LLVM xform pass that converts DebugInfo to special intrinsics
/// that are used by Instruments to debug Metal kernel.
///
/// TODO: Implement conversion to these intrinsics. For now DebugInfo is just
/// stripped as it's not used by Metal and causes crashes.
class MetalRewriteDebugInfoPass
    : public llvm::RequiredPassInfoMixin<MetalRewriteDebugInfoPass> {
public:
  llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &AM);
};

} // namespace M::KGEN

#endif // KGEN_COMPILER_LLVMIR_TRANSFORMS_METALREWRITEDEBUGINFO_H
