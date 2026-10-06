//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Metal Rewrite DebugInfo pass - Transform LLVM DebugInfo to Metal intrinsics
//
//===----------------------------------------------------------------------===//

#include "MetalRewriteDebugInfo.h"

#include "llvm/IR/DebugInfo.h"
#include "llvm/Pass.h"
#include "llvm/Passes/PassBuilder.h"

#define KGEN_DEBUG_TYPE "metal-air-pass"

using namespace M::KGEN;

llvm::PreservedAnalyses
MetalRewriteDebugInfoPass::run(llvm::Module &module,
                               llvm::ModuleAnalysisManager &am) {
  // FIXME: Replace it with a proper rewrite of DebugInfo with metal-expected
  // intrinsics.
  if (llvm::StripDebugInfo(module))
    return llvm::PreservedAnalyses::none();
  return llvm::PreservedAnalyses::all();
}
