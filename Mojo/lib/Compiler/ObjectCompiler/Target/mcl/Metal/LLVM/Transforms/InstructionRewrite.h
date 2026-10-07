//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// LLVM IR Instruction Selection Pass performs LLVM IR to LLVM IR transformation
// by replacing instructions or intrinsics that are not supported by the target
// to instructions or target-specific intrinsics.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_COMPILER_LLVMIR_TRANSFORMS_INSTRUCTIONREWRITE_H
#define KGEN_COMPILER_LLVMIR_TRANSFORMS_INSTRUCTIONREWRITE_H

#include "llvm/IR/PassManager.h"

namespace M::KGEN {

/// New pass manager pass that transforms LLVM IR for backend compilation
// that takes older version of LLVM IR.
///
/// This pass:
/// - Transforms llvm.lifetime related intrinsics
///
class InstructionRewritePass
    : public llvm::RequiredPassInfoMixin<InstructionRewritePass> {
public:
  llvm::PreservedAnalyses run(llvm::Function &func,
                              llvm::FunctionAnalysisManager &fam);
};

} // namespace M::KGEN

#endif // KGEN_COMPILER_LLVMIR_TRANSFORMS_INSTRUCTIONREWRITE_H
