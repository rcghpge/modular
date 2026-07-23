//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MOGGDIALECT_SUPPORT_MODULEHASHER_H
#define GENERICML_GRAPHCOMPILER_MOGGDIALECT_SUPPORT_MODULEHASHER_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"

namespace M::KGEN {
class GeneratorOp;
}

namespace M::MOGG {

void hashModuleContents(mlir::ModuleOp module);
void hashModuleContents(mlir::ModuleOp module, const mlir::SymbolTable &symtab);

/// Get the semantic hash attached to the generator by `hashModuleContents`.
std::string getSemanticHash(M::KGEN::GeneratorOp generator);

/// Computes the semantic hash for each operation in the given module and then
/// renames the `symbolsToRename` operations based on their semantic hash.
llvm::DenseMap<mlir::StringAttr, KGEN::GeneratorOp> applySemanticNaming(
    mlir::ModuleOp module, llvm::ArrayRef<mlir::StringAttr> symbolsToRename,
    const mlir::SymbolTable &symtab, llvm::StringRef namePrefix = "");

} // namespace M::MOGG

#endif // GENERICML_GRAPHCOMPILER_MOGGDIALECT_SUPPORT_MODULEHASHER_H
