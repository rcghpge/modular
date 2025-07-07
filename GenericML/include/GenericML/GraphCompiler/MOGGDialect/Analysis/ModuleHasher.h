//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MOGGDIALECT_ANALYSIS_MODULEHASHER_H
#define GENERICML_GRAPHCOMPILER_MOGGDIALECT_ANALYSIS_MODULEHASHER_H

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

} // namespace M::MOGG

#endif // GENERICML_GRAPHCOMPILER_MOGGDIALECT_ANALYSIS_MODULEHASHER_H
