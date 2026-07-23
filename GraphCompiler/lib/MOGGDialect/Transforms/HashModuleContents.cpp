//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GraphCompiler/MOGGDialect/MOGGPasses.h"

#include "GraphCompiler/MOGGDialect/Support/ModuleHasher.h"

#include "mlir/Analysis/SymbolTableAnalysis.h"

using namespace M;

namespace M::MOGG {

#define GEN_PASS_DEF_HASHMODULECONTENTS
#include "GraphCompiler/MOGGDialect/MOGGPasses.h.inc"

namespace {

struct HashModuleContents
    : public impl::HashModuleContentsBase<HashModuleContents> {
  using HashModuleContentsBase::HashModuleContentsBase;

  void runOnOperation() override {
    auto op = getOperation();
    mlir::SymbolTableAnalysis &analysis =
        getAnalysis<mlir::SymbolTableAnalysis>();
    SymbolTable &symtab = analysis.getTopLevelSymbolTable();
    hashModuleContents(op, symtab);
  }
};

} // namespace

} // namespace M::MOGG
