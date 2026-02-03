//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGPasses.h"
#include "GenericML/GraphCompiler/MOGGDialect/Support/ParametersUtils.h"

using namespace M;

namespace M::MOGG {

#define GEN_PASS_DEF_NORMALIZEALLNAMES
#include "GenericML/GraphCompiler/MOGGDialect/MOGGPasses.h.inc"

namespace {

struct NormalizeAllNames
    : public impl::NormalizeAllNamesBase<NormalizeAllNames> {
  using NormalizeAllNamesBase::NormalizeAllNamesBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    normalizeAllNames(module);
  }
};

} // namespace

} // namespace M::MOGG
