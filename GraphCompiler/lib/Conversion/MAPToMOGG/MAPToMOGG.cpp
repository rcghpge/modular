//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Lowers `map` dialect kernels into `mogg` codegen ops. Scaffolding only.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/Conversion/MAPToMOGG/MAPToMOGG.h"
#include "GraphCompiler/Conversion/MOToMAP/MOToMAP.h"
#include "GraphCompiler/MAPDialect/Dialect/MAPDialect.h"
#include "GraphCompiler/MOGGDialect/MOGGDialect.h"

using namespace M;

namespace M::MAP {

#define GEN_PASS_DEF_MAPTOMOGG
#include "GraphCompiler/Conversion/MAPToMOGG/MAPToMOGG.h.inc"

namespace {

struct MAPToMOGGPass : public MAP::impl::MAPToMOGGBase<MAPToMOGGPass> {
  using Base::Base;

  void runOnOperation() override {
    if (!MO::useAdvancedFusion())
      return;

    // TODO: lower `map` kernels into `mogg` codegen ops. No-op for now.
  }
};

} // namespace

} // namespace M::MAP
