//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Lowers MO ops into `map` / `imap` kernels. Scaffolding only.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/GraphCompiler/Conversion/MOToMAP.h"
#include "GraphCompiler/GraphCompiler/MAPDialect/Dialect/MAPDialect.h"
#include "GraphCompiler/GraphCompiler/MODialect/MODialect.h"

using namespace M;
using namespace mlir;

namespace M::MO {

#define GEN_PASS_DEF_MOTOMAP
#include "GraphCompiler/GraphCompiler/Conversion/MOToMAP.h.inc"

namespace {

struct MOToMAPPass : public M::MO::impl::MOToMAPBase<MOToMAPPass> {
  using Base::Base;

  void runOnOperation() override {
    if (!useAdvancedFusion())
      return;

    // TODO: lower MO ops into map/imap kernels. No-op for now.
  }
};

} // namespace

} // namespace M::MO
