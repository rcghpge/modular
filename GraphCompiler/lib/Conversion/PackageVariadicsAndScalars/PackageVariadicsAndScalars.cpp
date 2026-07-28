//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Materializes tuple and scalar operands on `mogg` calls. Scaffolding only.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/Conversion/PackageVariadicsAndScalars/PackageVariadicsAndScalars.h"
#include "GraphCompiler/Conversion/MOToMAP/MOToMAP.h"
#include "GraphCompiler/MOGGDialect/MOGGDialect.h"

using namespace M;

namespace M::MOGG {

#define GEN_PASS_DEF_PACKAGEVARIADICSANDSCALARS
#include "GraphCompiler/Conversion/PackageVariadicsAndScalars/PackageVariadicsAndScalars.h.inc"

namespace {

struct PackageVariadicsAndScalarsPass
    : public MOGG::impl::PackageVariadicsAndScalarsBase<
          PackageVariadicsAndScalarsPass> {
  using Base::Base;

  void runOnOperation() override {
    if (!MO::useAdvancedFusion())
      return;

    // TODO: inspect `mogg` call signatures and insert tuple/scalar conversion
    // ops. No-op for now.
  }
};

} // namespace

} // namespace M::MOGG
