//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Wraps structured MAP loops in `map.iter.opaque` regions so sequential setup
// can live outside the structured loop body. Scaffolding only.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/Conversion/MOToMAP/MOToMAP.h"
#include "GraphCompiler/MAPDialect/Transforms/MAPTransforms.h"

using namespace M;

namespace M::MAP {

#define GEN_PASS_DEF_OUTLINESTRUCTUREDKERNELBODIES
#include "GraphCompiler/MAPDialect/Transforms/MAPTransforms.h.inc"

namespace {

struct OutlineStructuredKernelBodiesPass
    : public MAP::impl::OutlineStructuredKernelBodiesBase<
          OutlineStructuredKernelBodiesPass> {
  using Base::Base;

  void runOnOperation() override {
    if (!MO::useAdvancedFusion())
      return;

    // TODO: wrap structured MAP loops (`map.iter.foreach`, future reduce /
    // rowwise) in `map.iter.opaque` regions. No-op for now.
  }
};

} // namespace

} // namespace M::MAP
