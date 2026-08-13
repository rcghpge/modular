//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Fuses MAP kernels by inlining producers into consumers.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MAPDialect/Support/ElementwiseFuser.h"
#include "GraphCompiler/MAPDialect/Support/EpilogueFuser.h"
#include "GraphCompiler/MAPDialect/Support/PrologueFuser.h"
#include "GraphCompiler/MAPDialect/Transforms/MAPTransforms.h"
#include "GraphCompiler/MODialect/MOOps.h"
#include "GraphCompiler/MOGGDialect/MOGGDialect.h"

using namespace M;

namespace M::MAP {

#define GEN_PASS_DEF_FUSEKERNELS
#include "GraphCompiler/MAPDialect/Transforms/MAPTransforms.h.inc"

namespace {

struct FuseKernelsPass : public MAP::impl::FuseKernelsBase<FuseKernelsPass> {
  using Base::Base;

  void runOnOperation() override {
    MO::GraphOp graphOp = getOperation();
    // Epilogue-fuse first: folding an elementwise consumer into a
    // producer's epilogue preserves strictly more fusion opportunity than
    // running elementwise fusion first would (an elementwise op already
    // absorbed into a chain is no longer a single op the epilogue fuser
    // can fold in). Prologue-fuse last: elementwise fusion having already
    // run is what guarantees every producer `PrologueFuser` sees is the
    // maximal elementwise chain, which is what lets it be a single pass.
    EpilogueFuser(graphOp).fuse();
    ElementwiseFuser(graphOp).fuse();
    PrologueFuser(graphOp).fuse();
  }
};

} // namespace

} // namespace M::MAP
