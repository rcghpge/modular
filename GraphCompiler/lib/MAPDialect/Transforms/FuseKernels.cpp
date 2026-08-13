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
    // Epilogue-fuse first: folding an elementwise consumer into a producer's
    // epilogue preserves strictly more fusion opportunity than running
    // elementwise fusion first would (an elementwise op already absorbed into
    // a chain is no longer a single op the epilogue fuser can fold in).
    // Prologue fusion has no place here yet -- MoToMAP does not lower
    // input-side (prologue) fusion at all.
    EpilogueFuser(graphOp).fuse();
    ElementwiseFuser(graphOp).fuse();
  }
};

} // namespace

} // namespace M::MAP
