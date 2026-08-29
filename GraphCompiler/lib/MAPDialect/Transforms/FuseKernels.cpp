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
#include "GraphCompiler/MAPDialect/Support/ViewFuser.h"
#include "GraphCompiler/MAPDialect/Transforms/MAPTransforms.h"
#include "GraphCompiler/MODialect/MOOps.h"
#include "GraphCompiler/MOGGDialect/MOGGDialect.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

using namespace M;

namespace M::MAP {

#define GEN_PASS_DEF_FUSEKERNELS
#include "GraphCompiler/MAPDialect/Transforms/MAPTransforms.h.inc"

namespace {

struct FuseKernelsPass : public MAP::impl::FuseKernelsBase<FuseKernelsPass> {
  using Base::Base;

  void runOnOperation() override {
    MO::GraphOp graphOp = getOperation();
    // Epilogue-fuse first each round, prologue-fuse last: elementwise and
    // view fusion having already run this round is what guarantees every
    // producer `PrologueFuser` sees is the maximal elementwise/view chain.
    mlir::OpPassManager cleanupPm(MO::GraphOp::getOperationName());
    cleanupPm.addPass(mlir::createCanonicalizerPass());
    cleanupPm.addPass(mlir::createCSEPass());

    unsigned numFused;
    do {
      numFused = EpilogueFuser(graphOp).fuse() +
                 ElementwiseFuser(graphOp).fuse() + ViewFuser(graphOp).fuse() +
                 PrologueFuser(graphOp).fuse();
      numKernelsFused += numFused;
      // Canonicalize/CSE between rounds: a fuser can leave behind dead ops
      // or duplicate loads a later round's own eligibility/compatibility
      // checks may depend on seeing cleaned up.
      if (failed(runPipeline(cleanupPm, graphOp)))
        return signalPassFailure();
    } while (numFused > 0);
  }
};

} // namespace

} // namespace M::MAP
