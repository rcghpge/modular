//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MODialect/MODevices.h"
#include "GraphCompiler/MODialect/MOOps.h"
#include "GraphCompiler/MOGGDialect/Analysis/TraitContractDerivation.h"
#include "GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GraphCompiler/MOGGDialect/MOGGPasses.h"
#include "GraphCompiler/MOGGDialect/Support/MOGGTypeTraits.h"

using namespace M;

namespace M::MOGG {

#define GEN_PASS_DEF_FUSECONSTANTS
#include "GraphCompiler/MOGGDialect/MOGGPasses.h.inc"

namespace {

// The max number of elements in a constant that we will allow to be fused
// into the body of a kernel.
constexpr int64_t kMaxConstantSize = 2048;

/// Try to get the original constant op that produced `val`, with optional
/// rebind (i.e., reinterpret cast) in the way.
static MO::ConstantOp tryGetConstantOp(Value val) {
  if (auto rebindOp = val.getDefiningOp<MO::RebindOp>())
    val = rebindOp.getInput();

  return val.getDefiningOp<MO::ConstantOp>();
}

/// Return the fusable constant producing the `index`-th operand of `kernel`,
/// or null when the operand cannot be folded into the kernel body.
static MO::ConstantOp getFusableConstant(MOGG::KernelOp kernel, int32_t index) {
  // Don't fuse constants into non-host (device) kernels: device constant
  // lowering is still incorrect (removing this guard crashes real GPU kernels
  // with illegal memory accesses). See
  // https://linear.app/modularml/issue/GEX-3839.
  if (!MO::isOnHost(kernel))
    return {};

  MO::ConstantOp cst = tryGetConstantOp(kernel.getOperand(index));
  if (!cst || !MO::isOnHost(cst))
    return {};

  ElementsAttr elems = cst.getValueAttr();
  if (elems.getNumElements() > kMaxConstantSize)
    return {};

  Type dtype = elems.getElementType();
  if (!MOGGTypeTraits::isSupportedMOGGType(dtype))
    return {};

  if (!isa<M::ArrayElementsAttr, DenseResourceElementsAttr>(elems))
    return {};

  return cst;
}

/// Fold `cst`, the producer of the `idx`-th operand of `kernel`, into the
/// kernel body.
static void fuseInputAtIndex(MOGG::KernelOp kernel, int32_t idx,
                             MO::ConstantOp cst) {
  Block *block = kernel.getBody();

  mlir::OpBuilder builder{kernel->getContext()};
  builder.setInsertionPointToStart(block);

  MO::ConstantOp newOp = cst.clone();
  builder.insert(newOp);
  block->getArgument(idx).replaceAllUsesWith(newOp.getResult());
  kernel.eraseInputsAt(idx);

  if (cst->use_empty())
    cst->erase();
}

/// Mutable/output contracts need writable backing; no-role shape arguments
/// keep their boundary path. Per-use contract: see
/// `KernelIOContractDeriver::deriveUse`.
static FailureOr<bool>
allUsesAcceptFoldedConstant(MOGG::KernelOp kernel, int32_t index,
                            MOGG::KernelIOContractDeriver &deriver) {
  for (OpOperand &use : kernel.getBody()->getArgument(index).getUses()) {
    FailureOr<MOGG::TensorArgContractAttr> contract = deriver.deriveUse(use);
    if (failed(contract))
      return failure();
    if (*contract && contract->getRole() != MOGG::TensorArgRoleKind::input)
      return false;
  }
  return true;
}

static LogicalResult fuseConstant(MOGG::KernelOp kernel,
                                  MOGG::KernelIOContractDeriver &deriver) {
  int32_t preFuseNumOperands = kernel.getNumOperands();
  for (int32_t index = preFuseNumOperands - 1; index >= 0; --index) {
    MO::ConstantOp cst = getFusableConstant(kernel, index);
    if (!cst)
      continue;

    FailureOr<bool> foldable =
        allUsesAcceptFoldedConstant(kernel, index, deriver);
    if (failed(foldable))
      return failure();
    if (!*foldable)
      continue;

    fuseInputAtIndex(kernel, index, cst);
  }
  return success();
}

struct FuseConstants : public impl::FuseConstantsBase<FuseConstants> {
  using FuseConstantsBase::FuseConstantsBase;

  void runOnOperation() override {
    MO::GraphOp graph = getOperation();
    MOGG::KernelIOContractDeriver deriver(
        graph->getParentOfType<mlir::ModuleOp>());
    for (MOGG::KernelOp kernel : graph.getOps<MOGG::KernelOp>())
      if (failed(fuseConstant(kernel, deriver)))
        return signalPassFailure();
  }
};

} // namespace

} // namespace M::MOGG
