//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/MOGGDialect/Support/FusionUtils.h"
#include "GenericML/GraphCompiler/MODialect/MOAttrs.h"
#include "GenericML/GraphCompiler/MODialect/MOInterfaces.h"
#include "GenericML/GraphCompiler/MODialect/MOOps.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GenericML/GraphCompiler/MOGGDialect/MojoParameters.h"
#include "GenericML/GraphCompiler/MOGGDialect/Support/MOGGTypeTraits.h"
#include "GenericML/GraphCompiler/MOGGDialect/Support/MojoTargetFunctions.h"
#include "GenericML/Support/DataflowUtils.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/LITDialect/LITUtils.h"
#include "KGEN/MOGGPreElab/MojoKernelAdaptor.h"
#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "Support/AssertStream.h"
#include "llvm/ADT/StringSet.h"

namespace M::MOGG {

// FusionInfo helper
FusionLeafAttr createFusionLeaf(Operation *op) {
  std::string name;
  if (auto shapeFuncOp = dyn_cast<MO::InvokeShapeFuncOp>(op)) {
    name = "shapeof(";
    name += shapeFuncOp.getShapeFuncName();
    name += ")";
  } else if (auto layoutTransformOp = dyn_cast<MO::LayoutTransformOp>(op)) {
    name = layoutTransformOp->getName().getIdentifier().getValue();
    name += "_";
    name += layoutTransformOp.getInput()
                .getType()
                .getLayoutAttr()
                .getFormat()
                .getValue();
    name += "_to_";
    name += layoutTransformOp.getResult()
                .getType()
                .getLayoutAttr()
                .getFormat()
                .getValue();
  } else if (auto customOp = dyn_cast<MO::CustomOp>(op)) {
    name = "custom__";
    name += customOp.getSymbol();
  } else {
    name = op->getName().getIdentifier().getValue();
  }

  uint32_t lineNumber = 0;
  if (auto fileLineLoc = dyn_cast<mlir::FileLineColLoc>(op->getLoc()))
    if (fileLineLoc.getLine() > 0)
      lineNumber = fileLineLoc.getLine();

  auto ctx = op->getContext();
  return FusionLeafAttr::get(ctx, StringAttr::get(ctx, name), lineNumber);
}

// The max number of elements in a constant that we will allow to be fused into
// the body of a kernel.
constexpr int64_t kMaxConstantSize = 2048;

/// Try to get the original constant op that produced `val`, with optional
/// rebind (i.e., reinterpret cast) in the way.
static MO::ConstantOp tryGetConstantOp(Value val) {
  // Look through potential rebind op
  if (auto rebindOp = val.getDefiningOp<MO::RebindOp>())
    val = rebindOp.getInput();

  return val.getDefiningOp<MO::ConstantOp>();
}

/// Check whether the `index`-th operand of `kernel` is a fusable constant op.
static bool canFuseOperand(MOGG::KernelOp kernel, int32_t index) {
  // HACK HACK HACK https://github.com/modularml/modular/issues/22959
  // Don't fuse constant unless on the host. TEMPORARY until constant
  // lowering fixed.
  if (!MO::isOnHost(kernel))
    return false;

  MO::ConstantOp cst = tryGetConstantOp(kernel.getOperand(index));
  if (!cst)
    return false;

  if (!MO::isOnHost(cst))
    return false;

  // Check it is a small constant with a attribute we understand.
  ElementsAttr elems = cst.getValueAttr();
  if (elems.getNumElements() > kMaxConstantSize)
    return false;

  Type dtype = elems.getElementType();
  if (!MOGGTypeTraits::isSupportedMOGGType(dtype))
    return false;

  if (!isa<M::ArrayElementsAttr, DenseResourceElementsAttr>(elems))
    return false;

  return true;

  return false;
}

/// Fold the constant at `idx`-th operand of `kernel` into the kernel body.
static void fuseInputAtIndex(MOGG::KernelOp kernel, int32_t idx) {
  Value inputVal = kernel.getOperand(idx);
  MO::ConstantOp cst = tryGetConstantOp(inputVal);

  Block *block = kernel.getBody();

  mlir::OpBuilder builder{kernel->getContext()};
  builder.setInsertionPointToStart(block);

  // Clone the constant and add it to the kernel.
  MO::ConstantOp newOp = cst.clone();

  builder.insert(newOp);

  // Replace all uses of it with the now in-kernel constant.
  block->getArgument(idx).replaceAllUsesWith(newOp.getResult());

  // Remove the argument.
  kernel.eraseInputsAt(idx);

  // Kill the constant if that was the last use.
  if (cst->use_empty())
    cst->erase();
}

void fuseConstant(MOGG::KernelOp kernel) {
  // Traverse from the back to simplify removing operands.
  int32_t preFuseNumOperands = kernel.getNumOperands();
  for (int32_t index = preFuseNumOperands - 1; index >= 0; --index) {
    if (canFuseOperand(kernel, index))
      fuseInputAtIndex(kernel, index);
  }
}

bool dependsOnNoAliasConstantLikesOnly(mlir::Operation *operation) {
  return dependsOnlyOnProperty(operation, [](mlir::Operation *op) -> bool {
    if (!isa<MO::ConstantLike, MO::ChainCreateOp>(op))
      return false;

    auto extConst = dyn_cast<MO::ConstantExternalOp>(op);
    return !extConst || !extConst.getHasAlias();
  });
}

static bool valueDependsTransitivelyOnImpl(Value start, Value targetValue,
                                           mlir::Operation *targetOp) {
  SmallVector<Value> worklist{start};
  llvm::SmallPtrSet<Value, 16> visited;
  while (!worklist.empty()) {
    Value current = worklist.pop_back_val();
    if (!visited.insert(current).second)
      continue;

    if (current == targetValue)
      return true;

    if (Operation *def = current.getDefiningOp()) {
      if (def == targetOp)
        return true;
      worklist.append(def->operand_begin(), def->operand_end());
    }
  }
  return false;
}

bool valueDependsTransitivelyOn(Value start, Value target) {
  if (start == target)
    return true;
  return valueDependsTransitivelyOnImpl(start, target, nullptr);
}

bool valueDependsTransitivelyOn(Value start, mlir::Operation *target) {
  return valueDependsTransitivelyOnImpl(start, nullptr, target);
}

ErrorOr<KernelOutputFusionLambdaType>
getKernelOutputFusionLambdaType(MojoLibraryAnalysis &mojoLibraryAnalysis,
                                MOGG::TensorFusedOp fusedOp,
                                bool enableComputeLambdas) {
  MOGG::BindLambdaOp lambdaOp = fusedOp.getBindLambdaOp();
  if (lambdaOp.getBindKind() != MOGG::BindLambdaKind::OutputLambda)
    return Error("BindLambdaOp kind must be a OutputLambda");

  Value fusedTensor = fusedOp.getResult();
  // Find the kernel call op.
  if (!fusedTensor.hasOneUse())
    return Error("TensorFusedOp should have a unique user");
  auto executeOp = dyn_cast<CallExecuteOp>(*fusedTensor.getUsers().begin());
  if (!executeOp) {
    // If the user is not a CallExecuteOp (e.g., after epilogue fusion with
    // distributed operations where the user is a BindLambdaOp), default to
    // Output fusion type which uses a store.
    return KernelOutputFusionLambdaType::Output;
  }

  // Find the corresponding output index.
  std::optional<unsigned> outputIdx = std::nullopt;
  for (auto [idx, outputVal] : llvm::enumerate(executeOp.getOutputs())) {
    if (outputVal == fusedTensor) {
      outputIdx = idx;
      break;
    }
  }

  if (!outputIdx.has_value())
    return Error("TensorFusedOp result must be a kernel call output");

  // Check if the output operand corresponds to an output lambda.
  auto function = mojoLibraryAnalysis.getRegisteredLITMojoExecuteFunctionFor(
      executeOp.getKernelName());
  KGEN::MOGGPreElab::MojoKernelFunctionAdaptor functionAdaptor(function);
  if (!functionAdaptor.outputArguments[*outputIdx].isTensorWithIOSpec(
          KGEN::MOGGPreElab::IOSpec::FusedComputeOutputTensor)) {
    return KernelOutputFusionLambdaType::Output;
  }

  // Now that we know the expect expects a _FusedComputeOutputTensor, let's
  // check if we build an output or compute lambda.
  if (!enableComputeLambdas)
    return KernelOutputFusionLambdaType::OutputLegacyForCompute;

  // Mixed-precision lambdas (eg cast) must use a classic output lambda.
  MO::TensorType inputTy =
      MO::dynCastFromTensorLike(lambdaOp.getFusibleTensors()[0].getType());
  MO::TensorType outputTy =
      MO::dynCastFromTensorLike(lambdaOp.getOutput().getType());
  if (!inputTy || !outputTy || inputTy.getDtype() != outputTy.getDtype())
    return KernelOutputFusionLambdaType::OutputLegacyForCompute;

  return KernelOutputFusionLambdaType::Compute;
}

} // namespace M::MOGG
