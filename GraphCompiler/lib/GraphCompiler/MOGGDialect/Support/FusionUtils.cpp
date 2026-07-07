//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/GraphCompiler/MOGGDialect/Support/FusionUtils.h"
#include "GraphCompiler/GraphCompiler/MODialect/MOAttrs.h"
#include "GraphCompiler/GraphCompiler/MODialect/MODevices.h"
#include "GraphCompiler/GraphCompiler/MODialect/MOInterfaces.h"
#include "GraphCompiler/GraphCompiler/MODialect/MOOps.h"
#include "GraphCompiler/GraphCompiler/MOGGDialect/Analysis/KernelDeclAdaptor.h"
#include "GraphCompiler/GraphCompiler/MOGGDialect/Analysis/MojoRegistry.h"
#include "GraphCompiler/GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GraphCompiler/GraphCompiler/MOGGDialect/MojoParameters.h"
#include "GraphCompiler/GraphCompiler/MOGGDialect/Support/MOGGTypeTraits.h"
#include "GraphCompiler/Support/DataflowUtils.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/LITDialect/LITUtils.h"
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

  // Store the op's location directly so the fusion tree carries full
  // provenance info.
  auto ctx = op->getContext();
  LocationAttr loc = op->getLoc();

  // Store the op's full function type signature.
  TypeAttr signature;
  auto funcType =
      FunctionType::get(ctx, op->getOperandTypes(), op->getResultTypes());
  signature = TypeAttr::get(funcType);

  return FusionLeafAttr::get(ctx, StringAttr::get(ctx, name), loc, signature);
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
  // Don't fuse constants into non-host (device) kernels: device constant
  // lowering is still incorrect (removing this guard crashes real GPU kernels
  // with illegal memory accesses). See
  // https://linear.app/modularml/issue/GEX-3839.
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

ErrorOr<KernelOutputFusionLambdaType> getKernelOutputFusionLambdaType(
    MojoRegistry &registry, MOGG::TensorFusedOp fusedOp,
    bool enableComputeLambdas, mlir::ModuleOp moduleOp) {
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
  KernelDeclAdaptor adaptor(
      registry.lookupKernelBySymName(executeOp.getKernelSymName()));
  if (!adaptor.executeFunction.has_value())
    return KernelOutputFusionLambdaType::Output;
  const auto &outputArgs = adaptor.executeFunction->outputArguments;
  if (*outputIdx >= outputArgs.size())
    return KernelOutputFusionLambdaType::Output;
  const auto *tensorArg = std::get_if<KernelDeclTensorAdaptor>(
      &outputArgs[*outputIdx].underlyingType);
  if (!tensorArg)
    return KernelOutputFusionLambdaType::Output;

  // The tile-based programming model is selected per-device via the
  // `tile_based_fusion` attribute on the device's `#M.device_info` entry
  // (surfaced to the user through the `tile_codegen` graph-API compile option).
  // When it is set for the device this kernel runs on, the graph compiler uses
  // tile-based kernels and asserts the primary kernel declares the tile-based
  // fused-output IOSpec (_FusedComputeOutputTileTensor); when unset, a
  // tile-based IOSpec is rejected. The device the kernel runs on is carried by
  // the fused output tensor's type (the call op itself defaults to the host
  // device).
  auto deviceRef = MO::getDeviceOrDefaultAny(fusedTensor.getType());
  bool tileBasedFusion =
      MO::DeviceInfoMapping(moduleOp).get(deviceRef).getTileBasedFusion();
  bool kernelIsTile =
      tensorArg->ioSpec == MOGG::IOSpec::FusedComputeOutputTileTensor;
  bool kernelIsSimdCompute =
      tensorArg->ioSpec == MOGG::IOSpec::FusedComputeOutputTensor;
  if (tileBasedFusion) {
    if (kernelIsSimdCompute)
      return Error(
          "device '" + deviceRef.getLabel().str() +
          "' is configured for tile codegen (tile_based_fusion) but kernel '" +
          executeOp.getKernelSymName().str() +
          "' declares a SIMD compute output (_FusedComputeOutputTensor); a "
          "tile kernel (_FusedComputeOutputTileTensor) is required");
    if (kernelIsTile)
      return KernelOutputFusionLambdaType::ComputeTile;
  } else if (kernelIsTile) {
    return Error(
        "kernel '" + executeOp.getKernelSymName().str() +
        "' declares a tile-based fused output (_FusedComputeOutputTileTensor) "
        "but device '" +
        deviceRef.getLabel().str() +
        "' is not configured for tile codegen (tile_based_fusion)");
  }

  if (tensorArg->ioSpec != MOGG::IOSpec::FusedComputeOutputTensor)
    return KernelOutputFusionLambdaType::Output;

  // Now that we know the kernel expects a _FusedComputeOutputTensor, let's
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

bool isTileCodegenForOp(mlir::Operation *op, mlir::ModuleOp moduleOp) {
  auto kernelStubOp = op->getParentOfType<MOGG::KernelStubOp>();
  auto deviceRef = kernelStubOp ? MO::getDeviceOrDefaultAny(kernelStubOp)
                                : MO::getDeviceOrDefaultAny(op);
  return MO::DeviceInfoMapping(moduleOp).get(deviceRef).getTileBasedFusion();
}

} // namespace M::MOGG
