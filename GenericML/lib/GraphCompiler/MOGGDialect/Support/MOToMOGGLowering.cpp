//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/MOGGDialect/Support/MOToMOGGLowering.h"
#include "GenericML/GraphCompiler/MODialect/MOOps.h"
#include "GenericML/GraphCompiler/MOGGDialect/Analysis/MojoLibraryAnalysis.h"
#include "GenericML/GraphCompiler/MOGGDialect/Analysis/Support.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GenericML/GraphCompiler/MOGGDialect/Support/FusionUtils.h"
#include "GenericML/GraphCompiler/MOGGDialect/Support/KGENStubBuilder.h"

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/LITDialect/LITUtils.h"
#include "Support/AssertStream.h"

#include "llvm/ADT/STLExtras.h"

namespace M::MOGG {

MOGGExpKernelBuilder::MOGGExpKernelBuilder(
    Operation *op, OpBuilder &insertPoint, bool forceSync,
    MOGG::MojoLibraryAnalysis &mojoLibraryAnalysis)
    : moOp(op), graphOp(op->getParentOfType<MO::GraphOp>()),
      builder(insertPoint), ctx(op->getContext()), loc(op->getLoc()),
      mojoLibraryAnalysis(mojoLibraryAnalysis),
      typeManager(
          MOGG::TypeManager::get(ctx, mojoLibraryAnalysis).takeValue()) {

  isAsync = (getDevice() == getHostDevice()) && !forceSync &&
            !isa<MO::LayoutTransformOp>(op);

  // Get ready for building ops.
  builder.setInsertionPoint(op);
}

void MOGGExpKernelBuilder::lowerToElementwiseKernel(
    ExpKernelOp kernelOp, const OperandMapping &mapping,
    const KernelAdaptor &adaptor) {
  auto execBody = &kernelOp.getExecBody().front();
  SmallVector<Value> inputs(execBody->getArguments().begin(),
                            execBody->getArguments().end());

  auto outTensorTp = cast<MO::TensorType>(kernelOp->getResultTypes()[0]);
  auto resultValTp = outTensorTp.getElementType();

  // Values
  auto inputValTps = llvm::map_to_vector(moOp->getOperandTypes(), [](Type t) {
    // Only handles tensor inputs for now.
    return cast<MO::TensorType>(t).getElementType();
  });

  SmallVector<Value> fusibleTensors;
  for (const auto &[idx, arg] : mapping.operandMapping) {
    if (arg.isFusedTensorType())
      fusibleTensors.push_back(inputs[idx]);
  }

  auto bindOp = builder.create<BindLambdaOp>(loc, outTensorTp, fusibleTensors,
                                             BindLambdaKind::ElemwiseLambda);

  Block *elemwiseBody = &bindOp.getLambda().front();
  builder.setInsertionPointToStart(&bindOp.getLambda().front());

  unsigned fusedArgIdx = 0;
  SmallVector<Value> outlinedArgs;
  for (auto [idx, v] : llvm::enumerate(inputs)) {
    // If this is a fusible input, use the fused value, otherwise we
    // captured the infusible inputs.
    if (mapping.operandMapping.at(idx).isFusedTensorType())
      outlinedArgs.push_back(elemwiseBody->getArgument(fusedArgIdx++));
    else
      outlinedArgs.push_back(v);
  }

  builder.setInsertionPointToStart(elemwiseBody);
  auto callOutlined = builder.create<CallOutlinedOp>(
      loc, resultValTp, adaptor.registeredOpName, outlinedArgs);
  builder.create<OutputOp>(loc, callOutlined->getResults());
  builder.setInsertionPointAfter(bindOp);
  builder.create<CallForeachOp>(loc, bindOp->getResult(0),
                                kernelOp.getOutputPlaceholder()->getResult(0));
}

static ErrorOr<DictionaryAttr> forwardParameters(Operation *op,
                                                 OpBuilder &builder) {
  auto customOpOperationName =
      mlir::OperationName(MO::CustomOp::getOperationName(), op->getContext());
  SmallVector<NamedAttribute> forwardedParameters;
  for (auto param : op->getAttrDictionary()) {
    // Using `CustomOp` for the parameter name but could have use anything.
    if (param.getName() ==
        MO::CustomOp::getOutputParamDeclsAttrName(customOpOperationName)) {
      continue;
    }

    if (isa<MO::CustomOp>(op)) {
      if (param.getName() ==
          MO::CustomOp::getOperandDevicesAttrName(customOpOperationName)) {
        continue;
      }
      if (param.getName() ==
          MO::CustomOp::getSymbolAttrName(customOpOperationName)) {
        continue;
      }

      if (param.getName() ==
          MO::CustomOp::getFunctionAttrName(customOpOperationName)) {
        continue;
      }
    }

    auto asTypeAttr = dyn_cast<TypeAttr>(param.getValue());
    if (asTypeAttr) {
      forwardedParameters.push_back(param);
      continue;
    }

    auto asTypedAttr = dyn_cast<TypedAttr>(param.getValue());
    if (!asTypedAttr) {
      return Error(llvm::formatv("Unsupported untyped attribute {} : {}",
                                 param.getName(), param.getValue()));
    }

    if (isa<IntegerAttr, StringAttr, BoolAttr>(asTypedAttr)) {
      forwardedParameters.push_back(param);
      continue;
    }

    auto coord = dyn_cast<MO::CoordinateTransformModeAttr>(param.getValue());
    if (!coord) {
      return Error(llvm::formatv("Unsupported untyped attribute {} : {}",
                                 param.getName(), param.getValue()));
    }

    forwardedParameters.push_back(NamedAttribute{
        param.getName(),
        builder.getIndexAttr(static_cast<int64_t>(coord.getValue()))});
  }

  return DictionaryAttr::get(op->getContext(), forwardedParameters);
}

void MOGGExpKernelBuilder::lowerToViewKernel(ExpKernelOp kernelOp,
                                             const OperandMapping &mapping,
                                             const KernelAdaptor &adaptor) {
  auto execBody = &kernelOp.getExecBody().front();
  Value outputVal = kernelOp.getOutputPlaceholder()->getResult(0);
  SmallVector<Value> inputs(execBody->getArguments().begin(),
                            execBody->getArguments().end());
  // Build the shape
  auto tensorTp = cast<MO::TensorType>(outputVal.getType());
  Value shapeInKGEN = MOGG::lowerShapeToKGEN(loc, builder, tensorTp.getDims());

  // TODO(MAXPLAT-212): use kernel adaptor, view kernels does not always take an
  // output shape.
  inputs.push_back(shapeInKGEN);
  Value viewChanged = builder.create<MOGG::CallUpdateViewOp>(
      loc, outputVal.getType(), adaptor.registeredOpName, inputs);
  // Relink input tensor ssa
  inputs[0] = viewChanged;

  builder.create<MOGG::CallMaterializeOp>(loc, inputs, outputVal);
}

ErrorOrSuccess
MOGGExpKernelBuilder::lowerToNormalKernel(ExpKernelOp kernelOp,
                                          const OperandMapping &mapping,
                                          const KernelAdaptor &adaptor) {
  auto execBody = &kernelOp.getExecBody().front();
  auto outputPlaceholder = kernelOp.getOutputPlaceholder();
  SmallVector<Value> outputs(outputPlaceholder->getResults());
  SmallVector<Value> inputs(execBody->getArguments().begin(),
                            execBody->getArguments().end());

  SmallVector<Value> fusibleTensors;
  for (const auto &[idx, arg] : mapping.operandMapping) {
    if (!arg.isFusedTensorType())
      continue;
    Value concreteInput = inputs[idx];
    Value bound = builder.create<BindLambdaOp>(loc, concreteInput.getType(),
                                               concreteInput,
                                               BindLambdaKind::InputLambda);
    inputs[idx] = bound;
  }

  for (const auto &[idx, arg] : mapping.resultMapping) {
    if (!arg.isFusedTensorType())
      continue;
    Value concreteOutput = outputs[idx];
    Value bound = builder.create<BindLambdaOp>(loc, concreteOutput.getType(),
                                               concreteOutput,
                                               BindLambdaKind::OutputLambda);
    outputs[idx] = bound;
  }

  auto forwardedParametersOr = forwardParameters(moOp, builder);
  if (forwardedParametersOr.isError())
    return forwardedParametersOr.takeError();

  auto exec = builder.create<CallExecuteOp>(loc, adaptor.registeredOpName,
                                            inputs, outputs);
  exec->setAttrs(*forwardedParametersOr);

  return success();
}

ErrorOr<ExpKernelOp> MOGGExpKernelBuilder::lowerToKernel() {
  auto mojoKernelOr = getMojoKernelTarget(mojoLibraryAnalysis, moOp);
  if (mojoKernelOr.isError())
    return mojoKernelOr.takeError();

  KernelAdaptor adaptor("", *mojoKernelOr, nullptr);
  auto operandMappingOr =
      mapMOOpToKernelImplementation(moOp, *adaptor.executeFunction);

  if (operandMappingOr.isError())
    return operandMappingOr.takeError();

  auto kernelOp =
      builder.create<ExpKernelOp>(loc,
                                  /*results=*/moOp->getResultTypes(),
                                  /*operands=*/moOp->getOperands(),
                                  /*summary=*/getSummaryString(moOp),
                                  /*isAsync=*/builder.getBoolAttr(isAsync),
                                  /*device=*/getDevice());
  builder.setInsertionPointToStart(&kernelOp.getExecBody().front());
  builder.create<OutputPlaceholdersOp>(loc, kernelOp->getResultTypes());

  ErrorOrSuccess err = success();
  if (adaptor.executeFunction->isElementwise())
    lowerToElementwiseKernel(kernelOp, *operandMappingOr, adaptor);
  else if (adaptor.executeFunction->isViewKernel())
    lowerToViewKernel(kernelOp, *operandMappingOr, adaptor);
  else
    err = lowerToNormalKernel(kernelOp, *operandMappingOr, adaptor);

  if (err.isError())
    return err.takeError();

  return kernelOp;
}
} // namespace M::MOGG
