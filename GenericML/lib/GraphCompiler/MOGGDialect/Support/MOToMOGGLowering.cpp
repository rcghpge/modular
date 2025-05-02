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
#include "GenericML/GraphCompiler/MOGGDialect/MojoParameters.h"
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

  isAsync = (getDevice() == getHostDevice()) && !forceSync;

  // Get ready for building ops.
  builder.setInsertionPoint(op);
}

// Convert mo.tensor with 1 element to mo.scalar if the kernel expects a scalar
// argument.
SmallVector<Value> convertScalarsIfNeeded(ArrayRef<Value> originalInputs,
                                          const OperandMapping &mapping,
                                          OpBuilder &builder) {
  SmallVector<Value> ret;
  for (auto [idx, v] : llvm::enumerate(originalInputs)) {
    // If this is a fusible input, use the fused value, otherwise we
    // captured the infusible inputs.
    const auto &mappedArg = mapping.operandMapping.lookup(idx);
    if (mappedArg.isScalarType()) {
      ret.push_back(builder.create<MOGG::AsScalarOp>(
          v.getLoc(),
          MO::ScalarType::get(v.getContext(),
                              cast<MO::TensorType>(v.getType()).getDtype()),
          v));
    } else {
      ret.push_back(v);
    }
  }

  return ret;
}

KernelTerminator
MOGGExpKernelBuilder::lowerToElementwiseKernel(ExpKernelOp kernelOp,
                                               const OperandMapping &mapping,
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
  unsigned fusedArgIdx = 0;
  SmallVector<Value> outlinedArgs;
  for (auto [idx, v] : llvm::enumerate(inputs)) {
    // If this is a fusible input, use the fused value, otherwise we
    // captured the infusible inputs.
    const auto &mappedArg = mapping.operandMapping.lookup(idx);
    if (mappedArg.isFusedTensorType())
      outlinedArgs.push_back(elemwiseBody->getArgument(fusedArgIdx++));
    else
      outlinedArgs.push_back(v);
  }

  builder.setInsertionPoint(bindOp);
  outlinedArgs = convertScalarsIfNeeded(outlinedArgs, mapping, builder);

  builder.setInsertionPointToStart(elemwiseBody);
  auto callOutlined = builder.create<CallOutlinedOp>(
      loc, resultValTp, adaptor.registeredOpName, outlinedArgs);
  builder.create<OutputOp>(loc, callOutlined->getResults());
  builder.setInsertionPointAfter(bindOp);
  return builder.create<CallForeachOp>(
      loc, bindOp->getResult(0), kernelOp.getOutputPlaceholder()->getResult(0));
}

ErrorOr<KernelTerminator>
MOGGExpKernelBuilder::lowerToViewKernel(ExpKernelOp kernelOp,
                                        const OperandMapping &mapping,
                                        const KernelAdaptor &adaptor) {
  if (kernelOp->getNumResults() != 1)
    return Error("Expected view op to have exactly one result");

  auto execBody = &kernelOp.getExecBody().front();
  Value outputVal = kernelOp.getOutputPlaceholder()->getResult(0);
  SmallVector<Value> inputs(execBody->getArguments().begin(),
                            execBody->getArguments().end());
  // Build the shape
  auto tensorTp = cast<MO::TensorType>(outputVal.getType());
  Value shapeInKGEN = MOGG::lowerShapeToKGEN(loc, builder, tensorTp.getDims());

  if (isa<MO::Static::ReshapeOp, MO::Static::BroadcastToOp>(moOp))
    inputs.push_back(shapeInKGEN);

  Operation *view = builder.create<MOGG::CallViewOp>(
      loc, outputVal.getType(), adaptor.registeredOpName,
      convertScalarsIfNeeded(inputs, mapping, builder));
  // Relink input tensor ssa
  inputs[0] = view->getResult(0);

  return builder.create<MOGG::CallMaterializeOp>(loc, inputs, outputVal);
}

KernelTerminator
MOGGExpKernelBuilder::lowerToNormalKernel(ExpKernelOp kernelOp,
                                          const OperandMapping &mapping,
                                          const KernelAdaptor &adaptor) {
  auto execBody = &kernelOp.getExecBody().front();
  auto outputPlaceholder = kernelOp.getOutputPlaceholder();
  SmallVector<Value> outputs;
  if (outputPlaceholder)
    outputs = outputPlaceholder->getResults();

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

  SmallVector<Type, 1> retVal;
  for (auto [idx, val] : mapping.resultMapping)
    if (!val.isDpsOutput())
      retVal.push_back(kernelOp->getResult(idx).getType());

  // Should we relax the constraint?
  ASSERT_STREAM(retVal.size() <= 1,
                "[MO-TO-MOGG] Should have at most one return by value result");
  TypeAttr regRetTypeAttr =
      retVal.empty() ? nullptr : TypeAttr::get(retVal.front());
  return builder.create<CallExecuteOp>(
      loc, adaptor.registeredOpName, regRetTypeAttr,
      convertScalarsIfNeeded(inputs, mapping, builder), outputs);
}

ErrorOr<DictionaryAttr> MOGGExpKernelBuilder::forwardParameters() {
  auto customOpOperationName =
      mlir::OperationName(MO::CustomOp::getOperationName(), moOp->getContext());
  SmallVector<NamedAttribute> forwardedParameters;
  for (auto param : moOp->getAttrDictionary()) {
    if (isa<M::DeviceRefAttr>(param.getValue()))
      continue;

    // Using `CustomOp` for the parameter name but could have use anything.
    if (param.getName() ==
        MO::CustomOp::getOutputParamDeclsAttrName(customOpOperationName)) {
      continue;
    }

    if (isa<MO::CustomOp>(moOp)) {
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

  /// TODO(GEX-2177): Remove this by adding an opt-in in the implementation and
  /// generically supporting static values in kernels.
  auto extractConstantValues = [&](Value ssaValue, StringLiteral paramName) {
    SmallVector<int64_t> values;
    auto asTensor = cast<MO::TensorValue>(ssaValue);
    if (auto maybeIntVec = tryGetConstantIntVector(asTensor)) {
      values = std::move(*maybeIntVec);
    } else if (auto maybeIntScalar = tryGetConstantInt(asTensor)) {
      values.push_back(*maybeIntScalar);
    } else {
      // Not constant.
      MO::TensorType t = asTensor.getType();
      if (t.hasStaticShape()) {
        int64_t numElems = *t.getNumElements();
        values.append(numElems, ShapedType::kDynamic);
      } else {
        values.push_back(mlir::ShapedType::kDynamic);
      }
    }

    SmallVector<Attribute> asAttrs;
    for (int64_t value : values)
      asAttrs.push_back(MO::ShapeAttr::getDim(moOp->getContext(), value));

    forwardedParameters.push_back(
        NamedAttribute(paramName, ArrayAttr::get(moOp->getContext(), asAttrs)));
  };
  if (auto convOp = dyn_cast<MO::ConvOp>(moOp)) {
    extractConstantValues(convOp.getStrides(),
                          MOGG::PARAMETER_CONV_STATIC_STRIDES);
    extractConstantValues(convOp.getDilations(),
                          MOGG::PARAMETER_CONV_STATIC_DILATIONS);
    extractConstantValues(convOp.getPaddings(),
                          MOGG::PARAMETER_CONV_STATIC_PADDING);
  } else if (auto sliceDimOp = dyn_cast<MO::SliceDimOp>(moOp)) {
    extractConstantValues(sliceDimOp.getStep(),
                          MOGG::PARAMETER_SLICE_DIM_STATIC_STEP);
  } else if (auto transposeOp = dyn_cast<MO::TransposeOp>(moOp)) {
    extractConstantValues(transposeOp.getPerm(),
                          MOGG::PARAMETER_TRANSPOSE_STATIC_PERM);
  } else if (auto sliceOp = dyn_cast<MO::SliceOp>(moOp)) {
    extractConstantValues(sliceOp.getStep(),
                          MOGG::PARAMETER_SLICE_STATIC_STEPS);
  }

  return DictionaryAttr::get(moOp->getContext(), forwardedParameters);
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

  SmallVector<Type> dpsReturns;
  const OperandMapping &operandMapping = operandMappingOr.get();
  for (auto [idx, val] : operandMapping.resultMapping)
    if (val.isDpsOutput())
      dpsReturns.push_back(kernelOp->getResult(idx).getType());

  // Only need output place holder for dps returned value (they need to be
  // allocated by caller).
  if (!dpsReturns.empty())
    builder.create<OutputPlaceholdersOp>(loc, dpsReturns);

  auto forwardedParametersOr = forwardParameters();
  if (forwardedParametersOr.isError())
    return forwardedParametersOr.takeError();

  ErrorOr<KernelTerminator> kernelCallingOperation = nullptr;
  if (adaptor.executeFunction->isElementwise()) {
    kernelCallingOperation =
        lowerToElementwiseKernel(kernelOp, operandMapping, adaptor);
  } else if (adaptor.executeFunction->isView()) {
    kernelCallingOperation =
        lowerToViewKernel(kernelOp, operandMapping, adaptor);
  } else {
    kernelCallingOperation =
        lowerToNormalKernel(kernelOp, operandMapping, adaptor);
  }

  if (kernelCallingOperation.isError())
    return kernelCallingOperation.takeError();

  kernelCallingOperation.get().setKernelParams(*forwardedParametersOr);

  // TODO(MAXPLAT-224): I am not sure whether this is correct, no idea why we
  // need the attribute: it seems that mo-to-primitive can only handle tensor
  // output anyway, simply skip non-tensor output during mo-to-primitives should
  // just work?
  SmallVector<int64_t> moResultIndicesAllocatedInPlace;
  for (auto [idx, val] : operandMapping.resultMapping)
    if (!val.isDpsOutput())
      moResultIndicesAllocatedInPlace.push_back(idx);

  kernelOp.setAllocIndicesAttr(
      builder.getIndexArrayAttr(moResultIndicesAllocatedInPlace));

  return kernelOp;
}
} // namespace M::MOGG
