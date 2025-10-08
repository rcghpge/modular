//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/MOGGDialect/Support/MOToMOGGLowering.h"
#include "GenericML/GraphCompiler/MODialect/MODevices.h"
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

  // TODO(GEX-2569): make `mo.transpose` async after enforcing high water mark.
  // All GPU ops are sync to reduce primitive launch overhead, except
  // allreduce.sum because it is both per-device and can block.
  isAsync = isa<MO::AllReduceSumOp>(op) ||
            (MO::isOnHost(op) && !forceSync &&
             !isa<MO::LayoutTransformOp>(op) && !isa<MO::TransposeOp>(op));

  // Get ready for building ops.
  builder.setInsertionPoint(op);
}

// Convert mo.tensor with 1 element to mo.scalar if the kernel expects a scalar
// argument.
SmallVector<Value>
convertScalarsIfNeeded(ArrayRef<Value> originalInputs,
                       const IndexToOperandAdaptorMapping &mapping,
                       OpBuilder &builder) {
  SmallVector<Value> ret;
  for (auto [idx, v] : llvm::enumerate(originalInputs)) {
    // If this is a fusible input, use the fused value, otherwise we
    // captured the infusible inputs.
    const auto &mappedArg = mapping.lookup(idx);
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

SmallVector<Value>
convertTuplesIfNeeded(ArrayRef<Value> originalInputs,
                      const IndexToOperandAdaptorMapping &mapping,
                      OpBuilder &builder) {
  SmallVector<Value> ret;
  uint64_t idx = 0;
  while (idx < originalInputs.size()) {
    if (!mapping.lookup(idx).isVariadicTensorType()) {
      ret.push_back(originalInputs[idx]);
      ++idx;
      continue;
    }

    SmallVector<Value> valuesThatMatchVariadics;
    SmallVector<Type> tupleTypes;
    KGEN::MOGGPreElab::MojoKernelOperandAdaptor variadicType;
    do {
      tupleTypes.push_back(originalInputs[idx].getType());
      valuesThatMatchVariadics.push_back(originalInputs[idx]);
      variadicType = mapping.lookup(idx);
      ++idx;
    } while (idx < originalInputs.size() &&
             mapping.lookup(idx).sourceDescriptor->position ==
                 variadicType.sourceDescriptor->position);
    ret.push_back(builder.create<MOGG::AsTupleOp>(
        valuesThatMatchVariadics.front().getLoc(),
        builder.getTupleType(tupleTypes), valuesThatMatchVariadics));
  }

  return ret;
}

template <class FuncT>
void addContextIfPresent(
    Operation *op, SmallVector<Value> &inputs,
    const KGEN::MOGGPreElab::MojoKernelAdaptor<FuncT> &adaptor,
    OpBuilder &builder, bool forceAddContext = false) {
  bool hasContextType = forceAddContext;
  bool isVariadic = false;
  for (auto &&arg : adaptor.executeFunction->inputArguments) {
    if (arg.isContextType()) {
      hasContextType = true;
      isVariadic = arg.isVariadicContextType();
    }
  }
  if (!hasContextType)
    return;

  auto loc = op->getLoc();
  auto ctx = op->getContext();

  if (isVariadic) {
    SmallVector<Value> contexts;
    for (auto deviceAttr : MO::getUniqueDevicesFromKernel(op)) {
      contexts.push_back(builder.create<MOGG::DeviceContextPlaceholderOp>(
          loc, MOGG::ContextType::get(ctx), deviceAttr));
    }

    inputs.push_back(builder.create<MOGG::AsTupleOp>(
        loc,
        builder.getTupleType(
            SmallVector<Type>(contexts.size(), MOGG::ContextType::get(ctx))),
        contexts));
  } else {
    inputs.push_back(builder.create<MOGG::DeviceContextPlaceholderOp>(
        loc, MOGG::ContextType::get(ctx), MO::getDeviceOrDefaultAny(op)));
  }
}

template <class FuncT>
KernelCallInterface MOGGExpKernelBuilder::lowerToElementwiseKernel(
    ExpKernelOp kernelOp, const OperandMapping &mapping,
    const KGEN::MOGGPreElab::MojoKernelAdaptor<FuncT> &adaptor) {
  auto execBody = &kernelOp.getExecBody().front();
  SmallVector<Value> inputs(execBody->getArguments().begin(),
                            execBody->getArguments().end());

  Value outputValue = nullptr;
  MO::TensorType tensorType = nullptr;
  /// TODO(GEX-2360): These checks are extremely brittle. We need to agree on
  /// clear rules around mutable elementwise kernels.
  bool isInPlaceElementwise =
      isa<MO::MutableStoreOp>(moOp) || !kernelOp.getOutputPlaceholder();
  if (isInPlaceElementwise) {
    for (auto argument : kernelOp.getExecBody().getArguments()) {
      if (auto asBufferType = dyn_cast<MO::BufferType>(argument.getType())) {
        outputValue = argument;
        tensorType = asBufferType.toTensorType();
        break;
      }
    }
  } else {
    outputValue = kernelOp.getOutputPlaceholder().getResult(0);
    tensorType = cast<MO::TensorType>(outputValue.getType());
  }

  /// TODO(GEX-2314): Support opaque elementwise in the new dialect.
  if (!tensorType)
    return lowerToNormalKernel(kernelOp, mapping, adaptor);

  Type resultValTp = tensorType.getElementType();

  SmallVector<Value> fusibleTensors;
  for (const auto &[idx, arg] : mapping.operandMapping) {
    if (arg.isFusedTensorType())
      fusibleTensors.push_back(inputs[idx]);
  }

  auto bindOp = builder.create<BindLambdaOp>(loc, tensorType, fusibleTensors,
                                             BindLambdaKind::ElemwiseLambda);

  Block *elemwiseBody = &bindOp.getLambda().front();
  unsigned fusedArgIdx = 0;
  SmallVector<Value> outlinedArgs;
  for (const auto &[idx, arg] : mapping.operandMapping) {
    // If this is a fusible input, use the fused value, otherwise we
    // captured the infusible inputs.
    if (arg.isFusedTensorType())
      outlinedArgs.push_back(elemwiseBody->getArgument(fusedArgIdx++));
    // If this is a mutable store and that this is the mutable input, don't put
    // it in the list of outlined args. Otherwise, just capture everything.
    else if (!isInPlaceElementwise || !arg.isMutableTensorType())
      outlinedArgs.push_back(inputs[idx]);
  }

  builder.setInsertionPoint(bindOp);
  outlinedArgs = convertTuplesIfNeeded(
      convertScalarsIfNeeded(outlinedArgs, mapping.operandMapping, builder),
      mapping.operandMapping, builder);

  // The elementwise function might have an indices argument at the end.
  if (!adaptor.elementwiseFunction->inputArguments.empty() &&
      adaptor.elementwiseFunction->inputArguments.back().isOpaqueWithTypeName(
          KGEN::MOGGPreElab::MOJO_INTERNAL_DPS_INDEX_LIST_TYPE_NAME))
    outlinedArgs.push_back(bindOp.getIndicesArg());

  builder.setInsertionPointToStart(elemwiseBody);
  auto callElementwise = builder.create<CallElementwiseOp>(
      loc, resultValTp, adaptor.registeredOpName, outlinedArgs);
  builder.create<OutputOp>(loc, callElementwise->getResults());
  builder.setInsertionPointAfter(bindOp);

  SmallVector<Value> contextValue;
  addContextIfPresent(moOp, contextValue, adaptor, builder, true);
  auto ret = builder.create<CallForeachOp>(loc, bindOp.getOutput(), outputValue,
                                           contextValue);

  return ret;
}

template <class FuncT>
ErrorOr<KernelCallInterface> MOGGExpKernelBuilder::lowerToViewKernel(
    ExpKernelOp kernelOp, const OperandMapping &mapping,
    const KGEN::MOGGPreElab::MojoKernelAdaptor<FuncT> &adaptor) {
  if (kernelOp->getNumResults() != 1)
    return Error("Expected view op to have exactly one result");

  auto execBody = &kernelOp.getExecBody().front();
  Value outputVal = kernelOp.getOutputPlaceholder()->getResult(0);
  SmallVector<Value> inputs(execBody->getArguments().begin(),
                            execBody->getArguments().end());

  if (isa<MO::Static::ReshapeOp, MO::Static::BroadcastToOp>(moOp)) {
    inputs.push_back(builder.create<MOGG::TensorGetShapeOp>(
        loc, TypeAttr::get(outputVal.getType())));
  }

  inputs = convertTuplesIfNeeded(
      convertScalarsIfNeeded(inputs, mapping.operandMapping, builder),
      mapping.operandMapping, builder);
  auto view = builder.create<MOGG::CallViewOp>(
      loc, outputVal.getType(), adaptor.registeredOpName, inputs);
  // MaterializeOp always takes an input tensor (with view updated by the
  // mogg.call.view above).
  SmallVector<Value> contextValue;
  addContextIfPresent(moOp, contextValue, adaptor, builder, true);
  builder.create<MOGG::CallMaterializeOp>(loc, view->getResult(0), outputVal,
                                          contextValue);

  return view;
}

template <class FuncT>
KernelCallInterface MOGGExpKernelBuilder::lowerToNormalKernel(
    ExpKernelOp kernelOp, const OperandMapping &mapping,
    const KGEN::MOGGPreElab::MojoKernelAdaptor<FuncT> &adaptor) {
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
  inputs = convertTuplesIfNeeded(
      convertScalarsIfNeeded(inputs, mapping.operandMapping, builder),
      mapping.operandMapping, builder);
  addContextIfPresent(moOp, inputs, adaptor, builder);
  outputs = convertTuplesIfNeeded(outputs, mapping.resultMapping, builder);

  return builder.create<CallExecuteOp>(loc, adaptor.registeredOpName,
                                       regRetTypeAttr, inputs, outputs);
}

static ErrorOr<NamedAttribute>
getValidatedParameter(const NamedAttribute &attr) {
  auto value = attr.getValue();

  // Validate supported attribute types.
  if (!isa<IntegerAttr, StringAttr, TypeAttr>(attr.getValue()))
    return Error(llvm::formatv("Unsupported typed attribute {}: {}",
                               attr.getName(), attr.getValue()));

  // Convert TypeAttr to DTypeConstantAttr
  if (auto typeAttr = dyn_cast<TypeAttr>(value)) {
    auto typedTypeAttr = KGEN::DTypeConstantAttr::get(
        typeAttr.getContext(), MO::getDTypeOrInvalid(typeAttr.getValue()));
    value = typedTypeAttr;
  }

  return NamedAttribute(attr.getName(), value);
}

ErrorOr<DictionaryAttr> MOGGExpKernelBuilder::forwardParameters() {
  auto customOpOperationName =
      mlir::OperationName(MO::CustomOp::getOperationName(), moOp->getContext());
  SmallVector<NamedAttribute> forwardedParameters;

  // Custom op is handled specially by parsing the parameters dictionary.
  if (auto custom = dyn_cast<MO::CustomOp>(moOp)) {
    if (auto validParameters =
            mojoLibraryAnalysis.validateCustomOpParameters(custom))
      return validParameters.takeError();

    for (NamedAttribute param : custom.getParameters()) {
      if (ErrorOr<NamedAttribute> validated = getValidatedParameter(param))
        return validated.takeError();
      else
        forwardedParameters.push_back(validated.takeValue());
    }

    return DictionaryAttr::get(moOp->getContext(), forwardedParameters);
  }

  for (auto param : moOp->getAttrDictionary()) {
    if (isa<M::DeviceRefAttr>(param.getValue()))
      continue;

    // Using `CustomOp` for the parameter name but could have use anything.
    if (param.getName() ==
        MO::CustomOp::getOutputParamDeclsAttrName(customOpOperationName)) {
      continue;
    }

    auto asTypeAttr = dyn_cast<TypeAttr>(param.getValue());
    if (asTypeAttr) {
      // Try converting TypeAttr to the corresponding DTypeConstantAttr. All
      // KGEN parameter has to be a TypedAttr.
      auto typedTypeAttr = KGEN::DTypeConstantAttr::get(
          asTypeAttr.getContext(),
          MO::getDTypeOrInvalid(asTypeAttr.getValue()));

      forwardedParameters.push_back(
          NamedAttribute(param.getName(), typedTypeAttr));
      continue;
    }

    auto asTypedAttr = dyn_cast<TypedAttr>(param.getValue());
    if (asTypedAttr) {
      if (isa<IntegerAttr, StringAttr, BoolAttr>(asTypedAttr)) {
        forwardedParameters.push_back(param);
        continue;
      }
      return Error(llvm::formatv("Unsupported typed attribute {} : {}",
                                 param.getName(), param.getValue()));
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

    if (useExperimentalMojoPath()) {
      // Build an ArrayAttr for EmitMojo.
      forwardedParameters.push_back(
          NamedAttribute(paramName, builder.getI64ArrayAttr(values)));
    } else {
      // Lower the dimensions to KGEN.
      auto attrOrError = typeManager.lowerStaticDims(builder, values);
      forwardedParameters.push_back(
          NamedAttribute(paramName, attrOrError.takeValue()));
    }
  };
  if (auto convOp = dyn_cast<MO::ConvOp>(moOp)) {
    extractConstantValues(convOp.getStrides(),
                          MOGG::PARAMETER_CONV_STATIC_STRIDES);
    extractConstantValues(convOp.getDilations(),
                          MOGG::PARAMETER_CONV_STATIC_DILATIONS);
    extractConstantValues(convOp.getPaddings(),
                          MOGG::PARAMETER_CONV_STATIC_PADDING);
  } else if (auto sliceDimOp = dyn_cast<MO::SliceDimOp>(moOp)) {
    extractConstantValues(sliceDimOp.getStart(),
                          MOGG::PARAMETER_SLICE_DIM_STATIC_START);
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

template <class FuncT>
ErrorOr<ExpKernelOp> MOGGExpKernelBuilder::lowerToKernelBase() {
  auto maybeAdaptor = getMojoKernelAdaptor<FuncT>(mojoLibraryAnalysis, moOp);
  if (maybeAdaptor.isError())
    return maybeAdaptor.takeError();

  KGEN::MOGGPreElab::MojoKernelAdaptor<FuncT> adaptor =
      maybeAdaptor.takeValue();
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
  if (auto memEffectOp = dyn_cast<mlir::MemoryEffectOpInterface>(moOp)) {
    kernelOp.setHasWriteEffect(
        memEffectOp.hasEffect<mlir::MemoryEffects::Write>());
    kernelOp.setHasReadEffect(
        memEffectOp.hasEffect<mlir::MemoryEffects::Read>());
  }
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

  ErrorOr<KernelCallInterface> kernelCallingOperation = nullptr;
  if (adaptor.isElementwise()) {
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

  MOGG::addForwardDeclsOfReferencedFuncs(
      kernelOp, kernelOp->getParentOfType<ModuleOp>(), mojoLibraryAnalysis);

  return kernelOp;
}

ErrorOr<ExpKernelOp> MOGGExpKernelBuilder::lowerToKernel() {
  if (useExperimentalMojoPath())
    return lowerToKernelBase<KGEN::LIT::FnOp>();
  else
    return lowerToKernelBase<KGEN::GeneratorOp>();
}
} // namespace M::MOGG
