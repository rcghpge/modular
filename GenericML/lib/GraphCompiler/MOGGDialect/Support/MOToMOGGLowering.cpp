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

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/LITDialect/LITUtils.h"
#include "KGEN/MOGGPreElab/MOGGPreElabDecorators.h"
#include "Support/AssertStream.h"

#include "llvm/ADT/STLExtras.h"

namespace M::MOGG {

/// Infer kernel parameters from opaque type parameters.
///
/// For emit-mojo path, we need to resolve kernel parameters (P0, P1, etc.)
/// from the opaque type parameters (S0, S1, etc.) on operands and results.
/// The kernel function has annotations that map these:
///   mogg.value_params: [{S0 = #kgen.param.decl.ref<P0>, ...}, ...]
///   mogg.value_result_params: [{S0 = #kgen.param.decl.ref<P0>, ...}]
///
/// This function matches the opaque type's concrete values to the kernel's
/// parameter declarations.
static void
inferOpaqueParameters(SmallVectorImpl<NamedAttribute> &forwardedParameters,
                      Operation *moOp, KGEN::LIT::FnOp function) {
  if (!function)
    return;

  // Get the kernel annotations for input and result parameters.
  auto inputParamsAttr = function->getAttrOfType<ArrayAttr>(
      KGEN::MOGGPreElab::kKernelValueParameterAttrName);
  auto resultParamsAttr = function->getAttrOfType<ArrayAttr>(
      KGEN::MOGGPreElab::kKernelResultParameterAttrName);

  // Build a set of explicitly declared parameter names.
  // Parameters with a backtick in their name are internal/inferred parameters
  // (e.g., "S1`2x1" from a placeholder "_" in the signature). We only want
  // the explicitly declared parameters (e.g., "P0" from "[P0: Int]").
  llvm::DenseSet<StringRef> declaredParams;
  for (auto param : function.getInputParams()) {
    StringRef paramName = param.getName();
    // Skip mangled parameter names (contain backtick) - these are inferred
    // parameters from type parameters, not explicitly declared.
    if (paramName.contains('`'))
      continue;
    declaredParams.insert(paramName);
  }

  // Helper to match parameters from an opaque type against kernel annotations.
  auto inferFromOpaque = [&](DictionaryAttr kernelParamDict,
                             MO::OpaqueType opaqueType) {
    if (!kernelParamDict || !opaqueType)
      return;

    DictionaryAttr opaqueParams = opaqueType.getParameters();
    if (!opaqueParams)
      return;

    for (NamedAttribute kernelParam : kernelParamDict) {
      // Check if this is a parameter declaration reference (e.g.,
      // #kgen.param.decl.ref<P0>)
      auto declRef = dyn_cast<KGEN::ParamDeclRefAttr>(kernelParam.getValue());
      if (!declRef)
        continue;

      // Only include parameters that are explicitly declared in the kernel's
      // parameter list. This filters out inferred/anonymous parameters like
      // "_" which have internal mangled names (contain backticks).
      StringRef paramName = declRef.getName();
      if (!declaredParams.contains(paramName))
        continue;

      // Look up the corresponding value in the opaque type's parameters.
      // The key is the same (e.g., "S0" matches "S0" in the opaque type).
      auto opaqueValue = opaqueParams.getNamed(kernelParam.getName());
      if (!opaqueValue)
        continue;

      auto typedValue = dyn_cast<TypedAttr>(opaqueValue->getValue());
      if (!typedValue)
        continue;

      // Add the resolved parameter: kernel param name (P0) -> concrete value.
      // declRef.getName() returns a StringAttr which is the parameter name.
      forwardedParameters.push_back(
          NamedAttribute(declRef.getName(), typedValue));
    }
  };

  // Infer parameters from operands.
  // Note: The kernel function's argument list has DPS outputs first, then
  // inputs. But the MO operation's operands only include the inputs, not
  // the DPS outputs. So we need to offset by the number of DPS outputs.
  if (inputParamsAttr) {
    // Get the number of DPS outputs to offset our index.
    unsigned numDpsOutputs = 0;
    if (auto dpsAttr = function->getAttrOfType<IntegerAttr>(
            KGEN::MOGGPreElab::kMOGGNumDPSOutputs))
      numDpsOutputs = dpsAttr.getInt();

    for (auto [idx, operand] : llvm::enumerate(moOp->getOperands())) {
      // The kernel argument index is offset by the number of DPS outputs.
      unsigned kernelArgIdx = idx + numDpsOutputs;
      if (kernelArgIdx >= inputParamsAttr.size())
        break;

      auto opaqueType = dyn_cast<MO::OpaqueType>(operand.getType());
      if (!opaqueType)
        continue;

      auto kernelParamDict =
          dyn_cast<DictionaryAttr>(inputParamsAttr[kernelArgIdx]);
      inferFromOpaque(kernelParamDict, opaqueType);
    }
  }

  // Infer parameters from results.
  if (!resultParamsAttr)
    return;

  for (auto [idx, result] : llvm::enumerate(moOp->getResults())) {
    if (idx >= resultParamsAttr.size())
      break;

    auto opaqueType = dyn_cast<MO::OpaqueType>(result.getType());
    if (!opaqueType)
      continue;

    auto kernelParamDict = dyn_cast<DictionaryAttr>(resultParamsAttr[idx]);
    inferFromOpaque(kernelParamDict, opaqueType);
  }
}

MOGGExpKernelBuilder::MOGGExpKernelBuilder(
    Operation *op, OpBuilder &insertPoint, bool forceSync,
    MOGG::MojoLibraryAnalysis &mojoLibraryAnalysis)
    : moOp(op), graphOp(op->getParentOfType<MO::GraphOp>()),
      builder(insertPoint), ctx(op->getContext()), loc(op->getLoc()),
      mojoLibraryAnalysis(mojoLibraryAnalysis) {

  // Most ops are async, with specific exceptions:
  // - forceSync overrides everything (unsafe, for debugging only)
  // - TransposeOp is always sync
  //   TODO(GEX-2569): make async after enforcing high water mark
  // - LayoutTransformOp is sync on CPU
  //
  // All GPU ops are async to ensure that the primitives are non-blocking (see
  // KeyConceptsInGraphRT.md).
  // Note that `cuModuleLoadDataEx` and sync kernel launch can block in general.
  // Rely on sequence fusion to amortize the associated primitive launch
  // overhead, instead of synchronous primitive execution.
  bool isSyncRequired = forceSync || isa<MO::TransposeOp>(op) ||
                        (MO::isOnHost(op) && isa<MO::LayoutTransformOp>(op));
  isAsync = !isSyncRequired;

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
      ret.push_back(MOGG::AsScalarOp::create(
          builder, v.getLoc(),
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
    ret.push_back(MOGG::AsTupleOp::create(
        builder, valuesThatMatchVariadics.front().getLoc(),
        builder.getTupleType(tupleTypes), valuesThatMatchVariadics));
  }

  return ret;
}

void addContextIfPresent(Operation *op, SmallVector<Value> &inputs,
                         const KGEN::MOGGPreElab::MojoKernelAdaptor &adaptor,
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
      contexts.push_back(MOGG::DeviceContextPlaceholderOp::create(
          builder, loc, MOGG::ContextType::get(ctx), deviceAttr));
    }

    inputs.push_back(MOGG::AsTupleOp::create(
        builder, loc,
        builder.getTupleType(
            SmallVector<Type>(contexts.size(), MOGG::ContextType::get(ctx))),
        contexts));
  } else {
    inputs.push_back(MOGG::DeviceContextPlaceholderOp::create(
        builder, loc, MOGG::ContextType::get(ctx),
        MO::getDeviceOrDefaultAny(op)));
  }
}

KernelCallInterface MOGGExpKernelBuilder::lowerToElementwiseKernel(
    KernelOp kernelOp, const OperandMapping &mapping,
    const KGEN::MOGGPreElab::MojoKernelAdaptor &adaptor) {
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

  auto bindOp = BindLambdaOp::create(builder, loc, tensorType, fusibleTensors,
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
  auto callElementwise = CallElementwiseOp::create(
      builder, loc, resultValTp, adaptor.registeredOpName, outlinedArgs);
  OutputOp::create(builder, loc, callElementwise->getResults());
  builder.setInsertionPointAfter(bindOp);

  SmallVector<Value> contextValue;
  addContextIfPresent(moOp, contextValue, adaptor, builder, true);
  auto ret = CallForeachOp::create(builder, loc, bindOp.getOutput(),
                                   outputValue, contextValue);

  return ret;
}

ErrorOr<KernelCallInterface> MOGGExpKernelBuilder::lowerToViewKernel(
    KernelOp kernelOp, const OperandMapping &mapping,
    const KGEN::MOGGPreElab::MojoKernelAdaptor &adaptor) {
  if (kernelOp->getNumResults() != 1)
    return Error("Expected view op to have exactly one result");

  auto execBody = &kernelOp.getExecBody().front();
  Value outputVal = kernelOp.getOutputPlaceholder()->getResult(0);
  SmallVector<Value> inputs(execBody->getArguments().begin(),
                            execBody->getArguments().end());

  if (isa<MO::Static::ReshapeOp, MO::Static::BroadcastToOp>(moOp)) {
    inputs.push_back(MOGG::TensorGetShapeOp::create(
        builder, loc, TypeAttr::get(outputVal.getType())));
  }

  inputs = convertTuplesIfNeeded(
      convertScalarsIfNeeded(inputs, mapping.operandMapping, builder),
      mapping.operandMapping, builder);
  auto view = MOGG::CallViewOp::create(builder, loc, outputVal.getType(),
                                       adaptor.registeredOpName, inputs);
  // MaterializeOp always takes an input tensor (with view updated by the
  // mogg.call.view above).
  SmallVector<Value> contextValue;
  addContextIfPresent(moOp, contextValue, adaptor, builder, true);
  MOGG::CallMaterializeOp::create(builder, loc, view->getResult(0), outputVal,
                                  contextValue);

  return view;
}

KernelCallInterface MOGGExpKernelBuilder::lowerToNormalKernel(
    KernelOp kernelOp, const OperandMapping &mapping,
    const KGEN::MOGGPreElab::MojoKernelAdaptor &adaptor) {
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
    Value bound =
        BindLambdaOp::create(builder, loc, concreteInput.getType(),
                             concreteInput, BindLambdaKind::InputLambda);
    inputs[idx] = bound;
  }

  for (const auto &[idx, arg] : mapping.resultMapping) {
    if (!arg.isFusedTensorType())
      continue;
    Value concreteOutput = outputs[idx];
    Value bound =
        BindLambdaOp::create(builder, loc, concreteOutput.getType(),
                             concreteOutput, BindLambdaKind::OutputLambda);
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
  bool isThrows = adaptor.executeFunction->mojoCode.isThrows();

  return CallExecuteOp::create(builder, loc, adaptor.registeredOpName,
                               regRetTypeAttr, isThrows, inputs, outputs);
}

KernelCallInterface MOGGExpKernelBuilder::lowerToShapeKernel(
    KernelOp kernelOp, const OperandMapping &mapping,
    const KGEN::MOGGPreElab::MojoKernelAdaptor &adaptor) {
  auto execBody = &kernelOp.getExecBody().front();
  SmallVector<Value> inputs(execBody->getArguments().begin(),
                            execBody->getArguments().end());

  // Get the shape function from the adaptor
  ASSERT_STREAM(adaptor.shapeFunction.has_value(),
                "[MO-TO-MOGG] Shape function adaptor must have shape function");

  // Determine the return type based on output param decls
  auto shapeFuncOp = cast<MO::InvokeShapeFuncOp>(moOp);
  const uint64_t rank = shapeFuncOp.getOutputParamDecls().size();
  Type retType = MOGG::ShapeFuncResultType::get(builder.getContext(), rank);

  // Convert scalars and tuples using existing machinery
  inputs = convertTuplesIfNeeded(
      convertScalarsIfNeeded(inputs, mapping.operandMapping, builder),
      mapping.operandMapping, builder);

  bool isThrows = adaptor.shapeFunction->mojoCode.isThrows();

  return MOGG::CallShapeOp::create(
      builder, loc, builder.getStringAttr(adaptor.registeredOpName),
      TypeAttr::get(retType), isThrows ? builder.getUnitAttr() : UnitAttr(),
      inputs);
}

static ErrorOr<NamedAttribute>
getValidatedParameter(const NamedAttribute &attr) {
  auto value = attr.getValue();

  // Validate supported attribute types.
  if (!isa<IntegerAttr, StringAttr, TypeAttr, DenseIntElementsAttr>(
          attr.getValue())) {
    return Error(llvm::formatv("Unsupported typed attribute {}: {}",
                               attr.getName(), attr.getValue()));
  }

  // This particular attr type is sometimes used to represent shape parameters
  // of a kernel. (e.g. 'a_shape' in layout_transform_KN_to_KNkni).
  if (auto vectorAttr = dyn_cast<DenseIntElementsAttr>(value)) {
    SmallVector<Attribute> intAttrs;
    for (const auto &intVal : vectorAttr.getValues<APInt>()) {
      intAttrs.push_back(IntegerAttr::get(
          IntegerType::get(vectorAttr.getContext(), intVal.getBitWidth()),
          intVal));
    }
    value = ArrayAttr::get(vectorAttr.getContext(), intAttrs);
  }

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

    // Infer kernel parameters from opaque types in operands and results.
    // This resolves parameters like P0, P1 from the opaque type's parameter
    // dictionary (S0=4, S1=8 -> P0=4, P1=8).
    // Get the LIT kernel function to access its parameter annotations.
    KGEN::LIT::FnOp litFunction =
        mojoLibraryAnalysis.getRegisteredLITMojoExecuteFunctionFor(
            custom.getSymbol());
    inferOpaqueParameters(forwardedParameters, moOp, litFunction);

    return DictionaryAttr::get(moOp->getContext(), forwardedParameters);
  }

  if (auto layoutTransform = dyn_cast<MO::LayoutTransformOp>(moOp)) {
    if (auto params = layoutTransform.getKgenParamsAttr()) {
      for (NamedAttribute param : layoutTransform.getKgenParamsAttr()) {
        if (ErrorOr<NamedAttribute> validated = getValidatedParameter(param))
          return validated.takeError();
        else
          forwardedParameters.push_back(validated.takeValue());
      }
    }
    return DictionaryAttr::get(moOp->getContext(), forwardedParameters);
  }

  if (auto invokeShapeFunc = dyn_cast<MO::InvokeShapeFuncOp>(moOp)) {

    const uint64_t rank = invokeShapeFunc.getOutputParamDecls().size();
    forwardedParameters.push_back(NamedAttribute(
        builder.getStringAttr(KGEN::MOGGPreElab::kMOGGOutputRankParameterName),
        builder.getIndexAttr(rank)));

    if (auto params = invokeShapeFunc.getKgenParamsAttr()) {
      for (NamedAttribute param : invokeShapeFunc.getKgenParamsAttr()) {
        if (ErrorOr<NamedAttribute> validated = getValidatedParameter(param))
          return validated.takeError();
        else
          forwardedParameters.push_back(validated.takeValue());
      }
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

    // Build an ArrayAttr for the Mojo path.
    forwardedParameters.push_back(
        NamedAttribute(paramName, builder.getI64ArrayAttr(values)));
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

ErrorOr<KernelOp> MOGGExpKernelBuilder::lowerToKernelBase() {
  auto maybeAdaptor = getMojoKernelAdaptor(mojoLibraryAnalysis, moOp);
  if (maybeAdaptor.isError())
    return maybeAdaptor.takeError();

  KGEN::MOGGPreElab::MojoKernelAdaptor adaptor = maybeAdaptor.takeValue();

  // For shape functions, use the shape function instead of execute function
  // Shape functions return IndexList for dimension binding, not as SSA values,
  // so we need to clear the outputResult to skip result validation.
  auto *functionToMap = adaptor.shapeFunction.has_value()
                            ? &*adaptor.shapeFunction
                            : &*adaptor.executeFunction;

  // Shape functions bind dimensions via parameter declarations, not SSA
  // results, so we explicitly clear the outputResult to avoid validation
  // errors.
  if (adaptor.shapeFunction.has_value()) {
    auto modifiedFunction = *functionToMap;
    modifiedFunction.outputResult = std::nullopt;
    auto operandMappingOr =
        mapMOOpToKernelImplementation(moOp, modifiedFunction);
    if (operandMappingOr.isError())
      return operandMappingOr.takeError();

    const OperandMapping &operandMapping = operandMappingOr.get();
    return buildKernelOpImpl(adaptor, operandMapping);
  }

  auto operandMappingOr = mapMOOpToKernelImplementation(moOp, *functionToMap);
  if (operandMappingOr.isError())
    return operandMappingOr.takeError();

  const OperandMapping &operandMapping = operandMappingOr.get();
  return buildKernelOpImpl(adaptor, operandMapping);
}

// Helper to build the kernel op (common code for both paths)
ErrorOr<KernelOp> MOGGExpKernelBuilder::buildKernelOpImpl(
    KGEN::MOGGPreElab::MojoKernelAdaptor &adaptor,
    const OperandMapping &operandMapping) {

  // For shape functions, forward the outputParamDecls
  ArrayRef<KGEN::ParamDeclAttr> outputParamDecls = {};
  if (auto shapeFuncOp = dyn_cast<MO::InvokeShapeFuncOp>(moOp)) {
    outputParamDecls = shapeFuncOp.getOutputParamDecls();
  }

  auto kernelOp = KernelOp::create(builder, loc,
                                   /*results=*/moOp->getResultTypes(),
                                   /*operands=*/moOp->getOperands(),
                                   /*summary=*/getSummaryString(moOp),
                                   /*isAsync=*/builder.getBoolAttr(isAsync),
                                   /*device=*/getDevice(),
                                   /*outputParamDecls=*/outputParamDecls);
  builder.setInsertionPointToStart(&kernelOp.getExecBody().front());
  if (auto memEffectOp = dyn_cast<mlir::MemoryEffectOpInterface>(moOp)) {
    kernelOp.setHasWriteEffect(
        memEffectOp.hasEffect<mlir::MemoryEffects::Write>());
    kernelOp.setHasReadEffect(
        memEffectOp.hasEffect<mlir::MemoryEffects::Read>());
  }
  SmallVector<Type> dpsReturns;
  for (auto [idx, val] : operandMapping.resultMapping)
    if (val.isDpsOutput())
      dpsReturns.push_back(kernelOp->getResult(idx).getType());

  // Only need output place holder for dps returned value (they need to be
  // allocated by caller).
  if (!dpsReturns.empty())
    OutputPlaceholdersOp::create(builder, loc, dpsReturns);

  auto forwardedParametersOr = forwardParameters();
  if (forwardedParametersOr.isError())
    return forwardedParametersOr.takeError();

  ErrorOr<KernelCallInterface> kernelCallingOperation = nullptr;
  if (adaptor.shapeFunction.has_value()) {
    // This is a shape function
    kernelCallingOperation =
        lowerToShapeKernel(kernelOp, operandMapping, adaptor);
  } else if (adaptor.isElementwise()) {
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

ErrorOr<KernelOp> MOGGExpKernelBuilder::lowerToKernel() {
  return lowerToKernelBase();
}
} // namespace M::MOGG
