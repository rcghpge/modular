//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/Interfaces/EmitMojoState.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGInterfaces.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGOps.h"
#include "KGEN/KGENDialect/ParameterEvaluator.h"
#include "KGEN/MOGGPreElab/MOGGPreElabDecorators.h"

using namespace M;

namespace M::MOGG {

/// Remove parameters from inferByInputs that appear in the function's result
/// type but not in the actual input argument types.
///
/// This walks both the user result type and all input argument types (excluding
/// byref result/error slots), collects all ParamDeclRefAttr references from
/// each, and removes from inferByInputs any parameter names that appear in the
/// result type but not in any input argument type.
static void removeResultOnlyParams(KGEN::LIT::FnOp function,
                                   llvm::DenseSet<StringRef> &inferByInputs) {
  // Collect params from result type.
  SmallVector<KGEN::ParamDeclRefAttr> resultParamRefs;
  bool hasConstExpr = false;
  Type userResultType = function.getUserResultType();
  KGEN::collectParameterReferences(userResultType, resultParamRefs,
                                   hasConstExpr);
  llvm::DenseSet<StringRef> resultParams;
  for (auto paramRef : resultParamRefs)
    resultParams.insert(paramRef.getName().getValue());

  // Collect params from actual input arguments (excluding byref result slot).
  auto sig = function.getFuncTypeGenerator().getBody();
  ArrayRef<Type> argTypes = function.getArgumentTypes();
  if (sig.hasMemoryOnlyResult())
    argTypes = argTypes.drop_back();
  if (sig.isThrows())
    argTypes = argTypes.drop_back();

  llvm::DenseSet<StringRef> inputParams;
  for (Type argType : argTypes) {
    SmallVector<KGEN::ParamDeclRefAttr> argParamRefs;
    KGEN::collectParameterReferences(argType, argParamRefs, hasConstExpr);
    for (auto paramRef : argParamRefs)
      inputParams.insert(paramRef.getName().getValue());
  }

  // Remove any parameters that are only in the result type, not in inputs.
  SmallVector<StringRef> toRemove;
  for (StringRef paramName : inferByInputs) {
    if (resultParams.contains(paramName) && !inputParams.contains(paramName))
      toRemove.push_back(paramName);
  }
  for (StringRef paramName : toRemove)
    inferByInputs.erase(paramName);
}

LogicalResult EmitMojoState::lowerOpToMojo(Operation *op) {
  if (auto emitMojoInterface = dyn_cast<MOGG::EmitsMojoInterface>(op)) {
    if (enableOpTraceComments)
      builder.comment(llvm::formatv("Emitting Mojo for: {0}", op->getName()));

    if (emitMojoInterface.emitMojo(*this).failed()) {
      return emitMojoInterface->emitError(
          "Could not lower this operation to Mojo");
    }
    return success();
  } else if (isa<MOGG::OutputOp, MOGG::YieldOp>(op)) {
    // These ops don't need to be converted.
    return success();
  } else {
    return op->emitError("This operation doesn't implement EmitsMojoInterface");
  }
}

ErrorOr<llvm::SmallVector<MojoValue>> EmitMojoState::inferKernelParameters(
    KGEN::LIT::FnOp function,
    std::function<ErrorOr<MojoValue>(StringRef)> &&inferCallback) {
  // First get the set of all parameters that can be inferred from the inputs.
  llvm::DenseSet<StringRef> inferByInputs;

  DictionaryAttr defaultAttrs = function->getAttrOfType<DictionaryAttr>(
      KGEN::MOGGPreElab::kKernelDefaultParameterAttrName);

  if (auto inputParams = function->getAttrOfType<ArrayAttr>(
          KGEN::MOGGPreElab::kKernelValueParameterAttrName)) {
    for (auto inputParam : inputParams) {
      if (auto inputParamDict = dyn_cast<DictionaryAttr>(inputParam))
        for (auto param : inputParamDict)
          if (auto funcParamName =
                  dyn_cast<KGEN::ParamDeclRefAttr>(param.getValue())) {
            inferByInputs.insert(funcParamName.getName().getValue());
          }
    }
  }

  // TODO(GEX-2731): The annotate-kernels pass incorrectly marks result-only
  // parameters as input parameters. For example, shape functions return
  // !mogg.shape_func_result<rank>, where 'output_rank' is a parameter of the
  // result type. This gets incorrectly added to the input parameter inference
  // list by annotate-kernels, but should only be inferred from the result type.
  // Remove any such parameters that appear only in the result type but not in
  // actual input arguments. The proper fix would be to correct
  // annotate-kernels.
  removeResultOnlyParams(function, inferByInputs);

  auto isIOSpecType = [](Type ty) {
    auto structType = dyn_cast<KGEN::LIT::StructType>(ty);
    return structType && HardcodedMojoSymbols::isMojoIOSpecTypeName(
                             structType.getSymbol().getNestedReferences());
  };

  // Now go through all params and infer those that need to be inferred.
  SmallVector<MojoValue> inferredParams;
  for (auto param : function.getInputParams()) {
    StringRef paramName = param.getName().getValue();
    if (inferByInputs.contains(paramName))
      continue;

    // TODO(GEX-2623): IOSpec / origin parameters are currently removed from the
    // parameters list because in KGEN-land empty-strust params are removed
    // during LITToKGEN. But they are still present in EmitMojo path (and would
    // cause an error: failed to infer parameter). Instead just skip them for
    // now, they'll be inferred automatically anyway. This is quite brittle as
    // it assumes all IOSpec / origin are always inferred (which is true
    // fortunately today). This test could be removed once EmitMojo is complete.
    if (isa<KGEN::LIT::OriginType>(param.getType()) ||
        isIOSpecType(param.getType()))
      continue;

    ErrorOr<MojoValue> maybeVal = inferCallback(paramName);
    if (maybeVal.isError()) {
      if (defaultAttrs && defaultAttrs.contains(paramName)) {
        // If there is already a default value we can skip it.
        // TODO: It might be worth making the distinction between "we didn't
        // found a parameter" vs we found it but the logic to infer its value
        // failed. In the latter case we should still emit an error.
        continue;
      }
      function.emitError(Twine("Failed to infer kernel parameter `") +
                         paramName + "`: " + maybeVal.getError());
      return maybeVal.takeError();
    }

    inferredParams.push_back(MojoValue(paramName).equal(*maybeVal));
  }

  return inferredParams;
}

// Returns the trace name attribute, or none if summary is not set.
static MojoValue getTraceNameValue(StringAttr summaryAttr) {
  // GEX-979: pass in trace description for fused kernel without
  // invalidating cache.
  if (!summaryAttr)
    return MojoValue("").quote();

  // The summary string might contain some loc info (line number).
  // Eg: "mo.add:42".
  // If there is no loc info just return the original name.
  if (!summaryAttr.getValue().contains(':')) {
    return MojoValue(summaryAttr.getValue()).quote();
  }

  StringRef summary = summaryAttr.getValue();
  size_t len = summary.size();
  std::string res;
  for (size_t i = 0; i < len; ++i) {
    if (summary[i] == ':' && i + 1 < len && isdigit(summary[i + 1])) {
      // Find a loc info, ignore all digits.
      while (i + 1 < len && isdigit(summary[i + 1]))
        ++i;
    } else {
      res.push_back(summary[i]);
    }
  }

  return MojoValue(res).quote();
}

static MojoValue getLambasHaveFusionValue(mlir::Operation *kernelCallOp) {
  auto bindOps = kernelCallOp->getBlock()->getOps<MOGG::BindLambdaOp>();
  return llvm::any_of(bindOps, [](MOGG::BindLambdaOp bindOp) {
    // Fused in affine view does not set the flag.
    //
    // TODO: we should come up with a better way, it works because current
    // affine view fusion is very limited and it does not change the contiguity
    // of the buffer, which might not be the case after treating all view
    // operations as index transform.
    return llvm::any_of(bindOp.getLambda().getOps(), [](Operation &op) {
      return isa<MOGG::CallElementwiseOp>(&op);
    });
  });
}

ErrorOr<MojoValue> EmitMojoState::inferDefaultKernelParamValue(
    Operation *callOp, StringRef paramName, DictionaryAttr customParamDict) {

  auto inferFromMliParam = [](Attribute customParam) -> ErrorOr<MojoValue> {
    // Convert it to mojo
    // TODO: Convert custom mlir parameter to the right mojo parameter type is
    // hardcoded here and brittle. Besides the param name, we would also need
    // the type from the kernel infos to convert it correctly.
    if (auto strAttr = dyn_cast<StringAttr>(customParam))
      return MojoValue(strAttr.getValue()).quote();
    else if (auto boolAttr = dyn_cast<BoolAttr>(customParam))
      return boolAttr.getValue();
    else if (auto intAttr = dyn_cast<IntegerAttr>(customParam))
      return intAttr.getInt();
    else if (auto arrAttr = dyn_cast<ArrayAttr>(customParam);
             arrAttr && !arrAttr.empty() && isa<IntegerAttr>(arrAttr[0])) {
      SmallVector<int64_t> values;
      for (auto val : arrAttr)
        values.push_back(cast<IntegerAttr>(val).getInt());
      return EmitMojoState::getAsDimListMojoValue(values);
    } else
      return Error("Unsupported custom param MLIR type");
  };

  // Look at the provided list of kernel parameters first
  if (auto kernelCallOp = dyn_cast<KernelCallInterface>(callOp);
      kernelCallOp && kernelCallOp.getKernelParams()) {
    if (auto customParam = kernelCallOp.getKernelParams().get(paramName))
      return inferFromMliParam(customParam);
  }

  if (customParamDict) {
    if (auto customParam = customParamDict.get(paramName))
      return inferFromMliParam(customParam);
  }

  // Some special parameters are common to all kernels.
  if (paramName == KGEN::MOGGPreElab::kMOGGTargetParameterName)
    return MojoValue(MO::getDeviceOrDefaultAny(callOp).getLabel()).quote();
  if (paramName == KGEN::MOGGPreElab::kMOGGTraceNameParameterName) {
    auto kernelStubOp = callOp->getParentOfType<KernelStubOp>();
    return getTraceNameValue(kernelStubOp.getSummaryAttr());
  }
  if (paramName == KGEN::MOGGPreElab::kMOGGLambdasHaveFusionParameterName)
    return getLambasHaveFusionValue(callOp);

  return Error("Failed to infer parameter");
}

MojoValue EmitMojoState::getShapeAsIndexListMojoValue(MO::TensorType tensorTy,
                                                      ValueRange dynamicDims) {
  // TODO: Make sure this is actually valid for dynamic cases.
  // The dimensions operand only contains the dynamic ones in
  // MOGGTensorInitOp. So instead we get the static ones from the type
  // directly.

  SmallVector<MojoValue> mojoDims;
  unsigned nextDymDim = 0;
  for (TypedAttr dim : tensorTy.getDims()) {
    if (auto staticDim = dyn_cast<IntegerAttr>(dim)) {
      mojoDims.push_back(MojoValue(staticDim.getSInt()));
    } else {
      ASSERT_STREAM(nextDymDim < dynamicDims.size(),
                    "Missing dynamic dimension");
      mojoDims.push_back(getMojoValueFor(dynamicDims[nextDymDim++]));
    }
  }
  if (mojoDims.empty())
    mojoDims.push_back(MojoValue(static_cast<int64_t>(1)));

  MojoValue rankParam =
      MojoValue(static_cast<uint64_t>(tensorTy.getRankNonZero()));

  return MojoValue(HardcodedMojoSymbols::struct_IndexList)({rankParam},
                                                           mojoDims);
}

} // namespace M::MOGG
