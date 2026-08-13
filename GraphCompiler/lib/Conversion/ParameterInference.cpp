//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/Conversion/ParameterInference.h"
#include "GraphCompiler/MODialect/MODialect.h"
#include "GraphCompiler/MODialect/MOOps.h"
#include "GraphCompiler/MODialect/MOTypes.h"
#include "GraphCompiler/MOGGDialect/Analysis/KernelDeclAdaptor.h"
#include "GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GraphCompiler/MOGGDialect/Support/MojoIdentifiers.h"
#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/KGENParameters.h"
#include "KGEN/KGENDialect/KGENPogUtils.h"
#include "KGEN/LITDialect/LITTypes.h"
#include "KGEN/LITDialect/LITUtils.h"

using namespace mlir;

namespace M {

DictionaryAttr inferParameters(Operation *op, OpBuilder &builder) {
  if (auto custom = dyn_cast<MO::CustomOp>(op)) {
    if (DictionaryAttr params = custom.getParameters())
      return params;
    return builder.getDictionaryAttr({});
  }

  SmallVector<NamedAttribute> params;
  for (NamedAttribute attr : op->getAttrs()) {
    Attribute value = attr.getValue();
    if (isa<M::DeviceRefAttr, KGEN::ParamDeclArrayAttr>(value))
      continue;
    if (auto typeAttr = dyn_cast<TypeAttr>(value)) {
      params.push_back(NamedAttribute(
          attr.getName(),
          KGEN::DTypeConstantAttr::get(
              value.getContext(), MO::getDTypeOrInvalid(typeAttr.getValue()))));
      continue;
    }
    if (isa<TypedAttr>(value))
      params.push_back(attr);
  }
  return builder.getDictionaryAttr(params);
}

} // namespace M

namespace M::MOGG {

/// Remove parameters from inferByInputs that appear in the function's result
/// type but not in the actual input argument types.
static void removeResultOnlyParams(KGEN::LIT::FnTypeGeneratorType sig,
                                   llvm::DenseSet<StringRef> &inferByInputs) {
  // Collect params from result type.
  SmallVector<KGEN::ParamDeclRefAttr> resultParamRefs;
  bool hasConstExpr = false;
  size_t requiredSignatureDepth = 0;
  Type userResultType = sig.getUserResultType();
  KGEN::collectParameterReferences(userResultType, resultParamRefs,
                                   hasConstExpr, requiredSignatureDepth);
  llvm::DenseSet<StringRef> resultParams;
  for (auto paramRef : resultParamRefs)
    resultParams.insert(paramRef.getName().getValue());

  // Collect params from actual input arguments (excluding byref result slot).
  ArrayRef<Type> argTypes = sig.getArguments();
  if (sig.hasMemoryOnlyResult())
    argTypes = argTypes.drop_back();
  if (sig.isThrows())
    argTypes = argTypes.drop_back();

  llvm::DenseSet<StringRef> inputParams;
  for (Type argType : argTypes) {
    SmallVector<KGEN::ParamDeclRefAttr> argParamRefs;
    KGEN::collectParameterReferences(argType, argParamRefs, hasConstExpr,
                                     requiredSignatureDepth);
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

/// Core parameter-inference logic shared by all inferKernelParameters
/// overloads.
static ErrorOr<InferredParameters>
inferParameters(ArrayRef<mlir::Attribute> fnLevelParams,
                ArrayRef<mlir::Attribute> structLevelParams,
                mlir::ArrayAttr inputParamMappings,
                mlir::DictionaryAttr defaultAttrs,
                KGEN::LIT::FnTypeGeneratorType sig,
                std::function<ErrorOr<MojoValue>(StringRef)> &inferCallback) {

  // Collect the set of params inferable from inputs by walking the function's
  // argument types directly. A param is inferable when an argument type
  // references it -- either as a named `ParamDeclRefAttr` or as a depth-0
  // `ParamIndexRefAttr` (the form elaborated bytecode uses for bound
  // parameter values).
  //
  // For IndexRefs, `*(0, K)` indexes into the generator's *full* parameter
  // list, which Mojo lays out as `structLevelParams + fnLevelParams` (struct
  // params first). `fnLevelParams` alone (from
  // `KernelFunctionAttr::params` / `FnOp::getInputParams()`) is just the
  // fn-level subset, so we build a merged lookup table to recover the right
  // name. A wrong lookup would silently misreport e.g. "freq_dtype" as
  // inferable when the kernel really references "dtype" -- breaking parameter
  // inference for kernels with both struct-level and fn-level params.
  llvm::SmallVector<mlir::Attribute> mergedParams;
  mergedParams.reserve(structLevelParams.size() + fnLevelParams.size());
  mergedParams.append(structLevelParams.begin(), structLevelParams.end());
  mergedParams.append(fnLevelParams.begin(), fnLevelParams.end());

  llvm::DenseSet<StringRef> inferByInputs;
  std::function<void(mlir::Attribute)> walkAttr;
  std::function<void(mlir::Type)> walkType;
  walkAttr = [&](mlir::Attribute attr) {
    if (!attr)
      return;
    if (auto sugar = dyn_cast<KGEN::SugarAttr>(attr)) {
      walkAttr(sugar.getCanonical());
      return;
    }
    if (auto declRef = dyn_cast<KGEN::ParamDeclRefAttr>(attr)) {
      inferByInputs.insert(declRef.getName().getValue());
      return;
    }
    if (auto indexRef = dyn_cast<KGEN::ParamIndexRefAttr>(attr)) {
      // Depth-0 references the fn's enclosing generator scope, which is the
      // merged `structLevelParams + fnLevelParams` list.
      if (indexRef.getDepth() == 0 &&
          indexRef.getIndex() < mergedParams.size()) {
        auto pd = cast<KGEN::ParamDeclAttr>(mergedParams[indexRef.getIndex()]);
        inferByInputs.insert(pd.getName().getValue());
      }
      return;
    }
    attr.walkImmediateSubElements([&](mlir::Attribute a) { walkAttr(a); },
                                  [&](mlir::Type t) { walkType(t); });
  };
  walkType = [&](mlir::Type type) {
    if (!type)
      return;
    type.walkImmediateSubElements([&](mlir::Attribute a) { walkAttr(a); },
                                  [&](mlir::Type t) { walkType(t); });
  };
  for (size_t i = 0, n = sig.getNumArguments(); i < n; ++i)
    walkType(sig.getArgument(i));

  // Remove result-only parameters from the inferByInputs set.
  removeResultOnlyParams(sig, inferByInputs);

  auto isIOSpecType = [](Type ty) {
    auto structType = dyn_cast<KGEN::LIT::StructType>(ty);
    return structType &&
           isMojoIOSpecTypeName(structType.getSymbol().getNestedReferences());
  };

  auto inferParametersInner =
      [&](ArrayRef<mlir::Attribute> params,
          SmallVector<MojoValue> &inferredParams) -> ErrorOrSuccess {
    for (auto paramAttr : params) {
      auto param = cast<KGEN::ParamDeclAttr>(paramAttr);
      StringRef paramName = param.getName().getValue();

      if (isa<KGEN::LIT::OriginType>(param.getType()) ||
          isIOSpecType(param.getType()))
        continue;
      if (paramName.contains("."))
        continue;

      ErrorOr<MojoValue> maybeVal = inferCallback(paramName);
      if (inferByInputs.contains(paramName) && maybeVal.isError())
        continue;

      if (maybeVal.isError()) {
        if (defaultAttrs && defaultAttrs.contains(paramName))
          continue;
        return maybeVal.takeError();
      }
      inferredParams.push_back(MojoValue(paramName).equal(*maybeVal));
    }
    return success();
  };

  InferredParameters ret;
  auto errOr = inferParametersInner(fnLevelParams, ret.functionParameters);
  if (errOr.isError())
    return errOr.takeError();
  errOr = inferParametersInner(structLevelParams, ret.structParameters);
  if (errOr.isError())
    return errOr.takeError();
  return ret;
}

// Build a {param-name -> UnitAttr} dictionary listing fn-level params with
// a declared default value. Sourced from the function's `PogListAttr`
// metadata; needed by `inferParameters` to know which missing inferences
// are tolerable.
static mlir::DictionaryAttr
computeDefaultParamsDict(KernelDeclFunctionAdaptor *fnAdaptor) {
  if (!fnAdaptor || !fnAdaptor->attr.getParams() ||
      fnAdaptor->attr.getParams().empty())
    return nullptr;
  auto fnParams = fnAdaptor->attr.getParams().getValue();
  KGEN::PogListAttr pogs = fnAdaptor->attr.getSignature().getParamListAttrs();
  size_t numImplicit =
      fnAdaptor->attr.getSignature().getNumImplicitOriginDecls();
  size_t totalPogs = pogs.getPogs().size();
  // Fn-level params are at the end of the merged pog list (struct-level
  // params come first).
  size_t fnStart =
      totalPogs >= fnParams.size() ? totalPogs - fnParams.size() : 0;
  size_t fnEnd =
      fnParams.size() > numImplicit ? fnParams.size() - numImplicit : 0;

  mlir::Builder builder(fnAdaptor->attr.getSignature().getContext());
  mlir::NamedAttrList defaultNames;
  for (size_t fnIdx = 0; fnIdx < fnEnd; ++fnIdx) {
    size_t pogIdx = fnStart + fnIdx;
    if (pogIdx >= totalPogs || !pogs.getDefault(pogIdx))
      continue;
    auto paramDecl = mlir::cast<KGEN::ParamDeclAttr>(fnParams[fnIdx]);
    defaultNames.append(
        mlir::StringAttr::get(
            fnAdaptor->attr.getSignature().getContext(),
            KGEN::demangleParameterName(paramDecl.getName().strref())),
        builder.getUnitAttr());
  }
  if (defaultNames.empty())
    return nullptr;
  return mlir::DictionaryAttr::get(fnAdaptor->attr.getSignature().getContext(),
                                   defaultNames.getAttrs());
}

ErrorOr<InferredParameters> inferKernelParameters(
    KernelDeclAdaptor &adaptor, llvm::StringRef funcLabel,
    std::function<ErrorOr<MojoValue>(StringRef)> &&inferCallback) {
  KernelDeclFunctionAdaptor *fnAdaptor =
      const_cast<KernelDeclFunctionAdaptor *>(
          adaptor.getFunctionByLabel(funcLabel));
  ASSERT_STREAM(fnAdaptor,
                "Function label '" << funcLabel << "' not found in adaptor");

  // Pull fn-level params from the function adaptor itself. Each variant
  // (execute / elementwise / shape / update_view) keeps its own param list
  // because their signatures can differ (e.g. elementwise typically omits
  // `target`/`_trace_name`). For an elementwise label that falls back to the
  // synthesised-from-elementwise path, the adaptor's `params` already reflects
  // the elementwise variant.
  ArrayRef<mlir::Attribute> fnParams;
  if (fnAdaptor->attr.getParams())
    fnParams = fnAdaptor->attr.getParams().getValue();

  ArrayRef<mlir::Attribute> structParams;
  if (adaptor.structParams)
    structParams = adaptor.structParams.getValue();

  // Default-params dict is computed from the *execute* fn for parity with
  // the WIP behavior; defaults are declared on the execute signature.
  mlir::DictionaryAttr defaultParams = computeDefaultParamsDict(
      adaptor.executeFunction ? &*adaptor.executeFunction : nullptr);
  return inferParameters(fnParams, structParams, fnAdaptor->attr.getArgParams(),
                         defaultParams, fnAdaptor->attr.getSignature(),
                         inferCallback);
}

ErrorOr<InferredParameters> inferKernelParameters(
    PrimitiveDeclOp primDecl,
    std::function<ErrorOr<MojoValue>(StringRef)> &&inferCallback) {
  // Primitives have no struct-level params. arg-param mappings let
  // inferParameters tolerate failures for parameters that are inferable from
  // input argument types at the Mojo call site.
  MOGG::KernelFunctionAttr fn = primDecl.getFunction();
  return inferParameters(fn.getParams().getValue(),
                         /*structLevelParams=*/{}, fn.getArgParams(),
                         /*defaultAttrs=*/nullptr, fn.getSignature(),
                         inferCallback);
}

ErrorOr<InferredParameters> inferKernelParameters(
    ShapeFunctionDeclOp shapeDecl,
    std::function<ErrorOr<MojoValue>(StringRef)> &&inferCallback) {
  // Standalone shape functions have no struct-level params.
  MOGG::KernelFunctionAttr fn = shapeDecl.getFunction();
  return inferParameters(fn.getParams().getValue(),
                         /*structLevelParams=*/{}, fn.getArgParams(),
                         /*defaultAttrs=*/nullptr, fn.getSignature(),
                         inferCallback);
}

} // namespace M::MOGG
