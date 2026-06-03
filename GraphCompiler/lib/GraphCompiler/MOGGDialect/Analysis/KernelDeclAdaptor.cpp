//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/GraphCompiler/MOGGDialect/Analysis/KernelDeclAdaptor.h"
#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/LITDialect/LITAttrs.h"
#include "KGEN/LITDialect/LITTypes.h"
#include "KGEN/LITDialect/LITUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FormatVariadic.h"

using namespace M;
using namespace M::MOGG;
using namespace M::KGEN;

namespace {

// Symbol path arrays for recognising known struct types. Components live in
// MojoIdentifiers.h (`kPackage*`, `kLeaf*`).
static constexpr std::array<llvm::StringLiteral, 3> kMaxManagedTensorSlice = {
    kPackageExtensibility, kPackageManagedTensorSlice, kLeafManagedTensorSlice};
static constexpr std::array<llvm::StringLiteral, 3> kMaxVariadicTensors = {
    kPackageExtensibility, kPackageManagedTensorSlice, kLeafVariadicTensors};
static constexpr std::array<llvm::StringLiteral, 3>
    kMaxFusedInputVariadicTensors = {kPackageExtensibility,
                                     kPackageManagedTensorSlice,
                                     kLeafFusedInputVariadicTensors};
static constexpr std::array<llvm::StringLiteral, 3>
    kMaxFusedOutputVariadicTensors = {kPackageExtensibility,
                                      kPackageManagedTensorSlice,
                                      kLeafFusedOutputVariadicTensors};

/// Check whether a symbol has the given root and leaf references, ignoring
/// any intermediate nested references. This mirrors MOGG::isXType
/// and is appropriate when the package path depth may vary (e.g.
/// DeviceContext lives under std::gpu::host but can also be re-exported more
/// shallowly).
static bool symbolMatchesRootAndLeaf(mlir::SymbolRefAttr symbol,
                                     llvm::StringLiteral root,
                                     llvm::StringLiteral leaf) {
  return symbol.getRootReference() == root && symbol.getLeafReference() == leaf;
}

/// Build a "root.intermediate...leaf" opaque type string from a symbol ref.
/// Every segment is joined by '.', matching Mojo's own qualified-name syntax
/// so that consumers (e.g. mlirOpaqueTypeToMojoType) can use the string
/// directly as a Mojo identifier with no further rewriting. The leaf segment
/// can be recovered by splitting on the last '.'.
static mlir::StringAttr buildOpaqueSymbol(mlir::MLIRContext *ctx,
                                          mlir::SymbolRefAttr sym) {
  std::string result = sym.getRootReference().strref().str();
  for (auto ref : sym.getNestedReferences())
    result += "." + ref.getValue().str();
  return mlir::StringAttr::get(ctx, result);
}

/// Check whether a symbol matches a dotted-path array (root + nested refs).
static bool symbolMatches(mlir::SymbolRefAttr symbol,
                          llvm::ArrayRef<llvm::StringLiteral> path) {
  if (symbol.getNestedReferences().size() != path.size() - 1)
    return false;
  if (!symbol.getRootReference().strref().contains(path.front()))
    return false;
  for (auto [i, ref] : llvm::enumerate(symbol.getNestedReferences())) {
    if (!ref.getValue().contains(path[i + 1]))
      return false;
  }
  return true;
}

/// Strip LIT::RefType sugar and return the underlying StructType, or null.
static KGEN::LIT::StructType getAsStructType(mlir::Type t) {
  if (auto ref = KGEN::sugarDynCast<KGEN::LIT::RefType>(t))
    return KGEN::sugarDynCast<KGEN::LIT::StructType>(ref.getElementType());
  return KGEN::sugarDynCast<KGEN::LIT::StructType>(t);
}

/// Extract the IOSpec encoded in the mut/input parameter values of a
/// ManagedTensorSlice or VariadicTensors struct type.  Returns std::nullopt
/// when the parameters are not yet bound.
static std::optional<IOSpec>
getIOSpecFromStructType(KGEN::LIT::StructType structType) {
  // Fused variadic types carry their IOSpec in the symbol name alone.
  if (symbolMatches(structType.getSymbol(), kMaxFusedInputVariadicTensors))
    return IOSpec::FusedInputTensor;
  if (symbolMatches(structType.getSymbol(), kMaxFusedOutputVariadicTensors))
    return IOSpec::FusedOutputTensor;

  // All other tensors encode mut and input as the first two param values.
  bool isManagedTensor =
      symbolMatches(structType.getSymbol(), kMaxManagedTensorSlice);
  bool isVariadicTensor =
      symbolMatches(structType.getSymbol(), kMaxVariadicTensors);
  if (!isManagedTensor && !isVariadicTensor)
    return std::nullopt;

  auto params = structType.getParamValues();
  if (params.size() < 2)
    return std::nullopt;

  mlir::TypedAttr mutAttr =
      KGEN::getCanonicalAttr(mlir::cast<mlir::TypedAttr>(params[0]));
  mlir::TypedAttr rawInputAttr = mlir::cast<mlir::TypedAttr>(params[1]);

  // mut is a Bool struct: { value: scalar<bool> }.
  auto mutStruct = KGEN::sugarDynCast<KGEN::LIT::LITStructAttr>(mutAttr);
  if (!mutStruct || mutStruct.getValues().empty())
    return std::nullopt;
  auto [_1, mutValueAttr] = mutStruct.getValues().front();
  auto mutBoolAttr = KGEN::sugarDynCast<KGEN::SIMDAttr>(mutValueAttr);
  if (!mutBoolAttr)
    return std::nullopt;
  bool mut = mutBoolAttr.getAsBool();

  int64_t input = kIOSpecIOUnknown;

  // Primary path: in the primary module (no dep elaboration), the IO param is
  // a SugarAttr(MemberAlias) whose member name directly encodes the IO variant.
  // getCanonicalAttr on a MemberAlias returns the canonical of the expanded
  // form (a ParamOperatorAttr like IO(3)), not a LITStructAttr, so we must
  // check the raw attr before canonicalisation.
  if (auto ioSugar = mlir::dyn_cast<KGEN::SugarAttr>(rawInputAttr)) {
    if (ioSugar.getKind() == KGEN::SugarKind::MemberAlias) {
      llvm::StringRef name = ioSugar.getMemberName().strref();
      if (name == kIoOutput)
        input = kIOSpecIOOutput;
      else if (name == kIoInput)
        input = kIOSpecIOInput;
      else if (name == kIoFusedInput)
        input = kIOSpecIOFusedInput;
      else if (name == kIoFusedOutput)
        input = kIOSpecIOFusedOutput;
      else if (name == kIoFusedComputeOutput)
        input = kIOSpecIOFusedComputeOutput;
    }
  }

  // Fallback path: in elaborated bytecode the IO param is a fully-resolved
  // LITStructAttr { value: SIMDSize { value: i64 } } (two levels of nesting).
  if (input == kIOSpecIOUnknown) {
    mlir::TypedAttr inputAttr = KGEN::getCanonicalAttr(rawInputAttr);
    auto inputStruct = KGEN::sugarDynCast<KGEN::LIT::LITStructAttr>(inputAttr);
    if (inputStruct && !inputStruct.getValues().empty()) {
      auto [_2, inputValueAttr] = inputStruct.getValues().front();
      auto innerStruct =
          KGEN::sugarDynCast<KGEN::LIT::LITStructAttr>(inputValueAttr);
      if (innerStruct && !innerStruct.getValues().empty()) {
        auto [_3, inputIntValueAttr] = innerStruct.getValues().front();
        auto inputIntAttr =
            KGEN::sugarDynCast<mlir::IntegerAttr>(inputIntValueAttr);
        if (inputIntAttr)
          input = inputIntAttr.getValue().getSExtValue();
      }
    }
  }

  if (input == kIOSpecIOUnknown)
    return std::nullopt;

  if (mut == kIOSpecMutable && input == kIOSpecIOOutput)
    return IOSpec::OutputTensor;
  if (mut == kIOSpecImmutable && input == kIOSpecIOInput)
    return IOSpec::InputTensor;
  if (mut == kIOSpecMutable && input == kIOSpecIOInput)
    return IOSpec::MutableInputTensor;
  if (mut == kIOSpecImmutable && input == kIOSpecIOFusedInput)
    return IOSpec::FusedInputTensor;
  if (mut == kIOSpecMutable && input == kIOSpecIOFusedOutput)
    return IOSpec::FusedOutputTensor;
  if (mut == kIOSpecMutable && input == kIOSpecIOFusedComputeOutput)
    return IOSpec::FusedComputeOutputTensor;
  return std::nullopt;
}

/// Classify a single argument type into a KernelDeclOperandVariant by
/// introspecting the LIT StructType symbol and parameter values.
static KernelDeclOperandVariant classifyArgType(mlir::MLIRContext *ctx,
                                                mlir::Type argType,
                                                bool promoteSIMDToFusedTensor) {
  KGEN::LIT::StructType structType = getAsStructType(argType);
  if (!structType)
    return KernelDeclUnsupportedAdaptor{};

  auto sym = structType.getSymbol();

  // --- Tensors ---
  if (symbolMatches(sym, kMaxManagedTensorSlice)) {
    auto ioSpec = getIOSpecFromStructType(structType);
    if (!ioSpec)
      return KernelDeclOpaqueAdaptor{buildOpaqueSymbol(ctx, sym)};
    return KernelDeclTensorAdaptor{*ioSpec};
  }

  if (symbolMatches(sym, kMaxVariadicTensors)) {
    auto ioSpec = getIOSpecFromStructType(structType);
    if (!ioSpec)
      return KernelDeclOpaqueAdaptor{buildOpaqueSymbol(ctx, sym)};
    return KernelDeclVariadicTensorAdaptor{*ioSpec};
  }

  if (symbolMatches(sym, kMaxFusedInputVariadicTensors))
    return KernelDeclVariadicTensorAdaptor{IOSpec::FusedInputTensor};

  if (symbolMatches(sym, kMaxFusedOutputVariadicTensors))
    return KernelDeclVariadicTensorAdaptor{IOSpec::FusedOutputTensor};

  // --- Scalar ---
  // Use root+leaf matching because SIMD may be re-exported at different depths
  // (e.g. @std::@builtin::@SIMD vs @std::@SIMD); mirrors the old path's
  // "std::SIMD" string comparison.
  if (symbolMatchesRootAndLeaf(sym, kPackageStd, kLeafSIMD)) {
    if (promoteSIMDToFusedTensor)
      return KernelDeclTensorAdaptor{IOSpec::FusedInputTensor};
    return KernelDeclScalarAdaptor{};
  }

  // --- TileTensor ---
  if (sym.getLeafReference().strref() == kLeafTileTensor)
    return KernelDeclTileTensorAdaptor{};

  // --- Device context ---
  // Use root+leaf matching because DeviceContext lives under std::gpu::host
  // (depth varies across package exports).
  if (symbolMatchesRootAndLeaf(sym, kPackageStd, kLeafDeviceContext))
    return KernelDeclContextPtrAdaptor{};

  if (symbolMatchesRootAndLeaf(sym, kPackageStd, kLeafDeviceContextList))
    return KernelDeclContextPtrListAdaptor{};

  // --- Device graph builder (launch arg of build_graph) ---
  if (symbolMatchesRootAndLeaf(sym, kPackageStd, kLeafDeviceGraphBuilder))
    return KernelDeclGraphBuilderAdaptor{};

  // --- Opaque: unrecognised struct ---
  return KernelDeclOpaqueAdaptor{buildOpaqueSymbol(ctx, sym)};
}

/// Count the number of leading DPS output arguments (output-IOSpec tensors
/// that come before any input or non-tensor argument).
static size_t countDPSOutputs(KGEN::LIT::FnTypeGeneratorType sig) {
  size_t count = 0;
  for (size_t i = 0, n = sig.getNumArguments(); i < n; ++i) {
    if (sig.getArgConvention(i) == KGEN::ArgConvention::ByRefResult)
      break;
    KGEN::LIT::StructType structType = getAsStructType(sig.getArgument(i));
    if (!structType)
      break;
    auto ioSpec = getIOSpecFromStructType(structType);
    if (!ioSpec || !isOutputIOSpec(*ioSpec))
      break;
    ++count;
  }
  return count;
}

} // namespace

//===----------------------------------------------------------------------===//
// KernelDeclFunctionAdaptor
//===----------------------------------------------------------------------===//

KernelDeclFunctionAdaptor::KernelDeclFunctionAdaptor(
    KGEN::LIT::FnTypeGeneratorType sig) {
  mlir::MLIRContext *ctx = sig.getContext();
  size_t numArgs = sig.getNumArguments();

  // Detect whether the result is passed as a trailing by-ref argument.
  bool resultAsArgument = false;
  for (size_t i = 0; i < numArgs; ++i) {
    if (sig.getArgConvention(i) == KGEN::ArgConvention::ByRefResult) {
      resultAsArgument = true;
      break;
    }
  }
  bool isThrow = sig.isThrows();
  size_t numByRefArgs = isThrow ? 2 : (resultAsArgument ? 1 : 0);
  size_t endOfInputArgs = numArgs - numByRefArgs;
  size_t numDPSOutputs = countDPSOutputs(sig);

  // Output (DPS) arguments come first.
  for (size_t i = 0; i < numDPSOutputs; ++i) {
    KernelDeclOperandAdaptor op;
    op.sourceDescriptor = KernelDeclOperandAdaptor::SourceDescriptor{
        sig.getArgName(i), i, /*isByRefResult=*/false};
    op.underlyingType =
        classifyArgType(ctx, sig.getArgument(i), /*promoteSIMD=*/false);
    outputArguments.push_back(std::move(op));
  }

  // Input arguments follow.
  for (size_t i = numDPSOutputs; i < endOfInputArgs; ++i) {
    KernelDeclOperandAdaptor op;
    op.sourceDescriptor = KernelDeclOperandAdaptor::SourceDescriptor{
        sig.getArgName(i), i, /*isByRefResult=*/false};
    op.underlyingType =
        classifyArgType(ctx, sig.getArgument(i), /*promoteSIMD=*/false);
    inputArguments.push_back(std::move(op));
  }

  // Optional result: either passed by reference (last arg) or as a value.
  if (resultAsArgument && numByRefArgs > 0) {
    size_t byrefIdx = endOfInputArgs + numByRefArgs - 1;
    KernelDeclOperandAdaptor op;
    op.sourceDescriptor = KernelDeclOperandAdaptor::SourceDescriptor{
        sig.getArgName(byrefIdx), byrefIdx, /*isByRefResult=*/true};
    op.underlyingType =
        classifyArgType(ctx, sig.getArgument(byrefIdx), /*promoteSIMD=*/false);
    outputResult = std::move(op);
  } else if (!resultAsArgument) {
    mlir::Type resultType = sig.getResultType();
    if (!KGEN::sugarIsa<KGEN::NoneType>(resultType)) {
      KernelDeclOperandAdaptor op;
      op.underlyingType =
          classifyArgType(ctx, resultType, /*promoteSIMD=*/false);
      outputResult = std::move(op);
    }
  }
}

KernelDeclFunctionAdaptor
KernelDeclFunctionAdaptor::synthesizeExecuteFromElementwise(
    KGEN::LIT::FnTypeGeneratorType sig) {
  mlir::MLIRContext *ctx = sig.getContext();
  KernelDeclFunctionAdaptor result;
  // Synthesised adaptor: wrap the synthetic signature in a minimal
  // KernelFunctionAttr so `attr.getSignature()` works at call sites.
  // The cached params / arg_params / result_params are intentionally null
  // (no real fn-level decls or slot mappings to project).
  result.attr = KernelFunctionAttr::get(ctx, sig, /*params=*/{},
                                        /*arg_params=*/{},
                                        /*result_params=*/{});

  size_t numArgs = sig.getNumArguments();
  bool resultAsArgument = false;
  for (size_t i = 0; i < numArgs; ++i) {
    if (sig.getArgConvention(i) == KGEN::ArgConvention::ByRefResult) {
      resultAsArgument = true;
      break;
    }
  }
  bool isThrow = sig.isThrows();
  size_t numByRefArgs = isThrow ? 2 : (resultAsArgument ? 1 : 0);
  size_t endOfInputArgs = numArgs - numByRefArgs;

  for (size_t i = 0; i < endOfInputArgs; ++i) {
    mlir::Type argType = sig.getArgument(i);

    // Skip a trailing IndexList argument (it is an elementwise-only concept).
    if (i + 1 == endOfInputArgs) {
      KGEN::LIT::StructType structType = getAsStructType(argType);
      if (structType && symbolMatchesRootAndLeaf(structType.getSymbol(),
                                                 kPackageStd, kLeafIndexList))
        continue;
    }

    KernelDeclOperandAdaptor op;
    op.sourceDescriptor = KernelDeclOperandAdaptor::SourceDescriptor{
        sig.getArgName(i), i, /*isByRefResult=*/false};
    op.underlyingType =
        classifyArgType(ctx, argType, /*promoteSIMDToFusedTensor=*/true);
    result.inputArguments.push_back(std::move(op));
  }

  // Synthesise the single output tensor.
  KernelDeclOperandAdaptor outOp;
  outOp.sourceDescriptor = KernelDeclOperandAdaptor::SourceDescriptor{
      mlir::StringAttr::get(ctx, "out"), 0, /*isByRefResult=*/false};
  outOp.underlyingType = KernelDeclTensorAdaptor{IOSpec::FusedOutputTensor};
  result.outputArguments.push_back(std::move(outOp));

  return result;
}

//===----------------------------------------------------------------------===//
// KernelDeclAdaptor
//===----------------------------------------------------------------------===//

KernelDeclAdaptor::KernelDeclAdaptor(KernelDeclOp decl) {
  mlir::MLIRContext *ctx = decl->getContext();
  registeredOpName = mlir::StringAttr::get(ctx, decl.getRegistration());
  fullyResolvedName = decl.getSymNameAttr();

  // Build a function adaptor from an optional `MOGG_KernelFunctionAttr`.
  // The adaptor stores the attr directly; consumers read its cached
  // fn-level params and slot-name mappings through `f.attr.*`.
  auto buildFn = [](std::optional<KernelFunctionAttr> fnAttr)
      -> std::optional<KernelDeclFunctionAdaptor> {
    if (!fnAttr)
      return std::nullopt;
    KernelDeclFunctionAdaptor f(fnAttr->getSignature());
    f.attr = *fnAttr;
    return f;
  };

  llvm::StringRef structName = fullyResolvedName.strref();
  executeFunction = buildFn(decl.getExecute());
  if (executeFunction) {
    executeFunction->fullyResolvedName = mlir::StringAttr::get(
        ctx, llvm::formatv("{}.execute", structName).str());
  }
  shapeFunction = buildFn(decl.getShape());
  if (shapeFunction) {
    shapeFunction->fullyResolvedName =
        mlir::StringAttr::get(ctx, llvm::formatv("{}.shape", structName).str());
  }
  updateViewFunction = buildFn(decl.getUpdateView());
  if (updateViewFunction) {
    updateViewFunction->fullyResolvedName = mlir::StringAttr::get(
        ctx, llvm::formatv("{}.update_input_view", structName).str());
  }
  buildGraphFunction = buildFn(decl.getBuildGraph());
  if (buildGraphFunction) {
    buildGraphFunction->fullyResolvedName = mlir::StringAttr::get(
        ctx, llvm::formatv("{}.build_graph", structName).str());
  }

  if (auto ewFn = decl.getElementwise()) {
    elementwiseFunction = buildFn(ewFn);
    elementwiseFunction->fullyResolvedName = mlir::StringAttr::get(
        ctx, llvm::formatv("{}.elementwise", structName).str());
    // When there is no explicit execute, synthesise one from the elementwise
    // signature so callers always have an executeFunction to query.
    if (!executeFunction) {
      executeFunction =
          KernelDeclFunctionAdaptor::synthesizeExecuteFromElementwise(
              ewFn->getSignature());
    }
  }

  if (auto tileEwFn = decl.getTileElementwise()) {
    tileElementwiseFunction = buildFn(tileEwFn);
    tileElementwiseFunction->fullyResolvedName = mlir::StringAttr::get(
        ctx, llvm::formatv("{}.elementwise", structName).str());
  }

  structParams = decl.getStructParams();
}

const KernelDeclFunctionAdaptor *
KernelDeclAdaptor::getFunctionByLabel(llvm::StringRef label) const {
  if (label == kMoggExecuteFunctionLabel)
    return executeFunction ? &*executeFunction : nullptr;
  if (label == kMoggShapeFunctionLabel)
    return shapeFunction ? &*shapeFunction : nullptr;
  if (label == kMoggUpdateViewFunctionLabel)
    return updateViewFunction ? &*updateViewFunction : nullptr;
  if (label == kMoggElementwiseFunctionLabel)
    return elementwiseFunction ? &*elementwiseFunction : nullptr;
  if (label == "tile_elementwise")
    return tileElementwiseFunction ? &*tileElementwiseFunction : nullptr;
  if (label == kMoggBuildGraphFunctionLabel)
    return buildGraphFunction ? &*buildGraphFunction : nullptr;
  return nullptr;
}

KernelFunctionDeclAdaptor::KernelFunctionDeclAdaptor(PrimitiveDeclOp decl)
    : registration(
          mlir::StringAttr::get(decl->getContext(), decl.getRegistration())),
      function(decl.getFunction().getSignature()) {
  function.fullyResolvedName = decl.getSymNameAttr();
}

//===----------------------------------------------------------------------===//
// IOSpec helpers.
//===----------------------------------------------------------------------===//

namespace M::MOGG {

StringRef toString(IOSpec spec) {
  switch (spec) {
  case IOSpec::InputTensor:
    return kInputTensor;
  case IOSpec::OutputTensor:
    return kOutputTensor;
  case IOSpec::MutableInputTensor:
    return kMutableInputTensor;
  case IOSpec::FusedInputTensor:
    return kFusedInputTensor;
  case IOSpec::FusedOutputTensor:
    return kFusedOutputTensor;
  case IOSpec::FusedComputeOutputTensor:
    return kFusedComputeOutputTensor;
  }
}

std::optional<IOSpec> toIOSpec(StringRef str) {
  if (str == kInputTensor)
    return IOSpec::InputTensor;
  if (str == kOutputTensor)
    return IOSpec::OutputTensor;
  if (str == kMutableInputTensor)
    return IOSpec::MutableInputTensor;
  if (str == kFusedInputTensor)
    return IOSpec::FusedInputTensor;
  if (str == kFusedOutputTensor)
    return IOSpec::FusedOutputTensor;
  if (str == kFusedComputeOutputTensor)
    return IOSpec::FusedComputeOutputTensor;
  return std::nullopt;
}

std::pair<TypedAttr, TypedAttr> getParams(IOSpec ioSpec, Builder &builder) {
  switch (ioSpec) {
  case IOSpec::InputTensor:
    return {builder.getBoolAttr(kIOSpecImmutable),
            builder.getIndexAttr(kIOSpecIOInput)};

  case IOSpec::OutputTensor:
    return {builder.getBoolAttr(kIOSpecMutable),
            builder.getIndexAttr(kIOSpecIOOutput)};

  case IOSpec::MutableInputTensor:
    return {builder.getBoolAttr(kIOSpecMutable),
            builder.getIndexAttr(kIOSpecIOInput)};

  case IOSpec::FusedInputTensor:
    return {builder.getBoolAttr(kIOSpecImmutable),
            builder.getIndexAttr(kIOSpecIOFusedInput)};

  case IOSpec::FusedOutputTensor:
    return {builder.getBoolAttr(kIOSpecMutable),
            builder.getIndexAttr(kIOSpecIOFusedOutput)};

  case IOSpec::FusedComputeOutputTensor:
    return {builder.getBoolAttr(kIOSpecMutable),
            builder.getIndexAttr(kIOSpecIOFusedComputeOutput)};
  }
}

bool isOutputIOSpec(IOSpec spec) {
  return spec == IOSpec::OutputTensor || spec == IOSpec::FusedOutputTensor ||
         spec == IOSpec::FusedComputeOutputTensor;
}

bool isFusableIOSpec(IOSpec spec) {
  return spec == IOSpec::FusedInputTensor ||
         spec == IOSpec::FusedOutputTensor ||
         spec == IOSpec::FusedComputeOutputTensor;
}

bool isMutableIOSpec(IOSpec spec) { return spec == IOSpec::MutableInputTensor; }

} // namespace M::MOGG
