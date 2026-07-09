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
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include <iterator>

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
      else if (name == kIoFusedComputeOutputTile)
        input = kIOSpecIOFusedComputeOutputTile;
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
  if (mut == kIOSpecMutable && input == kIOSpecIOFusedComputeOutputTile)
    return IOSpec::FusedComputeOutputTileTensor;
  return std::nullopt;
}

/// If `argType` is a trait-bound tensor argument, re-derive its
/// `TensorArgContract` from the written trait composition (the same path
/// `distillTensorArgContract` exercises at import; errors are swallowed here
/// because import already validated and reported any malformed composition).
/// Returns nullopt for a non-trait arg or an unresolvable composition.
static std::optional<TensorArgContract>
recoverTraitArgContract(mlir::Type argType) {
  KGEN::LIT::TraitType traitType = getTensorArgTraitType(argType);
  if (!traitType)
    return std::nullopt;
  mlir::FailureOr<TensorArgContract> contract = distillTensorArgContract(
      traitType.getSymbols(), [](const llvm::Twine &) {});
  if (mlir::failed(contract))
    return std::nullopt;
  return *contract;
}

/// Build a `KernelDeclTensorAdaptor` carrying the role/mut implied by a trait
/// arg's contract, so the downstream predicates
/// (isTensorType/isMutableTensorType/isDpsOutput) behave the same as for the
/// equivalent legacy `IOSpec`. The mapping mirrors `getParams` /
/// `isMutableIOSpec`: `output -> OutputTensor`, `mutable_input ->
/// MutableInputTensor`, `input -> InputTensor`. Fusion is not modeled in the
/// adaptor for trait args in this slice (dense-only); `fused` stays false.
static KernelDeclTensorAdaptor
traitContractToTensorAdaptor(const TensorArgContract &contract) {
  switch (contract.role) {
  case TensorArgRole::output:
    return KernelDeclTensorAdaptor{IOSpec::OutputTensor};
  case TensorArgRole::mutableInput:
    return KernelDeclTensorAdaptor{IOSpec::MutableInputTensor};
  case TensorArgRole::input:
    return KernelDeclTensorAdaptor{IOSpec::InputTensor};
  }
  llvm_unreachable("unhandled TensorArgRole");
}

/// Classify a single argument type into a KernelDeclOperandVariant by
/// introspecting the LIT StructType symbol and parameter values.
static KernelDeclOperandVariant classifyArgType(mlir::MLIRContext *ctx,
                                                mlir::Type argType,
                                                bool promoteSIMDToFusedTensor) {
  KGEN::LIT::StructType structType = getAsStructType(argType);
  if (!structType) {
    // Trait lane: a trait-bound tensor argument is non-struct
    // (getAsStructType null). Classify it as a tensor adaptor carrying the
    // role/mut from its distilled contract, so it participates in operand
    // mapping and the DPS-output split like the equivalent legacy IOSpec arg.
    // Non-trait non-struct args remain unsupported, exactly as before.
    if (std::optional<TensorArgContract> contract =
            recoverTraitArgContract(argType))
      return traitContractToTensorAdaptor(*contract);
    return KernelDeclUnsupportedAdaptor{};
  }

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
    mlir::Type argType = sig.getArgument(i);
    KGEN::LIT::StructType structType = getAsStructType(argType);
    if (!structType) {
      // Trait lane: a trait-bound tensor argument is non-struct. Recover its
      // role from the distilled contract; an `output` role continues the
      // leading-output count, anything else (input/mutable_input) or a
      // non-trait non-struct arg stops it -- preserving the legacy
      // "break at the first non-output" semantics.
      std::optional<TensorArgContract> contract =
          recoverTraitArgContract(argType);
      if (!contract || contract->role != TensorArgRole::output)
        break;
      ++count;
      continue;
    }
    auto ioSpec = getIOSpecFromStructType(structType);
    if (!ioSpec || !isOutputIOSpec(*ioSpec))
      break;
    ++count;
  }
  return count;
}

//===----------------------------------------------------------------------===//
// Trait-lane vocabulary classification
//===----------------------------------------------------------------------===//

// How a vocabulary trait participates in the distilled contract. The
// distiller's validation logic is written per category, never per trait, so
// adding a trait to an existing category is a single `kTensorArgVocab` row
// with no logic changes; only a genuinely new *kind* of trait grows this enum
// (and with it the distiller's per-category handling).
enum class TraitCategory {
  /// An io-role marker (`Input` / `Output` / `MutableInput`); exactly one is
  /// required. The row's `role` payload names the role it contributes.
  role,
  /// The `Fused` fusion marker.
  fused,
  /// A metadata-only base trait (`Tensor` / `DenseTensor`); carries no
  /// contract meaning.
  metadata,
  /// A dense data-access capability (`TileTensorable`); carries no contract
  /// meaning on its own.
  dataCapability,
  /// An access-form marker (`LoadAccess` / `StoreAccess` / `TransformAccess`).
  /// The row's `access` payload names the form it contributes.
  accessMarker,
  /// An access-form callable capability (`Loadable` / `Storable` /
  /// `Transformable`). The row's `access` payload names the form.
  accessCapability,
  /// Reserved for the deferred tile-transform fusion traits (Phase 8); always
  /// rejected until then.
  deferred,
};

// One row of the tensor-argument trait vocabulary: the trait's leaf name, its
// category, and the category's payload (the role a `role` row contributes, or
// the access form an `accessMarker` / `accessCapability` row contributes).
struct VocabEntry {
  StringLiteral leaf;
  TraitCategory category;
  /// The io-role, for `role` rows.
  TensorArgRole role = TensorArgRole::input;
  /// The access form, for `accessMarker` / `accessCapability` rows.
  TensorArgAccess access = TensorArgAccess::none;

private:
  // Deliberately not constexpr. `kTensorArgVocab` is constexpr, so its rows
  // are built in constant evaluation, where reaching a call to a
  // non-constexpr function is a compile error that names both the function
  // (the violated invariant) and the offending row. Unreachable at runtime:
  // only the constructors call these, and every VocabEntry is a
  // `kTensorArgVocab` row.
  static void payloadCategoryNeedsAPayload() {
    llvm_unreachable("vocab row invariants are compile-time only");
  }
  static void onlyRoleRowsCarryARole() {
    llvm_unreachable("vocab row invariants are compile-time only");
  }
  static void onlyAccessRowsCarryAForm() {
    llvm_unreachable("vocab row invariants are compile-time only");
  }

public:
  constexpr VocabEntry(StringLiteral leaf, TraitCategory category)
      : leaf(leaf), category(category) {
    if (category == TraitCategory::role ||
        category == TraitCategory::accessMarker ||
        category == TraitCategory::accessCapability)
      payloadCategoryNeedsAPayload();
  }
  constexpr VocabEntry(StringLiteral leaf, TraitCategory category,
                       TensorArgRole role)
      : leaf(leaf), category(category), role(role) {
    if (category != TraitCategory::role)
      onlyRoleRowsCarryARole();
  }
  constexpr VocabEntry(StringLiteral leaf, TraitCategory category,
                       TensorArgAccess access)
      : leaf(leaf), category(category), access(access) {
    if (category != TraitCategory::accessMarker &&
        category != TraitCategory::accessCapability)
      onlyAccessRowsCarryAForm();
    if (access == TensorArgAccess::none)
      onlyAccessRowsCarryAForm();
  }
};

// The tensor-argument trait vocabulary defined in
// `extensibility::tensor_arg_traits`. Keep in sync with tensor_arg_traits.mojo
// -- except `TileTransformAccess` / `TileTransformable`, reserved ahead of
// time for the deferred tile-transform fusion traits (Phase 8), which do not
// exist in the shipped module yet; `classifyVocabTrait` can never match them
// against a real trait symbol until then, so the `deferred` rejection in the
// distiller is unreachable in the interim.
constexpr VocabEntry kTensorArgVocab[] = {
    {"Input", TraitCategory::role, TensorArgRole::input},
    {"Output", TraitCategory::role, TensorArgRole::output},
    {"MutableInput", TraitCategory::role, TensorArgRole::mutableInput},
    {"Fused", TraitCategory::fused},
    {"Tensor", TraitCategory::metadata},
    {"DenseTensor", TraitCategory::metadata},
    {"LoadAccess", TraitCategory::accessMarker, TensorArgAccess::load},
    {"Loadable", TraitCategory::accessCapability, TensorArgAccess::load},
    {"StoreAccess", TraitCategory::accessMarker, TensorArgAccess::store},
    {"Storable", TraitCategory::accessCapability, TensorArgAccess::store},
    {"TransformAccess", TraitCategory::accessMarker,
     TensorArgAccess::transform},
    {"Transformable", TraitCategory::accessCapability,
     TensorArgAccess::transform},
    {"TileTensorable", TraitCategory::dataCapability},
    {"TileTransformAccess", TraitCategory::deferred},
    {"TileTransformable", TraitCategory::deferred},
};

// Whole-table invariants, checked at compile time. The per-row invariants
// (payload/category coherence) live in the VocabEntry constructors; these
// cover what no single row can see.
// `StringRef::operator==` is not constexpr here; compare manually.
constexpr bool sameLeaf(StringLiteral a, StringLiteral b) {
  if (a.size() != b.size())
    return false;
  for (size_t i = 0; i < a.size(); ++i)
    if (a.data()[i] != b.data()[i])
      return false;
  return true;
}

constexpr bool leavesAreUnique() {
  for (size_t i = 0; i < std::size(kTensorArgVocab); ++i)
    for (size_t j = i + 1; j < std::size(kTensorArgVocab); ++j)
      if (sameLeaf(kTensorArgVocab[i].leaf, kTensorArgVocab[j].leaf))
        return false;
  return true;
}

constexpr bool payloadsAreUnique() {
  for (size_t i = 0; i < std::size(kTensorArgVocab); ++i)
    for (size_t j = i + 1; j < std::size(kTensorArgVocab); ++j) {
      const VocabEntry &a = kTensorArgVocab[i];
      const VocabEntry &b = kTensorArgVocab[j];
      if (a.category != b.category)
        continue;
      if (a.category == TraitCategory::role && a.role == b.role)
        return false;
      if ((a.category == TraitCategory::accessMarker ||
           a.category == TraitCategory::accessCapability) &&
          a.access == b.access)
        return false;
    }
  return true;
}

constexpr bool accessFormsArePaired() {
  // The distiller keys marker and capability rows of one form to each other;
  // a form must carry both rows.
  for (const VocabEntry &a : kTensorArgVocab) {
    if (a.category != TraitCategory::accessMarker &&
        a.category != TraitCategory::accessCapability)
      continue;
    TraitCategory wanted = a.category == TraitCategory::accessMarker
                               ? TraitCategory::accessCapability
                               : TraitCategory::accessMarker;
    bool paired = false;
    for (const VocabEntry &b : kTensorArgVocab)
      paired |= b.category == wanted && b.access == a.access;
    if (!paired)
      return false;
  }
  return true;
}

static_assert(leavesAreUnique(), "duplicate leaf in kTensorArgVocab");
static_assert(payloadsAreUnique(),
              "two kTensorArgVocab rows claim the same (category, payload)");
static_assert(accessFormsArePaired(),
              "unpaired access marker/capability row in kTensorArgVocab");

// Classify a trait symbol as a tensor-argument vocabulary trait, returning its
// vocabulary row, or nullptr if it is foreign (not from
// `extensibility::tensor_arg_traits`) or its leaf is unrecognized. The module
// is pinned -- root contains "extensibility" and some qualifier contains
// "tensor_arg_traits" -- so a same-named trait from another module is
// rejected. The leaf is matched exactly so that, e.g., `Input` does not
// spuriously match `MutableInput`.
static const VocabEntry *classifyVocabTrait(mlir::SymbolRefAttr sym) {
  if (!sym.getRootReference().strref().contains("extensibility"))
    return nullptr;
  bool inModule = false;
  for (mlir::FlatSymbolRefAttr ref : sym.getNestedReferences()) {
    if (ref.getValue().contains("tensor_arg_traits")) {
      inModule = true;
      break;
    }
  }
  if (!inModule)
    return nullptr;
  for (const VocabEntry &entry : kTensorArgVocab)
    if (sym.getLeafReference() == entry.leaf)
      return &entry;
  return nullptr;
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

KernelDeclFunctionAdaptor::KernelDeclFunctionAdaptor(
    KernelFunctionAttr attr, StringAttr fullyResolvedName)
    : KernelDeclFunctionAdaptor(attr.getSignature()) {
  this->attr = attr;
  this->fullyResolvedName = fullyResolvedName;
}

KernelDeclFunctionAdaptor::KernelDeclFunctionAdaptor(
    ShapeFunctionDeclOp shapeDecl)
    : KernelDeclFunctionAdaptor(shapeDecl.getFunction(),
                                shapeDecl.getSymNameAttr()) {}

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
                                        /*result_params=*/{},
                                        /*tensor_arg_contracts=*/{});

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
  auto buildFn =
      [&](std::optional<KernelFunctionAttr> fnAttr,
          StringRef methodName) -> std::optional<KernelDeclFunctionAdaptor> {
    if (!fnAttr)
      return std::nullopt;
    KernelDeclFunctionAdaptor f(fnAttr->getSignature());
    f.attr = *fnAttr;
    f.fullyResolvedName = mlir::StringAttr::get(
        ctx, llvm::formatv("{}.{}", fullyResolvedName, methodName).str());
    return f;
  };

  executeFunction = buildFn(decl.getExecute(), kMoggExecuteFuncName);
  updateViewFunction = buildFn(decl.getUpdateView(), kMoggUpdateViewFuncName);
  buildGraphFunction = buildFn(decl.getBuildGraph(), kMoggBuildGraphFuncName);
  elementwiseFunction =
      buildFn(decl.getElementwise(), kMoggElementwiseFuncName);
  tileElementwiseFunction =
      buildFn(decl.getTileElementwise(), kMoggElementwiseFuncName);

  if (elementwiseFunction && !executeFunction) {
    // When there is no explicit execute, synthesise one from the elementwise
    // signature so callers always have an executeFunction to query.
    executeFunction =
        KernelDeclFunctionAdaptor::synthesizeExecuteFromElementwise(
            decl.getElementwise()->getSignature());
  }

  structParams = decl.getStructParams();
}

const KernelDeclFunctionAdaptor *
KernelDeclAdaptor::getFunctionByLabel(llvm::StringRef label) const {
  if (label == kMoggExecuteFunctionLabel)
    return executeFunction ? &*executeFunction : nullptr;
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
  case IOSpec::FusedComputeOutputTileTensor:
    return kFusedComputeOutputTileTensor;
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
  if (str == kFusedComputeOutputTileTensor)
    return IOSpec::FusedComputeOutputTileTensor;
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

  case IOSpec::FusedComputeOutputTileTensor:
    return {builder.getBoolAttr(kIOSpecMutable),
            builder.getIndexAttr(kIOSpecIOFusedComputeOutputTile)};
  }
}

bool isOutputIOSpec(IOSpec spec) {
  return spec == IOSpec::OutputTensor || spec == IOSpec::FusedOutputTensor ||
         spec == IOSpec::FusedComputeOutputTensor ||
         spec == IOSpec::FusedComputeOutputTileTensor;
}

bool isFusableIOSpec(IOSpec spec) {
  return spec == IOSpec::FusedInputTensor ||
         spec == IOSpec::FusedOutputTensor ||
         spec == IOSpec::FusedComputeOutputTensor ||
         spec == IOSpec::FusedComputeOutputTileTensor;
}

bool isMutableIOSpec(IOSpec spec) { return spec == IOSpec::MutableInputTensor; }

mlir::FailureOr<TensorArgContract> distillTensorArgContract(
    llvm::ArrayRef<mlir::SymbolRefAttr> writtenSymbols,
    llvm::function_ref<void(const llvm::Twine &)> emitError) {
  // Per-row presence over the vocabulary table. Presence is idempotent, so a
  // trait surfacing more than once (e.g. both as written and as another
  // trait's ancestor) counts once.
  bool present[std::size(kTensorArgVocab)] = {};
  for (mlir::SymbolRefAttr sym : writtenSymbols) {
    if (const VocabEntry *entry = classifyVocabTrait(sym)) {
      present[entry - kTensorArgVocab] = true;
      continue;
    }
    // Not a vocabulary trait. Trait-type canonicalization folds a composition
    // into the ancestor closure of the traits it composes, so `writtenSymbols`
    // carries the declared traits plus their ancestors. The only non-vocabulary
    // ancestor a valid composition can surface is the universal `AnyType` root
    // (every trait conforms to it) -- every other ancestor of a vocabulary
    // trait is itself a vocabulary trait (`DenseTensor: Tensor`,
    // `Loadable: LoadAccess`, ...). So accept `AnyType` and reject anything
    // else as a genuinely foreign trait (which the author had to write
    // explicitly; it is not implied by the vocabulary).
    if (sym.getLeafReference() == "AnyType")
      continue;
    emitError("trait-bound tensor argument references '" +
              sym.getLeafReference().getValue() +
              "', which is not a tensor-argument trait; only traits from "
              "extensibility.tensor_arg_traits are allowed");
    return mlir::failure();
  }

  // Visit the present vocabulary rows of one category, in table order.
  auto forEachPresent = [&](TraitCategory category,
                            llvm::function_ref<void(const VocabEntry &)> fn) {
    for (size_t i = 0; i < std::size(kTensorArgVocab); ++i)
      if (present[i] && kTensorArgVocab[i].category == category)
        fn(kTensorArgVocab[i]);
  };
  auto anyPresent = [&](TraitCategory category) {
    bool any = false;
    forEachPresent(category, [&](const VocabEntry &) { any = true; });
    return any;
  };

  // Exactly one role trait.
  int roleCount = 0;
  TensorArgRole role = TensorArgRole::input;
  forEachPresent(TraitCategory::role, [&](const VocabEntry &entry) {
    ++roleCount;
    role = entry.role;
  });
  if (roleCount == 0) {
    emitError("trait-bound tensor argument has no role trait (expected exactly "
              "one of Input, Output, MutableInput)");
    return mlir::failure();
  }
  if (roleCount > 1) {
    emitError("trait-bound tensor argument has more than one role trait "
              "(expected exactly one of Input, Output, MutableInput)");
    return mlir::failure();
  }

  TensorArgContract contract;
  contract.role = role;

  contract.fused = anyPresent(TraitCategory::fused);

  // Tile-transform fusion is deferred (Phase 8).
  if (anyPresent(TraitCategory::deferred)) {
    emitError("tile-transform fusion is not supported yet");
    return mlir::failure();
  }

  // Access form: at most one, keyed off either the marker or the capability
  // leaf. The written composition may carry only the capability (e.g.
  // `Loadable` without `LoadAccess`), or only the marker (opaque-device host
  // descriptor); both map to the same access form.
  int accessCount = 0;
  for (TensorArgAccess form : {TensorArgAccess::load, TensorArgAccess::store,
                               TensorArgAccess::transform}) {
    bool marker = false;
    forEachPresent(TraitCategory::accessMarker, [&](const VocabEntry &entry) {
      marker |= entry.access == form;
    });
    bool capability = false;
    forEachPresent(
        TraitCategory::accessCapability,
        [&](const VocabEntry &entry) { capability |= entry.access == form; });
    if (marker || capability) {
      ++accessCount;
      contract.access = form;
      contract.hasCapability = capability;
    }
  }
  if (accessCount > 1) {
    emitError("trait-bound tensor argument declares more than one access form "
              "(at most one of load, store, transform)");
    return mlir::failure();
  }

  // A fused argument must declare exactly one access form.
  if (contract.fused && contract.access == TensorArgAccess::none) {
    emitError("a Fused tensor argument must declare an access form (one of "
              "LoadAccess/Loadable, StoreAccess/Storable, "
              "TransformAccess/Transformable)");
    return mlir::failure();
  }

  // Role / access compatibility.
  switch (contract.access) {
  case TensorArgAccess::load:
    if (contract.role == TensorArgRole::output) {
      emitError("load access is not valid on an Output argument");
      return mlir::failure();
    }
    break;
  case TensorArgAccess::store:
    if (contract.role == TensorArgRole::input) {
      emitError("store access is not valid on a read-only Input argument");
      return mlir::failure();
    }
    break;
  case TensorArgAccess::transform:
    if (contract.role != TensorArgRole::output) {
      emitError("transform access requires an Output argument");
      return mlir::failure();
    }
    break;
  case TensorArgAccess::none:
    break;
  }

  return contract;
}

KGEN::LIT::TraitType getTensorArgTraitType(mlir::Type argType) {
  mlir::Type t = argType;
  // Borrowed tensor args are wrapped in a ref; strip it (mirrors
  // getAsStructType).
  if (auto ref = KGEN::sugarDynCast<KGEN::LIT::RefType>(t))
    t = ref.getElementType();
  // A trait-bound arg `x: T` is a reference to the generic type parameter `T`,
  // whose metatype is the written `!lit.trait<...>` bound. Mirrors the
  // type-parameter branch of `simplifyConformsToAgainstTypeValue`.
  auto paramType = KGEN::sugarDynCast<KGEN::ParamType>(t);
  if (!paramType)
    return {};
  auto traitType = mlir::dyn_cast_if_present<KGEN::LIT::TraitType>(
      KGEN::getCanonicalType(paramType.getMetaType()));
  if (!traitType)
    return {};
  // Only a *tensor-argument* bound, not just any trait-bound generic parameter:
  // a generic arg bound purely by stdlib/other traits (e.g. a primitive's
  // `[T: ImplicitlyCopyable]`) is not a tensor arg. Require at least one
  // tensor-argument vocabulary trait before distilling.
  for (mlir::SymbolRefAttr sym : traitType.getSymbols())
    if (classifyVocabTrait(sym))
      return traitType;
  return {};
}

StringRef toString(TensorArgRole role) {
  switch (role) {
  case TensorArgRole::input:
    return "input";
  case TensorArgRole::output:
    return "output";
  case TensorArgRole::mutableInput:
    return "mutable_input";
  }
  llvm_unreachable("bad TensorArgRole");
}

StringRef toString(TensorArgAccess access) {
  switch (access) {
  case TensorArgAccess::none:
    return "none";
  case TensorArgAccess::load:
    return "load";
  case TensorArgAccess::store:
    return "store";
  case TensorArgAccess::transform:
    return "transform";
  }
  llvm_unreachable("bad TensorArgAccess");
}

mlir::DictionaryAttr
tensorArgContractToDict(mlir::MLIRContext *ctx,
                        const TensorArgContract &contract) {
  mlir::Builder b(ctx);
  return mlir::DictionaryAttr::get(
      ctx,
      {b.getNamedAttr("role", b.getStringAttr(toString(contract.role))),
       b.getNamedAttr("fused", b.getBoolAttr(contract.fused)),
       b.getNamedAttr("access", b.getStringAttr(toString(contract.access))),
       b.getNamedAttr("has_capability",
                      b.getBoolAttr(contract.hasCapability))});
}

std::optional<TensorArgContract> dictToTensorArgContract(mlir::Attribute attr) {
  // The "no contract" sentinel: a non-trait argument is baked as a UnitAttr
  // placeholder (and a null attr is treated the same way).
  auto dict = mlir::dyn_cast_if_present<mlir::DictionaryAttr>(attr);
  if (!dict)
    return std::nullopt;

  auto getStr = [&](StringRef key) -> std::optional<StringRef> {
    if (auto s = dict.getAs<mlir::StringAttr>(key))
      return s.strref();
    return std::nullopt;
  };
  auto getBool = [&](StringRef key) -> std::optional<bool> {
    if (auto b = dict.getAs<mlir::BoolAttr>(key))
      return b.getValue();
    return std::nullopt;
  };

  std::optional<StringRef> roleStr = getStr("role");
  std::optional<bool> fused = getBool("fused");
  std::optional<StringRef> accessStr = getStr("access");
  std::optional<bool> hasCapability = getBool("has_capability");
  if (!roleStr || !fused || !accessStr || !hasCapability)
    return std::nullopt;

  // Inverse of `toString(TensorArgRole)`.
  TensorArgRole role;
  if (*roleStr == toString(TensorArgRole::input))
    role = TensorArgRole::input;
  else if (*roleStr == toString(TensorArgRole::output))
    role = TensorArgRole::output;
  else if (*roleStr == toString(TensorArgRole::mutableInput))
    role = TensorArgRole::mutableInput;
  else
    return std::nullopt;

  // Inverse of `toString(TensorArgAccess)`.
  TensorArgAccess access;
  if (*accessStr == toString(TensorArgAccess::none))
    access = TensorArgAccess::none;
  else if (*accessStr == toString(TensorArgAccess::load))
    access = TensorArgAccess::load;
  else if (*accessStr == toString(TensorArgAccess::store))
    access = TensorArgAccess::store;
  else if (*accessStr == toString(TensorArgAccess::transform))
    access = TensorArgAccess::transform;
  else
    return std::nullopt;

  return TensorArgContract{role, *fused, access, *hasCapability};
}

} // namespace M::MOGG
