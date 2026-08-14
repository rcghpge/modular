//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// MOGG kernel signature introspection.
//
// `KernelDeclAdaptor` wraps a `mogg.kernel.decl` op and decodes its stored
// `FnTypeGeneratorType` signatures directly. The `IOSpec` enum and the
// `is*` predicates / `kParameter*` constants used to drive decoding live
// alongside it.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MOGGDIALECT_ANALYSIS_KERNELDECLADAPTOR_H
#define GENERICML_GRAPHCOMPILER_MOGGDIALECT_ANALYSIS_KERNELDECLADAPTOR_H

#include "GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GraphCompiler/MOGGDialect/Support/MojoIdentifiers.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/LITDialect/LITTypes.h"
#include "KGEN/LITDialect/LITUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/bit.h"
#include <cassert>
#include <optional>
#include <type_traits>
#include <variant>

namespace M::MOGG {

enum class IOSpec {
  InputTensor,        // Input tensor, read-only
  OutputTensor,       // Output tensor, write-only
  MutableInputTensor, // Input tensor that can be modified
  FusedInputTensor,
  FusedOutputTensor,
  FusedComputeOutputTensor,
  FusedComputeOutputTileTensor,
  FusedOutputTileTensor,
};

//===----------------------------------------------------------------------===//
// Tensor-argument trait contract (trait lane)
//
// The trait lane classifies a trait-bound tensor argument directly from the
// *written* trait composition on its `!lit.trait<...>` type, independent of the
// legacy `IOSpec` / `ManagedTensorSlice` struct-parameter decode. The two lanes
// coexist; see `distillTensorArgContract`.
//===----------------------------------------------------------------------===//

/// Role a tensor argument plays in the graph signature. `mut` is not stored;
/// see `TensorArgContract::isMutable`.
///
/// `none` is the no-role flavor: a shape-function tensor argument carries no
/// io-role at all (it reads only metadata, e.g. `.shape()`, and is always
/// read-only). It is distinct from `input` -- an `input`-roled arg is an
/// `execute` read operand, whereas `none` records the structural *absence* of a
/// role. The distiller (`distillTensorArgContract` in shape-function mode)
/// requires `none` and rejects every io-role marker; an `execute` argument
/// conversely requires exactly one of `input`/`output`/`mutableInput`.
enum class TensorArgRole { input, output, mutableInput, none };

/// Element-access form a tensor argument exposes -- the fusion access kind, or
/// `none` for a plain dense argument. Used both as a single canonical form
/// (e.g. for a fused arg, which carries exactly one) and as an index into the
/// access/capability bitmasks of `TensorArgContract`.
enum class TensorArgAccess { none, load, store, transform };

/// Single-bit mask for one access form (`none` maps to the empty mask 0).
inline unsigned bit(TensorArgAccess access) {
  switch (access) {
  case TensorArgAccess::none:
    return 0;
  case TensorArgAccess::load:
    return 1u << 0;
  case TensorArgAccess::store:
    return 1u << 1;
  case TensorArgAccess::transform:
    return 1u << 2;
  }
  return 0;
}

/// The distilled tensor-argument contract for a trait-bound kernel argument.
///
/// The access model is a SET of forms, not a single one: a non-fused dense
/// argument may carry several (e.g. both `load` and `store`, for an in-place
/// read-modify-write). `accessMask` records which forms are present; the
/// parallel `capabilityMask` records, per present form, whether the callable
/// capability leaf (e.g. `Loadable`) is present rather than only the access
/// marker (e.g. `LoadAccess`). A fused argument is constrained to exactly one
/// present form (see `distillTensorArgContract`).
struct TensorArgContract {
  TensorArgRole role = TensorArgRole::input;
  bool fused = false;
  /// Bitmask of present access forms (`bit(load) | bit(store) | ...`).
  unsigned accessMask = 0;
  /// Bitmask of present capability leaves, a subset of `accessMask`. A bit set
  /// here means the capability (e.g. `Loadable`) was required, not just the
  /// marker (e.g. `LoadAccess`). Load-bearing for opaque-device targets, where
  /// only the marker may be available host-side.
  unsigned capabilityMask = 0;

  /// `mut` is derived, never stored. Only `output` / `mutableInput` are
  /// mutable; both `input` and the no-role shape-function flavor (`none`) are
  /// read-only.
  bool isMutable() const {
    return role == TensorArgRole::output || role == TensorArgRole::mutableInput;
  }

  /// Whether the given access form is present in the set.
  bool hasAccess(TensorArgAccess access) const {
    return (accessMask & bit(access)) != 0;
  }
  /// Whether the given access form's capability leaf is present.
  bool hasCapability(TensorArgAccess access) const {
    return (capabilityMask & bit(access)) != 0;
  }
  /// Whether any access form is present.
  bool hasAnyAccess() const { return accessMask != 0; }
  /// Number of present access forms.
  int accessCount() const { return llvm::popcount(accessMask); }
  /// The sole present access form. Only valid when `accessCount() == 1` (a
  /// fused arg guarantees this); asserts otherwise.
  TensorArgAccess soleAccess() const {
    assert(accessCount() == 1 && "soleAccess requires exactly one access form");
    for (TensorArgAccess a : {TensorArgAccess::load, TensorArgAccess::store,
                              TensorArgAccess::transform})
      if (hasAccess(a))
        return a;
    return TensorArgAccess::none;
  }

  bool operator==(const TensorArgContract &o) const {
    return role == o.role && fused == o.fused && accessMask == o.accessMask &&
           capabilityMask == o.capabilityMask;
  }
  bool operator!=(const TensorArgContract &o) const { return !(*this == o); }
};

/// Distill the written trait composition of a trait-bound tensor argument (the
/// `getSymbols()` of its `!lit.trait<...>` type) into a `TensorArgContract`.
///
/// `writtenSymbols` are matched against the tensor-argument trait vocabulary
/// defined in `extensibility.tensor_arg_traits`. Any symbol from another module
/// (a foreign trait such as `Copyable`/`DevicePassable`), or any unrecognized
/// leaf, is rejected via `emitError`. The input may be either the written
/// composition or its ancestor-inclusive canonical form -- both distill
/// identically, since the parent traits (`Tensor`, the `*Access` markers) are
/// themselves in the allowlist and carry no role/access meaning. The universal
/// `AnyType` ancestor (present in every trait's canonical closure) is also
/// ignored. Returns failure (after invoking `emitError`) on any validation
/// failure.
///
/// `isShapeFunc` selects the shape-function lane: a shape-fn tensor argument is
/// metadata-only (reads `.shape()`), host-executed, and read-only, so it
/// carries **no role** (distills to `TensorArgRole::none`) and the allowed
/// vocabulary is `Tensor` / `DenseTensor` only. Any io-role marker
/// (`Input`/`Output`/`MutableInput`), `Fused`, `TileTensorable`, or any access
/// capability is a hard error -- data-reading shape functions stay on the
/// legacy `InputTensor` lane until the data-dependent projection is wired (see
/// the shape-fn execution plan, Slice 5). When `isShapeFunc` is false (the
/// `execute` lane) exactly one io-role is required, as before.
mlir::FailureOr<TensorArgContract> distillTensorArgContract(
    llvm::ArrayRef<M::KGEN::TraitSymbolAttr> writtenSymbols,
    llvm::function_ref<void(const llvm::Twine &)> emitError,
    bool isShapeFunc = false);

/// If `argType` is a trait-bound tensor argument -- a `!lit.ref` over a generic
/// type parameter whose bound is a `!lit.trait<...>` -- return that `TraitType`
/// (carrying the written trait composition). Otherwise (a `ManagedTensorSlice`
/// struct, a context type, a scalar, a non-trait-bound param, ...) return null.
KGEN::LIT::TraitType getTensorArgTraitType(mlir::Type argType);

llvm::StringRef toString(TensorArgRole role);
llvm::StringRef toString(TensorArgAccess access);

/// Encode a contract as the per-argument `DictionaryAttr`
/// `{role, fused, access, capabilities}` baked into
/// `KernelFunctionAttr.tensor_arg_contracts`. `access` and `capabilities` are
/// the integer bitmasks (sets of access forms) of the contract.
mlir::DictionaryAttr tensorArgContractToDict(mlir::MLIRContext *ctx,
                                             const TensorArgContract &contract);

/// Build the `#mogg.tensor_arg<{role, fused, access, capabilities}>`
/// attribute (`MOGG::TensorArgContractAttr`) carried by
/// `mogg.tensor_arg.__init__` from a distilled contract. `access` and
/// `capabilities` are bit-enum sets. The op-borne attribute is the single
/// source of truth in the trait lane (D1); this is the bridge from the C++
/// contract struct to the IR attribute.
MOGG::TensorArgContractAttr
tensorArgContractToAttr(mlir::MLIRContext *ctx,
                        const TensorArgContract &contract);

/// Decode a per-argument entry of `KernelFunctionAttr.tensor_arg_contracts`
/// back into a `TensorArgContract` -- the inverse of `tensorArgContractToDict`.
///
/// The baked array uses a `UnitAttr` placeholder for a non-trait argument; for
/// that placeholder (or any null/unexpected attr) this returns `std::nullopt`
/// (the "no contract" sentinel). A `DictionaryAttr` with the expected
/// `{role, fused, access, capabilities}` keys decodes to the matching
/// contract; the `role` string mapping mirrors `toString` exactly, and the
/// `access`/`capabilities` integer bitmasks round-trip directly.
std::optional<TensorArgContract> dictToTensorArgContract(mlir::Attribute attr);

//===----------------------------------------------------------------------===//
// Type predicates and IOSpec helpers
//===----------------------------------------------------------------------===//

inline bool isXType(KGEN::LIT::StructType maybeTensor, llvm::StringLiteral root,
                    llvm::StringLiteral className) {
  if (maybeTensor.getSymbol().getRootReference() != root)
    return false;
  return maybeTensor.getSymbol().getLeafReference() == className;
}

inline bool isExtensibilityTensor(KGEN::LIT::StructType maybeTensor) {
  return isXType(maybeTensor, "extensibility", "Tensor");
}

inline bool isCustomType(KGEN::LIT::StructType maybeCustom) {
  return !isExtensibilityTensor(maybeCustom);
}

llvm::StringRef toString(IOSpec spec);
std::optional<IOSpec> toIOSpec(llvm::StringRef str);
std::pair<mlir::TypedAttr, mlir::TypedAttr> getParams(IOSpec ioSpec,
                                                      mlir::Builder &builder);

bool isOutputIOSpec(IOSpec spec);
bool isFusableIOSpec(IOSpec spec);
bool isMutableIOSpec(IOSpec spec);

//===----------------------------------------------------------------------===//
// KernelDeclAdaptor: reads mogg.kernel.decl op signatures directly.
//===----------------------------------------------------------------------===//

struct KernelDeclTensorAdaptor {
  bool mut = false;
  bool fused = false;
  IOSpec ioSpec = {};
  /// Present only for a trait-lane argument: its distilled contract. The legacy
  /// `ManagedTensorSlice` lane leaves this empty and is unaffected. Carried so
  /// downstream lowering can route a trait arg to `mogg.tensor_arg.__init__`
  /// instead of `mogg.tensor.__init__` (D1/D5); the `ioSpec`/`mut`/`fused`
  /// fields above mirror the equivalent legacy values so the existing
  /// predicates behave identically for both lanes.
  std::optional<TensorArgContract> traitContract;
  KernelDeclTensorAdaptor() = default;
  explicit KernelDeclTensorAdaptor(IOSpec ioSpec)
      : mut(isMutableIOSpec(ioSpec)), fused(isFusableIOSpec(ioSpec)),
        ioSpec(ioSpec) {}
  bool operator==(const KernelDeclTensorAdaptor &o) const {
    return mut == o.mut;
  }
};

struct KernelDeclVariadicTensorAdaptor {
  bool mut = false;
  bool fused = false;
  IOSpec ioSpec = {};
  KernelDeclVariadicTensorAdaptor() = default;
  explicit KernelDeclVariadicTensorAdaptor(IOSpec ioSpec)
      : mut(isMutableIOSpec(ioSpec)), fused(isFusableIOSpec(ioSpec)),
        ioSpec(ioSpec) {}
  bool operator==(const KernelDeclVariadicTensorAdaptor &o) const {
    return mut == o.mut;
  }
};

struct KernelDeclScalarAdaptor {
  bool operator==(const KernelDeclScalarAdaptor &) const { return true; }
};

struct KernelDeclTileTensorAdaptor {
  bool operator==(const KernelDeclTileTensorAdaptor &) const { return true; }
};

struct KernelDeclContextPtrAdaptor {
  bool operator==(const KernelDeclContextPtrAdaptor &) const { return true; }
};

struct KernelDeclContextPtrListAdaptor {
  bool operator==(const KernelDeclContextPtrListAdaptor &) const {
    return true;
  }
};

struct KernelDeclGraphBuilderAdaptor {
  bool operator==(const KernelDeclGraphBuilderAdaptor &) const { return true; }
};

struct KernelDeclUnsupportedAdaptor {
  bool operator==(const KernelDeclUnsupportedAdaptor &) const { return false; }
};

/// Opaque argument type; its name is owned by the MLIR context via StringAttr.
struct KernelDeclOpaqueAdaptor {
  mlir::StringAttr typeName;
  bool operator==(const KernelDeclOpaqueAdaptor &o) const {
    if (typeName == o.typeName)
      return true;
    llvm::StringRef a = typeName.strref(), b = o.typeName.strref();
    return a.ends_with(b) || b.ends_with(a);
  }
};

using KernelDeclOperandVariant =
    std::variant<KernelDeclTensorAdaptor, KernelDeclVariadicTensorAdaptor,
                 KernelDeclScalarAdaptor, KernelDeclTileTensorAdaptor,
                 KernelDeclOpaqueAdaptor, KernelDeclContextPtrAdaptor,
                 KernelDeclContextPtrListAdaptor, KernelDeclGraphBuilderAdaptor,
                 KernelDeclUnsupportedAdaptor>;

/// One argument decoded from a kernel's stored LIT signature.
struct KernelDeclOperandAdaptor {
  struct SourceDescriptor {
    mlir::StringAttr sourceName;
    uint64_t position;
    bool isByRefResult;
  };

  std::optional<SourceDescriptor> sourceDescriptor;
  KernelDeclOperandVariant underlyingType;

  bool isDpsOutput() const {
    return sourceDescriptor.has_value() && !sourceDescriptor->isByRefResult;
  }

  bool isTensorType() const {
    return std::holds_alternative<KernelDeclTensorAdaptor>(underlyingType) ||
           std::holds_alternative<KernelDeclVariadicTensorAdaptor>(
               underlyingType);
  }

  bool isVariadicTensorType() const {
    return std::holds_alternative<KernelDeclVariadicTensorAdaptor>(
        underlyingType);
  }

  bool isMutableTensorType() const {
    if (const auto *t = std::get_if<KernelDeclTensorAdaptor>(&underlyingType))
      return t->mut;
    if (const auto *t =
            std::get_if<KernelDeclVariadicTensorAdaptor>(&underlyingType))
      return t->mut;
    return false;
  }

  bool isFusedTensorType() const {
    // Trait lane (Phase 5): fusion participation is read straight off the
    // distilled contract -- the single source of truth -- and never re-encoded
    // into a legacy fused `IOSpec`. The prologue/epilogue split (input vs
    // output `BindLambda`) still falls out of operand position in
    // `lowerToNormalKernel`, and the access form (load/store/transform) is
    // resolved later at emit; the IR only needs to know *whether* the arg is
    // fused here.
    if (std::optional<TensorArgContract> contract = getTraitContract())
      return contract->fused;
    if (const auto *t = std::get_if<KernelDeclTensorAdaptor>(&underlyingType))
      return t->fused;
    if (const auto *t =
            std::get_if<KernelDeclVariadicTensorAdaptor>(&underlyingType))
      return t->fused;
    return false;
  }

  /// The distilled trait contract for a trait-lane tensor argument, or nullopt
  /// for a legacy `ManagedTensorSlice` argument (or any non-tensor argument).
  /// Used by lowering to route trait args to `mogg.tensor_arg.__init__`.
  std::optional<TensorArgContract> getTraitContract() const {
    if (const auto *t = std::get_if<KernelDeclTensorAdaptor>(&underlyingType))
      return t->traitContract;
    return std::nullopt;
  }

  bool isTraitTensorType() const { return getTraitContract().has_value(); }

  bool isScalarType() const {
    return std::holds_alternative<KernelDeclScalarAdaptor>(underlyingType);
  }

  bool isTileType() const {
    return std::holds_alternative<KernelDeclTileTensorAdaptor>(underlyingType);
  }

  bool isOpaqueType() const {
    return std::holds_alternative<KernelDeclOpaqueAdaptor>(underlyingType);
  }

  bool isContextType() const {
    return std::holds_alternative<KernelDeclContextPtrAdaptor>(
               underlyingType) ||
           std::holds_alternative<KernelDeclContextPtrListAdaptor>(
               underlyingType);
  }

  bool isVariadicContextType() const {
    return std::holds_alternative<KernelDeclContextPtrListAdaptor>(
        underlyingType);
  }

  bool isBuilderType() const {
    return std::holds_alternative<KernelDeclGraphBuilderAdaptor>(
        underlyingType);
  }

  /// Launch arguments are the trailing "where to run" argument of a kernel
  /// function: a DeviceContext / DeviceContextArray for `execute` / `shape`,
  /// or a DeviceGraphBuilder for `build_graph`. They share the
  /// "must-appear-last" placement rule and are skipped together when
  /// aligning input arguments across function variants.
  bool isLaunchArgument() const { return isContextType() || isBuilderType(); }

  bool isUnsupportedType() const {
    return std::holds_alternative<KernelDeclUnsupportedAdaptor>(underlyingType);
  }

  bool isOpaqueWithTypeName(llvm::StringRef name) const {
    const auto *o = std::get_if<KernelDeclOpaqueAdaptor>(&underlyingType);
    if (!o)
      return false;
    // Strict equality. The stored typeName is produced by
    // KernelDeclAdaptor's buildOpaqueSymbol in the dotted-path form
    // ("root[.intermediate...]::leaf"). Callers must pass a name in the same
    // format; see e.g. kMojoInternalDpsIndexListTypeQualifiedName.
    return o->typeName.strref() == name;
  }

  bool operator==(const KernelDeclOperandAdaptor &other) const {
    return underlyingType == other.underlyingType;
  }

  template <typename StreamType>
  StreamType &printNested(StreamType &os, const std::string &nesting) const {
    if (sourceDescriptor.has_value())
      os << nesting << sourceDescriptor->sourceName.strref().str() << ": ";
    os << underlyingType;
    return os;
  }
};

/// Decoded view of one function (execute / shape / elementwise) built
/// directly from its FnTypeGeneratorType — no FnOp body or annotation
/// attributes required.
///
/// `attr` is the underlying `MOGG_KernelFunctionAttr`. Consumers read
/// `signature`, fn-level `params`, per-arg `arg_params`, and per-result
/// `result_params` through `attr.getSignature()`, `attr.getParams()`,
/// `attr.getArgParams()`, `attr.getResultParams()`. `attr` is null for
/// synthesized adaptors (see `synthesizeExecuteFromElementwise`); callers
/// that may consume a synthesized adaptor must null-check.
struct KernelDeclFunctionAdaptor {
  KernelFunctionAttr attr;
  llvm::SmallVector<KernelDeclOperandAdaptor> inputArguments;
  llvm::SmallVector<KernelDeclOperandAdaptor> outputArguments;
  std::optional<KernelDeclOperandAdaptor> outputResult;
  /// Fully-resolved Mojo symbol name for this function
  /// (e.g. `max.kernels.AddOp.execute`). Null for synthesized adaptors.
  mlir::StringAttr fullyResolvedName;

  /// Build a function adaptor from a stored LIT signature. `tensorArgContracts`
  /// is the import-baked `KernelFunctionAttr.tensor_arg_contracts` array (in
  /// signature-arg order) and is the single source of truth for the trait lane
  /// (Option A / D1/D7); pass it whenever it is available (the attr-bearing
  /// ctors do). When omitted/empty -- primitives and synthesized adaptors,
  /// which carry no trait args -- classification re-distills the written
  /// composition as a behavior-preserving fallback.
  explicit KernelDeclFunctionAdaptor(KGEN::LIT::FnTypeGeneratorType sig,
                                     mlir::ArrayAttr tensorArgContracts = {});

  KernelDeclFunctionAdaptor(KernelFunctionAttr attr,
                            StringAttr fullyResolvedName);

  KernelDeclFunctionAdaptor(ShapeFunctionDeclOp shapeDecl);

  /// Build an execute-compatible adaptor from an elementwise signature:
  /// promotes SIMD arguments to fused tensors, strips a trailing IndexList
  /// argument, and injects a synthetic fused-output tensor.
  static KernelDeclFunctionAdaptor
  synthesizeExecuteFromElementwise(KGEN::LIT::FnTypeGeneratorType sig);

  template <typename StreamType>
  StreamType &printNested(StreamType &os, const std::string &nesting) const {
    os << nesting << "(\n";
    for (const auto &arg : outputArguments)
      arg.printNested(os, nesting + "  [out] ") << ",\n";
    for (const auto &arg : inputArguments)
      arg.printNested(os, nesting + "  [in]  ") << ",\n";
    os << nesting << ")";
    if (outputResult.has_value())
      os << " -> " << *outputResult;
    return os;
  }

private:
  KernelDeclFunctionAdaptor() = default;
};

/// Top-level adaptor for a mogg.kernel.decl op. Decodes execute, shape,
/// elementwise, tile_elementwise, update_view, and build_graph signatures
/// directly from the stored LIT types without relying on MOGG pre-elaboration
/// annotation attributes.
struct KernelDeclAdaptor {
  mlir::StringAttr registeredOpName;
  /// Fully-resolved Mojo name of the kernel struct
  /// (e.g. `max.kernels.AddOp`). Built from `KernelDeclOp::getSymName()`.
  mlir::StringAttr fullyResolvedName;
  std::optional<KernelDeclFunctionAdaptor> executeFunction;
  std::optional<KernelDeclFunctionAdaptor> elementwiseFunction;
  std::optional<KernelDeclFunctionAdaptor> tileElementwiseFunction;
  std::optional<KernelDeclFunctionAdaptor> updateViewFunction;
  std::optional<KernelDeclFunctionAdaptor> buildGraphFunction;
  /// Struct-level ParamDeclAttrs shared by all functions of this kernel
  /// (e.g. `dtype`, `target`). Per-function `ParamDeclAttr`s are accessible
  /// via each function variant's owning `MOGG_KernelFunctionAttr`.
  mlir::ArrayAttr structParams;

  explicit KernelDeclAdaptor(KernelDeclOp decl);

  bool isElementwise() const { return elementwiseFunction.has_value(); }
  bool isTileElementwise() const { return tileElementwiseFunction.has_value(); }
  bool isViewKernel() const { return updateViewFunction.has_value(); }

  /// Return the function adaptor for a given MOGG function label
  /// (e.g. `kMoggExecuteFunctionLabel`, `kMoggShapeFunctionLabel`,
  ///  `kMoggUpdateViewFunctionLabel`, `kMoggElementwiseFunctionLabel`,
  ///  `kMoggBuildGraphFunctionLabel`).
  /// Returns nullptr if the label is not recognised or the function is absent.
  const KernelDeclFunctionAdaptor *
  getFunctionByLabel(llvm::StringRef label) const;
};

/// Adaptor for a mogg.primitive.decl op — a standalone (non-struct) primitive
/// Mojo function decorated with @register_internal (e.g. the `foreach`
/// driver or the view-materialisation kernel).
struct KernelFunctionDeclAdaptor {
  mlir::StringAttr registration;
  KernelDeclFunctionAdaptor function;

  explicit KernelFunctionDeclAdaptor(PrimitiveDeclOp decl);
};

//===----------------------------------------------------------------------===//
// Printers
//===----------------------------------------------------------------------===//

// KernelDeclAdaptor printers.

/// Print a KernelDeclOperandVariant.  Uses if-constexpr to keep ADL confined
/// to M::MOGG and avoid accidental dispatch into other namespaces.
template <typename StreamType>
StreamType &operator<<(StreamType &os, const KernelDeclOperandVariant &var) {
  std::visit(
      [&os](const auto &v) {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, KernelDeclTensorAdaptor>) {
          os << "Tensor[mut=" << (v.mut ? "true" : "false")
             << ", fused=" << (v.fused ? "true" : "false") << "]";
        } else if constexpr (std::is_same_v<T,
                                            KernelDeclVariadicTensorAdaptor>) {
          os << "*Tensor[mut=" << (v.mut ? "true" : "false")
             << ", fused=" << (v.fused ? "true" : "false") << "]";
        } else if constexpr (std::is_same_v<T, KernelDeclScalarAdaptor>) {
          os << "Scalar";
        } else if constexpr (std::is_same_v<T, KernelDeclOpaqueAdaptor>) {
          os << "<" << v.typeName.strref() << ">";
        } else if constexpr (std::is_same_v<T, KernelDeclContextPtrAdaptor>) {
          os << "Context";
        } else if constexpr (std::is_same_v<T,
                                            KernelDeclContextPtrListAdaptor>) {
          os << "*Context";
        } else if constexpr (std::is_same_v<T, KernelDeclGraphBuilderAdaptor>) {
          os << "GraphBuilder";
        } else if constexpr (std::is_same_v<T, KernelDeclUnsupportedAdaptor>) {
          os << "Unsupported";
        }
      },
      var);
  return os;
}

template <typename StreamType>
StreamType &operator<<(StreamType &os, const KernelDeclOperandAdaptor &op) {
  return op.printNested(os, "");
}

template <typename StreamType>
StreamType &operator<<(StreamType &os, const KernelDeclFunctionAdaptor &fn) {
  return fn.printNested(os, "");
}

template <typename StreamType>
StreamType &operator<<(StreamType &os, const KernelDeclAdaptor &adaptor) {
  os << "@register(" << adaptor.registeredOpName.strref() << ")\n";
  if (adaptor.executeFunction.has_value()) {
    os << "execute";
    adaptor.executeFunction->printNested(os, "  ");
    os << "\n";
  }
  if (adaptor.updateViewFunction.has_value()) {
    os << "update_view";
    adaptor.updateViewFunction->printNested(os, "  ");
    os << "\n";
  }
  return os;
}

template <typename StreamType>
StreamType &operator<<(StreamType &os,
                       const KernelFunctionDeclAdaptor &adaptor) {
  os << "@primitive(" << adaptor.registration.strref() << ")\n";
  os << "function";
  adaptor.function.printNested(os, "  ");
  os << "\n";
  return os;
}

} // namespace M::MOGG

#endif // GENERICML_GRAPHCOMPILER_MOGGDIALECT_ANALYSIS_KERNELDECLADAPTOR_H
