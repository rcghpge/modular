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

#include "GraphCompiler/GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GraphCompiler/GraphCompiler/MOGGDialect/Support/MojoIdentifiers.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/LITDialect/LITTypes.h"
#include "KGEN/LITDialect/LITUtils.h"
#include "mlir/IR/Attributes.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
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
};

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
    if (const auto *t = std::get_if<KernelDeclTensorAdaptor>(&underlyingType))
      return t->fused;
    if (const auto *t =
            std::get_if<KernelDeclVariadicTensorAdaptor>(&underlyingType))
      return t->fused;
    return false;
  }

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
  /// function: a DeviceContext / DeviceContextList for `execute` / `shape`,
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

  explicit KernelDeclFunctionAdaptor(KGEN::LIT::FnTypeGeneratorType sig);

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
  std::optional<KernelDeclFunctionAdaptor> shapeFunction;
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
  if (adaptor.shapeFunction.has_value()) {
    os << "shape";
    adaptor.shapeFunction->printNested(os, "  ");
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
