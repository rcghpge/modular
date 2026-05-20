//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// MOGG kernel signature introspection.
//
// This header collects everything needed to decode a Mojo kernel's signature
// at compile time: the MOGG attribute and decorator name constants, the type
// name strings emitted by the Mojo compiler, the `IOSpec` enum and its
// helpers, plus two adaptor families that consume them:
//
//   - `MojoKernelAdaptor` (legacy): wraps an annotated `LIT::FnOp` and reads
//     its `mogg.*` attribute payload to produce a typed view of the kernel.
//
//   - `KernelDeclAdaptor` (current): wraps a `mogg.kernel.decl` op and decodes
//     its stored `FnTypeGeneratorType` signatures directly, without relying on
//     MOGG pre-elaboration annotation attributes.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MOGGDIALECT_ANALYSIS_KERNELDECLADAPTOR_H
#define GENERICML_GRAPHCOMPILER_MOGGDIALECT_ANALYSIS_KERNELDECLADAPTOR_H

#include "GraphCompiler/GraphCompiler/MOGGDialect/MOGGOps.h"
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

//===----------------------------------------------------------------------===//
// MOGG attribute name constants
//===----------------------------------------------------------------------===//

// Attribute on generator ops to look for which marks the function as being a
// kernel.
constexpr llvm::StringLiteral kernelRegistrationAttr = "mogg.kernel";

// The name of the registered function at parse time.
constexpr llvm::StringLiteral mojoSymbolName = "mogg.mojo.name";

inline bool isKernel(mlir::Operation *gen) {
  return gen != nullptr && gen->hasAttr(kernelRegistrationAttr);
}

constexpr llvm::StringLiteral shapeFuncRegistrationAttr = "mogg.v1_shape_func";

constexpr llvm::StringLiteral IS_VIEW_ATTR = "mogg.view";
constexpr llvm::StringLiteral OUTLINED_ATTR = "mogg.outlined";

/// Module-level `DictionaryAttr` mapping simple struct names (e.g.
/// "MyStruct") to their fully-qualified Mojo names (e.g.
/// "foo.bar.MyStruct"). Populated by `mogg-import-packages` while walking
/// the loaded packages' `StructDeclOp` trees, and consumed by
/// `mogg-rewrite-opaque-types` to rewrite `!mo.opaque<"...">` references
/// in the module. The mapping is the canonical persistence form: it
/// travels with the module (e.g. when constant folding copies decls into
/// a sub-module) so downstream callers don't need to re-walk packages.
constexpr llvm::StringLiteral kOpaqueTypeMappingAttrName =
    "mogg.opaque_type_mapping";

/// Tracks the mojo parameter value for each of the input parameters.
constexpr llvm::StringLiteral MOGG_ARG_TYPE_NAMES = "mogg.arg_type_names";
constexpr llvm::StringLiteral MOGG_RESULT_TYPE_NAME = "mogg.result_type_name";
constexpr llvm::StringLiteral MOGG_INPUT_PARAM_TYPES = "mogg.input_param_types";

// The names as they appear in the lit source.
constexpr llvm::StringLiteral MOGG_ARG_SRC_NAMES = "mogg.arg_src_names";

/// Tracks the value witness entries of each argument and result type. A
/// DictionaryAttr where the key is the method name and the value is the
/// (potentially unresolved) method reference as a TypedAttr. The
/// canonical definitions live in `M::KGEN` alongside the Mojo parser; these
/// aliases re-export them so GraphCompiler-side consumers can keep using
/// `MOGG::MOGG_*_VALUE_WITNESSES`.
constexpr llvm::StringLiteral MOGG_ARGUMENT_VALUE_WITNESSES =
    KGEN::MOGG_ARGUMENT_VALUE_WITNESSES;
constexpr llvm::StringLiteral MOGG_RESULT_VALUE_WITNESSES =
    KGEN::MOGG_RESULT_VALUE_WITNESSES;

/// MOGG Intrinsic for the register kernel decorator.
constexpr llvm::StringLiteral MOGG_INTRINSIC_REGISTER =
    "mogg.intrinsic_register";

/// MOGG Intrinsic for the view kernel decorator.
constexpr llvm::StringLiteral MOGG_INTRINSIC_VIEW_KERNEL = "mogg.view_kernel";

/// MOGG Intrinsic for the for_each function
constexpr llvm::StringLiteral MOGG_INTRINSIC_FOR_EACH = "mogg.for_each";
constexpr llvm::StringLiteral MOGG_INTRINSIC_FOR_EACH_OUT_FUNC_OVERLOAD =
    "mogg.for_each.out_func";
constexpr llvm::StringLiteral MOGG_FOR_EACH_OUT_FUNC_PARAM = "out_func";

/// MOGG Intrinsic for a singleton for_each function that is used to drive
/// elemwise kernel execution
constexpr llvm::StringLiteral MOGG_INTRINSIC_ELEMWISE_FOR_EACH =
    "mogg.elemwise_for_each";

/// MOGG Intrinsic for a singleton function that is used to materialize a view
/// tensor.
constexpr llvm::StringLiteral MOGG_INTRINSIC_VIEW_MATERIALIZE =
    "mogg.view_materialize";

/// MOGG Intrinsic for the input / output lambda implementations.
constexpr llvm::StringLiteral MOGG_INTRINSIC_INPUT_FUSION_HOOK =
    "mogg.dps_input_fusion_hook";
constexpr llvm::StringLiteral MOGG_INTRINSIC_OUTPUT_FUSION_HOOK =
    "mogg.dps_output_fusion_hook";

constexpr llvm::StringLiteral
    MOGG_INTRINSIC_MIXED_PRECISION_OUTPUT_FUSION_HOOK =
        "mogg.dps_mixed_precision_output_fusion_hook";
constexpr llvm::StringLiteral
    MOGG_INTRINSIC_MIXED_PRECISION_COMPUTE_OUTPUT_FUSION_HOOK =
        "mogg.dps_mixed_precision_compute_output_fusion_hook";
constexpr llvm::StringLiteral MOGG_INTRINSIC_MIXED_PRECISION_INPUT_FUSION_HOOK =
    "mogg.dps_mixed_precision_input_fusion_hook";

//===----------------------------------------------------------------------===//
// DPS Tensor API definitions
//===----------------------------------------------------------------------===//

// Supported static methods under the registered structs.
static constexpr llvm::StringLiteral kMOGGKernelStructName =
    "mogg.kernel.struct";
static constexpr llvm::StringLiteral kMOGGExecuteFunctionLabel = "mogg.execute";
static constexpr llvm::StringLiteral kMOGGShapeFunctionLabel = "mogg.shape";
static constexpr llvm::StringLiteral kMOGGUpdateViewFunctionLabel =
    "mogg.update_view";
static constexpr llvm::StringLiteral kMOGGElementwiseFunctionLabel =
    "mogg.elementwise";

// An array of attributes, each of which correspond to an input argument type.
//
// If the associated argument is a parameterized tensor type, the attribute will
// be a dictionary attribute mapping the name of the unbound parameter
// (e.g. dtype) to the associated KGEN parameter decl ref.
//
// If the associated argument is not a supported tensor type, it will be the
// unit attribute.
static constexpr llvm::StringLiteral kKernelValueParameterAttrName =
    "mogg.value_params";

static constexpr llvm::StringLiteral kKernelResultParameterAttrName =
    "mogg.value_result_params";

/// Dictionary containing the default value assignments for parameters of
/// a kernel.
static constexpr llvm::StringLiteral kKernelDefaultParameterAttrName =
    "mogg.param_default_values";

/// Array of parameter names used as keys into kKernelDefaultParameterAttrName.
/// This is needed later in the lowering workflow as parameters must be
/// processed in declaration order.
static constexpr llvm::StringLiteral kKernelParameterNamesAttrName =
    "mogg.input_param_names";

// An array of StringAttr corresponding to each input argument type.
//
// If the associated string is non-empty, it refers to the name of the
// tensor spec parameter. This is often added by a specialized Mojo pass
// like autoparameterization.
static constexpr llvm::StringLiteral kKernelTensorSpecParameterAttrName =
    "mogg.tensor_spec_params";

// The generator level label for the target device for the kernel, e.g. cpu.
static constexpr llvm::StringLiteral kMOGGTargetLabel = "mogg.target";
static constexpr llvm::StringLiteral kMOGGTargetParameterName = "target";

// The name of the output_rank parameter which we support lowering specially.
static constexpr llvm::StringLiteral kMOGGOutputRankParameterName =
    "output_rank";

// Generator level annotations for fusion.
static constexpr llvm::StringLiteral kMOGGViewKernel = "mogg.view_kernel";
static constexpr llvm::StringLiteral kMOGGElementwiseKernel =
    "mogg.elementwise_kernel";

// The number of DPS (Destination Passing Style) output operands there are.
static constexpr llvm::StringLiteral kMOGGNumDPSOutputs =
    "mogg.num_dps_outputs";

// Annotations which indicate whether a fusion lambda is present. Needed for
// accumulation-type ops like matmul and conv where only the last store should
// be to the lambda.
static constexpr llvm::StringLiteral kMOGGLambdasHaveFusionLabel =
    "mogg.lambdas_have_fusion";
static constexpr llvm::StringLiteral kMOGGLambdasHaveFusionParameterName =
    "lambdas_have_fusion";

// Trace name for the fused kernel.
// Need to be used in the kernel code like this:
// `with Trace[TraceLevel.OP, target=target](_trace_name):`
static constexpr llvm::StringLiteral kMOGGTraceNameLabel = "mogg.trace_name";
static constexpr llvm::StringLiteral kMOGGTraceNameParameterName =
    "_trace_name";

// An ArrayAttr of indices which correspond which operands have fusion enabled.
static constexpr llvm::StringLiteral kMOGGFusableArgs = "mogg.fusable_args";

// An ArrayAttr of of the encoded IOSpec value for all kernel arguments.
static constexpr llvm::StringLiteral kMOGGArgsIOSpecs = "mogg.args_io_specs";

inline bool isExecuteFunc(mlir::Operation *gen) {
  return gen != nullptr && gen->hasAttr(kMOGGExecuteFunctionLabel);
}

inline bool isShapeFunc(mlir::Operation *gen) {
  return gen != nullptr && gen->hasAttr(kMOGGShapeFunctionLabel);
}

inline bool isUpdateViewFunc(mlir::Operation *gen) {
  return gen != nullptr && gen->hasAttr(kMOGGUpdateViewFunctionLabel);
}

inline bool isViewMaterializeFunc(mlir::Operation *gen) {
  return gen != nullptr && gen->hasAttr(MOGG_INTRINSIC_VIEW_MATERIALIZE);
}

inline bool isElemwiseForeachFunc(mlir::Operation *gen) {
  return gen != nullptr && gen->hasAttr(MOGG_INTRINSIC_ELEMWISE_FOR_EACH);
}

inline bool isExtensibilityFunc(mlir::Operation *gen) {
  return gen != nullptr && (gen->hasAttr(kMOGGExecuteFunctionLabel) ||
                            gen->hasAttr(kMOGGShapeFunctionLabel) ||
                            gen->hasAttr(kMOGGUpdateViewFunctionLabel) ||
                            gen->hasAttr(kMOGGElementwiseFunctionLabel));
}

//===----------------------------------------------------------------------===//
// DPS Tensor API type strings
//===----------------------------------------------------------------------===//

// The stored mojo type symbol name of Tensor type in extensibility kernels.
constexpr llvm::StringLiteral MOJO_DPS_TENSOR_TYPE_NAME =
    "tensor::ManagedTensorSlice";

constexpr llvm::StringLiteral MOJO_INTERNAL_DPS_SIMD_TYPE_NAME = "std::SIMD";

constexpr llvm::StringLiteral MOJO_INTERNAL_DPS_INT_TYPE_NAME = "std::Int";

constexpr llvm::StringLiteral MOJO_INTERNAL_DPS_UINT_TYPE_NAME = "std::UInt";

constexpr llvm::StringLiteral MOJO_INTERNAL_DPS_BOOL_TYPE_NAME = "std::Bool";

constexpr llvm::StringLiteral MOJO_INTERNAL_DPS_INDEX_LIST_TYPE_NAME =
    "std::IndexList";

// Fully-qualified form of IndexList's type name, matching the dotted format
// produced by KernelDeclAdaptor's buildOpaqueSymbol (every segment joined by
// '.', usable directly as a Mojo qualified name). The KernelDeclAdaptor-based
// code path stores opaque type names in this form, so isOpaqueWithTypeName
// checks against it directly.
constexpr llvm::StringLiteral MOJO_INTERNAL_DPS_INDEX_LIST_TYPE_QUALIFIED_NAME =
    "std.utils.index.IndexList";

// We support tuples of DPS tensors for operations with variadic input/outputs
// VariadicTensors is the unfused variant; the fused variants carry per-element
// fusion structs via a heterogeneous Tuple.
constexpr llvm::StringLiteral MOJO_VARIADIC_TENSORS_NAME =
    "tensor::VariadicTensors";
constexpr llvm::StringLiteral MOJO_FUSED_INPUT_VARIADIC_TENSORS_NAME =
    "tensor::_FusedInputVariadicTensors";
constexpr llvm::StringLiteral MOJO_FUSED_OUTPUT_VARIADIC_TENSORS_NAME =
    "tensor::_FusedOutputVariadicTensors";

// The stored mojo type symbol name of device contexts in extensibility kernels.
constexpr llvm::StringLiteral MOJO_EXTENSIBILITY_API_DEVICE_CONTEXT_TYPE_NAME =
    "std::DeviceContext";

// The stored mojo type symbol name of device-context lists in extensibility
// kernels (used by multi-device custom ops).
constexpr llvm::StringLiteral
    MOJO_EXTENSIBILITY_API_DEVICE_CONTEXT_LIST_TYPE_NAME =
        "std::DeviceContextList";

// These type names can be confusing. Inside the Graph Compiler code, the
// types show up like this:
// - StaticString -> std::StringSlice
// - String -> std::String
// - StringSlice -> std::Origin
constexpr llvm::StringLiteral MOJO_INTERNAL_STATIC_STRING_TYPE_NAME =
    "std::StringSlice";

constexpr llvm::StringLiteral MOJO_STATIC_STRING_TYPE_NAME =
    "std::StaticString";

// In the emit-mojo path, DType parameters appear as LIT struct types with
// type name "std::DType" rather than KGEN::DTypeType.
constexpr llvm::StringLiteral MOJO_DTYPE_TYPE_NAME = "std::DType";

constexpr llvm::StringLiteral MOJO_STRING_TYPE_NAME = "std::String";

constexpr llvm::StringLiteral MOJO_INTERNAL_STRING_SLICE_TYPE_NAME =
    "std::Origin";

//===----------------------------------------------------------------------===//
// Parameter Inference
//===----------------------------------------------------------------------===//

// Names of common params. We use these strings as the key to a parameter ref.
static constexpr llvm::StringLiteral kParameterMut = "mut";
static constexpr llvm::StringLiteral kParameterInput = "input";
static constexpr llvm::StringLiteral kParameterDType = "dtype";
static constexpr llvm::StringLiteral kParameterRank = "rank";
static constexpr llvm::StringLiteral kParameterIOSpec = "io_spec";
static constexpr llvm::StringLiteral kParameterStaticSpec = "static_spec";

// Used in variadics/tuple/SIMD types.
static constexpr llvm::StringLiteral kParameterSize = "size";
static constexpr llvm::StringLiteral kParameterStaticSpecs = "static_specs";

static constexpr bool kIOSpecImmutable = false;
static constexpr bool kIOSpecMutable = true;

static constexpr int64_t kIOSpecIOOutput = 0;
static constexpr int64_t kIOSpecIOInput = 1;
static constexpr int64_t kIOSpecIOFusedInput = 2;
static constexpr int64_t kIOSpecIOFusedOutput = 3;
static constexpr int64_t kIOSpecIOFusedComputeOutput = 31;

static constexpr int64_t kIOSpecIOUnknown = -1;

static constexpr llvm::StringLiteral kInputTensor = "InputTensor";
static constexpr llvm::StringLiteral kOutputTensor = "OutputTensor";
static constexpr llvm::StringLiteral kMutableInputTensor = "MutableInputTensor";
static constexpr llvm::StringLiteral kFusedInputTensor = "FusedInputTensor";
static constexpr llvm::StringLiteral kFusedOutputTensor = "FusedOutputTensor";
static constexpr llvm::StringLiteral kFusedComputeOutputTensor =
    "FusedComputeOutputTensor";

enum class IOSpec {
  InputTensor,        // Input tensor, read-only
  OutputTensor,       // Output tensor, write-only
  MutableInputTensor, // Input tensor that can be modified
  FusedInputTensor,
  FusedOutputTensor,
  FusedComputeOutputTensor,
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
// Variant visiting helpers (used by MojoKernelAdaptor)
//===----------------------------------------------------------------------===//

namespace detail {
template <typename T, typename... Ts>
constexpr bool is_one_of_v = (std::is_same_v<T, Ts> || ...);

template <typename... TargetTypes, typename Variant, typename Func>
void partial_visit(Func &&func, Variant &var) {
  std::visit(
      [&](auto &val) {
        using T = std::decay_t<decltype(val)>;
        if constexpr (is_one_of_v<T, TargetTypes...>)
          func(val);
      },
      var);
}
} // namespace detail

//===----------------------------------------------------------------------===//
// MojoKernelAdaptor (legacy: reads annotated LIT::FnOp)
//===----------------------------------------------------------------------===//

struct TensorOperandAdaptor {
  static constexpr llvm::StringLiteral typeName = MOJO_DPS_TENSOR_TYPE_NAME;
  // If the tensor is a mutable input or not.
  bool mut;
  bool fused;
  IOSpec ioSpec;
  TensorOperandAdaptor() = default;
  TensorOperandAdaptor(IOSpec ioSpec)
      : mut(isMutableIOSpec(ioSpec)), fused(isFusableIOSpec(ioSpec)),
        ioSpec(ioSpec) {}

  bool operator==(const TensorOperandAdaptor &other) const {
    // Not comparing fused because it can't be specified anywhere except in Mojo
    // (nothing to compare it to).
    return mut == other.mut;
  }
};

struct VariadicTensorOperandAdaptor : TensorOperandAdaptor {
  static constexpr llvm::StringLiteral typeName = MOJO_VARIADIC_TENSORS_NAME;
  bool operator==(const VariadicTensorOperandAdaptor &other) const {
    return mut == other.mut;
  }
};

struct ScalarOperandAdaptor {
  static constexpr llvm::StringLiteral typeName =
      MOJO_INTERNAL_DPS_SIMD_TYPE_NAME;
  bool operator==(const ScalarOperandAdaptor &other) const { return true; }
};

struct DeviceContextOperandAdaptor {
  static constexpr llvm::StringLiteral typeName =
      MOJO_EXTENSIBILITY_API_DEVICE_CONTEXT_TYPE_NAME;
  bool operator==(const DeviceContextOperandAdaptor &other) const {
    return true;
  }
};

struct DeviceContextListOperandAdaptor {
  static constexpr llvm::StringLiteral typeName =
      MOJO_EXTENSIBILITY_API_DEVICE_CONTEXT_LIST_TYPE_NAME;
  bool operator==(const DeviceContextListOperandAdaptor &other) const {
    return true;
  }
};

struct OpaqueOperandAdaptor {
  llvm::StringRef typeName;
  bool operator==(const OpaqueOperandAdaptor &other) const {
    if (typeName == other.typeName)
      return true;

    /// FIXME(GEX-2036): This is a gross hack. In `MOGGPreElab`, we maintain
    /// module names. In other parts of the stack, we use the leaf name.
    /// Therefore, we check if one name ends with the other. A proper fix would
    /// be to use the same name at all times.
    if (typeName.ends_with(other.typeName) ||
        other.typeName.ends_with(typeName))
      return true;

    return false;
  }
};

struct UnsupportedOperandAdaptor {
  bool operator==(const UnsupportedOperandAdaptor &other) const {
    return false;
  }
};

struct MojoKernelOperandSourceDescriptor {
  // The name of the variable in the original Mojo code.
  llvm::StringRef sourceName;
  // Its position in the function.
  uint64_t position;
  // If it was a return value that was promoted to a by ref result by the Mojo
  // compiler.
  bool isByRefResult;
};

using MojoKernelOperandVariant =
    std::variant<TensorOperandAdaptor, VariadicTensorOperandAdaptor,
                 ScalarOperandAdaptor, OpaqueOperandAdaptor,
                 DeviceContextOperandAdaptor, DeviceContextListOperandAdaptor,
                 UnsupportedOperandAdaptor>;

struct MojoKernelOperandAdaptor {
  MojoKernelOperandAdaptor() = default;

  // The source information which is optional in the case of a return value.
  std::optional<MojoKernelOperandSourceDescriptor> sourceDescriptor;
  // A union between all kinds of supported operands.
  MojoKernelOperandVariant underlyingType;

  MojoKernelOperandAdaptor(std::optional<uint64_t> positionInFunction,
                           mlir::StringAttr typeName,
                           mlir::ArrayAttr argumentSourceNames,
                           mlir::ArrayAttr argsIoSpecs,
                           bool isByRefResult = false,
                           bool promoteSIMDToFusedTensor = false);

  static MojoKernelOperandAdaptor buildElementwiseOutputOperand();

  template <typename StreamType>
  StreamType &printNested(StreamType &os, const std::string &nesting) const {
    if (sourceDescriptor.has_value())
      os << nesting << *sourceDescriptor;

    os << underlyingType;

    return os;
  }

  bool operator==(const MojoKernelOperandAdaptor &other) const {
    if (underlyingType.index() != other.underlyingType.index())
      return false;

    return std::visit(
        [&](auto &&arg) { return underlyingType == other.underlyingType; },
        underlyingType);
  }

  bool operator!=(const MojoKernelOperandAdaptor &other) const {
    return !(*this == other);
  }

  bool isDpsOutput() const {
    return sourceDescriptor.has_value() && !sourceDescriptor->isByRefResult;
  }

  bool isVariadicTensorType() const {
    return std::holds_alternative<VariadicTensorOperandAdaptor>(underlyingType);
  }

  bool isOpaqueType() const {
    return std::holds_alternative<OpaqueOperandAdaptor>(underlyingType);
  }

  bool isUnsupportedType() const {
    return std::holds_alternative<UnsupportedOperandAdaptor>(underlyingType);
  }

  bool isContextType() const {
    return std::holds_alternative<DeviceContextOperandAdaptor>(
               underlyingType) ||
           std::holds_alternative<DeviceContextListOperandAdaptor>(
               underlyingType);
  }

  bool isVariadicContextType() const {
    return std::holds_alternative<DeviceContextListOperandAdaptor>(
        underlyingType);
  }

  bool isScalarType() const {
    return std::holds_alternative<ScalarOperandAdaptor>(underlyingType);
  }

  bool isTensorType() const {
    return std::holds_alternative<TensorOperandAdaptor>(underlyingType) ||
           std::holds_alternative<VariadicTensorOperandAdaptor>(underlyingType);
  }

  bool isFusedTensorType() const {
    if (!isTensorType())
      return false;

    bool fused;
    detail::partial_visit<TensorOperandAdaptor, VariadicTensorOperandAdaptor>(
        [&](auto &&obj) { fused = obj.fused; }, underlyingType);

    return fused;
  }

  bool isTensorWithIOSpec(IOSpec spec) const {
    if (!isTensorType())
      return false;

    bool match = false;
    detail::partial_visit<TensorOperandAdaptor, VariadicTensorOperandAdaptor>(
        [&](auto &&obj) { match = (spec == obj.ioSpec); }, underlyingType);

    return match;
  }

  bool isMutableTensorType() const {
    if (!isTensorType())
      return false;

    bool isMutable;
    detail::partial_visit<TensorOperandAdaptor, VariadicTensorOperandAdaptor>(
        [&](auto &&obj) { isMutable = obj.mut; }, underlyingType);

    return isMutable;
  }

  bool isOpaqueWithTypeName(llvm::StringRef name) const {
    if (!std::holds_alternative<OpaqueOperandAdaptor>(underlyingType))
      return false;

    return std::get<OpaqueOperandAdaptor>(underlyingType).typeName == name;
  }
};

struct MojoKernelFunctionAdaptor {
  // The underlying LIT or KGEN function.
  // Marked mutable because getSymName is not const for some reason.
  mutable KGEN::LIT::FnOp mojoCode;
  // Input arguments.
  llvm::SmallVector<MojoKernelOperandAdaptor> inputArguments;
  // Output arguments (DPS).
  llvm::SmallVector<MojoKernelOperandAdaptor> outputArguments;
  // Output result which could be an argument if the function throws.
  std::optional<MojoKernelOperandAdaptor> outputResult;

  MojoKernelFunctionAdaptor(KGEN::LIT::FnOp op) : mojoCode(op) {
    auto argumentTypesNames =
        mojoCode->template getAttrOfType<mlir::ArrayAttr>(MOGG_ARG_TYPE_NAMES);
    auto resultTypeNameAttr = mojoCode->getAttr(MOGG_RESULT_TYPE_NAME);
    auto numberOfOutputArgumentsAttr =
        mojoCode->template getAttrOfType<mlir::IntegerAttr>(kMOGGNumDPSOutputs);
    auto argumentSourceNames =
        mojoCode->template getAttrOfType<mlir::ArrayAttr>(MOGG_ARG_SRC_NAMES);
    auto argsIoSpecsAttr =
        mojoCode->template getAttrOfType<mlir::ArrayAttr>(kMOGGArgsIOSpecs);

    uint64_t begOfInputArguments =
        numberOfOutputArgumentsAttr ? numberOfOutputArgumentsAttr.getInt() : 0;
    auto funcTypeGenerator = mojoCode.getFuncTypeGenerator().getBody();
    bool resultAsArgument = false;
    for (uint64_t i = 0; i < mojoCode.getNumArguments(); ++i) {
      if (funcTypeGenerator.getArgConvention(i) ==
          KGEN::ArgConvention::ByRefResult) {
        resultAsArgument = true;
        break;
      }
    }
    bool isThrow = funcTypeGenerator.isThrows();
    uint64_t numberOfArgumentsRelatedToByrefResult =
        isThrow ? 2 : resultAsArgument;
    size_t endOfInputArguments =
        argumentTypesNames.size() - numberOfArgumentsRelatedToByrefResult;

    for (size_t i = begOfInputArguments; i < endOfInputArguments; ++i) {
      auto argTypeName =
          mlir::dyn_cast<mlir::StringAttr>(argumentTypesNames[i]);
      inputArguments.emplace_back(i, argTypeName, argumentSourceNames,
                                  argsIoSpecsAttr, begOfInputArguments);
    }

    for (size_t i = 0; i < begOfInputArguments; ++i) {
      auto argTypeName =
          mlir::dyn_cast<mlir::StringAttr>(argumentTypesNames[i]);
      // Providing no mutable tensor positions because outputs can't be mutable
      // inputs.
      outputArguments.emplace_back(i, argTypeName, argumentSourceNames,
                                   argsIoSpecsAttr);
    }

    if (resultAsArgument) {
      auto argTypeName = mlir::dyn_cast<mlir::StringAttr>(
          argumentTypesNames[endOfInputArguments +
                             numberOfArgumentsRelatedToByrefResult - 1]);
      if (argTypeName) {
        outputResult = MojoKernelOperandAdaptor(
            endOfInputArguments + numberOfArgumentsRelatedToByrefResult - 1,
            argTypeName, argumentSourceNames, argsIoSpecsAttr,
            /*isByRefResult=*/true);
      }
    } else if (resultTypeNameAttr) {
      auto resultTypeName =
          mlir::dyn_cast<mlir::StringAttr>(resultTypeNameAttr);
      // Providing no mutable or fused tensor positions because they don't make
      // sense for output results.
      outputResult = MojoKernelOperandAdaptor(
          {}, resultTypeName, argumentSourceNames, argsIoSpecsAttr);
    }
  }

  // Synthesize the execute function from an elementwise method.
  static MojoKernelFunctionAdaptor
  synthesizeExecuteFromElementwise(KGEN::LIT::FnOp op) {
    KGEN::LIT::FnOp mojoCode = op;
    MojoKernelFunctionAdaptor res;
    res.mojoCode = op;
    auto argumentTypesNames =
        mojoCode->template getAttrOfType<mlir::ArrayAttr>(MOGG_ARG_TYPE_NAMES);
    auto numberOfOutputArgumentsAttr =
        mojoCode->template getAttrOfType<mlir::IntegerAttr>(kMOGGNumDPSOutputs);
    auto argumentSourceNames =
        mojoCode->template getAttrOfType<mlir::ArrayAttr>(MOGG_ARG_SRC_NAMES);
    auto argsIoSpecsAttr =
        mojoCode->template getAttrOfType<mlir::ArrayAttr>(kMOGGArgsIOSpecs);

    uint64_t begOfInputArguments =
        numberOfOutputArgumentsAttr ? numberOfOutputArgumentsAttr.getInt() : 0;
    auto funcTypeGenerator = mojoCode.getFuncTypeGenerator().getBody();
    bool resultAsArgument = false;
    for (uint64_t i = 0; i < mojoCode.getNumArguments(); ++i) {
      if (funcTypeGenerator.getArgConvention(i) ==
          KGEN::ArgConvention::ByRefResult) {
        resultAsArgument = true;
        break;
      }
    }
    bool isThrow = funcTypeGenerator.isThrows();
    uint64_t numberOfArgumentsRelatedToByrefResult =
        isThrow ? 2 : resultAsArgument;
    size_t endOfInputArguments =
        argumentTypesNames.size() - numberOfArgumentsRelatedToByrefResult;

    for (size_t i = begOfInputArguments; i < endOfInputArguments; ++i) {
      auto argTypeName =
          mlir::dyn_cast<mlir::StringAttr>(argumentTypesNames[i]);
      if (i + 1 == endOfInputArguments &&
          argTypeName == MOJO_INTERNAL_DPS_INDEX_LIST_TYPE_NAME) {
        // The last argument of an elementwise function can be an IndexList
        // argument. Skip it for the execute function.
        continue;
      }
      res.inputArguments.emplace_back(i, argTypeName, argumentSourceNames,
                                      argsIoSpecsAttr, begOfInputArguments,
                                      /*promoteSIMDToFusedTensor=*/true);
    }

    // Manually build the output tensor for the elementwise op.
    res.outputArguments.push_back(
        MojoKernelOperandAdaptor::buildElementwiseOutputOperand());
    return res;
  }

  inline bool isView() { return mojoCode->hasAttr(kMOGGViewKernel); }

  template <typename StreamType>
  StreamType &printNested(StreamType &os, const std::string &nesting) const {
    llvm::StringRef sourceName;
    if (mojoCode.getSymName().has_value())
      sourceName = *mojoCode.getSymName();

    if (auto name = mojoCode->getAttr("sourceName"))
      sourceName = mlir::cast<mlir::StringAttr>(name).strref();

    os << nesting << sourceName.str() << "\n" << nesting << "(\n";

    for (auto &arg : outputArguments)
      arg.printNested(os, nesting + "  [out] ") << ",\n";

    for (auto &arg : inputArguments)
      arg.printNested(os, nesting + "  [in]  ") << ",\n";

    os << nesting << ")";
    if (outputResult.has_value())
      os << " -> " << *outputResult;

    return os;
  }

private:
  MojoKernelFunctionAdaptor() = default;
};

// A helper struct that provides useful information about a kernel after being
// lowered through the Mojo compiler.
struct MojoKernelAdaptor {
  llvm::StringRef originalStructName;
  // The name of the op it implements.
  llvm::StringRef registeredOpName;
  // The kernel's entry point.
  std::optional<MojoKernelFunctionAdaptor> executeFunction;
  // The kernel's optional shape function.
  std::optional<MojoKernelFunctionAdaptor> shapeFunction;

  // The kernel's optional elementwise function.
  std::optional<MojoKernelFunctionAdaptor> elementwiseFunction;

  MojoKernelAdaptor(llvm::StringRef structName, KGEN::LIT::FnOp execute,
                    KGEN::LIT::FnOp shape, KGEN::LIT::FnOp elementwise)
      : originalStructName(structName) {
    if (execute) {
      executeFunction.emplace(execute);
      registeredOpName = execute
                             ->template getAttrOfType<mlir::StringAttr>(
                                 kMOGGExecuteFunctionLabel)
                             .strref();
    }

    if (shape) {
      shapeFunction.emplace(shape);
      registeredOpName = shape
                             ->template getAttrOfType<mlir::StringAttr>(
                                 kMOGGShapeFunctionLabel)
                             .strref();
    }

    if (elementwise) {
      elementwiseFunction.emplace(elementwise);
      registeredOpName = elementwise
                             ->template getAttrOfType<mlir::StringAttr>(
                                 kMOGGElementwiseFunctionLabel)
                             .strref();
      // TODO: We should always synthetize the execute signature. (GEX-2453).
      if (!execute) {
        // We synthetize the execute function signature from the elementwise
        // function.
        executeFunction =
            MojoKernelFunctionAdaptor::synthesizeExecuteFromElementwise(
                elementwise);
      }
    }
  }

  /// TODO: (GEX-1994) Remove this.
  bool isCoreMOOperation() const { return registeredOpName.starts_with("mo."); }

  bool isElementwise() const { return elementwiseFunction.has_value(); }
};

//===----------------------------------------------------------------------===//
// KernelDeclAdaptor (current: reads mogg.kernel.decl op signatures directly)
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

struct KernelDeclContextPtrAdaptor {
  bool operator==(const KernelDeclContextPtrAdaptor &) const { return true; }
};

struct KernelDeclContextPtrListAdaptor {
  bool operator==(const KernelDeclContextPtrListAdaptor &) const {
    return true;
  }
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
                 KernelDeclScalarAdaptor, KernelDeclOpaqueAdaptor,
                 KernelDeclContextPtrAdaptor, KernelDeclContextPtrListAdaptor,
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
    // format; see e.g. MOJO_INTERNAL_DPS_INDEX_LIST_TYPE_QUALIFIED_NAME.
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
/// elementwise, and update_view signatures directly from the stored LIT types
/// without relying on MOGG pre-elaboration annotation attributes.
struct KernelDeclAdaptor {
  mlir::StringAttr registeredOpName;
  /// Fully-resolved Mojo name of the kernel struct
  /// (e.g. `max.kernels.AddOp`). Built from `KernelDeclOp::getSymName()`.
  mlir::StringAttr fullyResolvedName;
  std::optional<KernelDeclFunctionAdaptor> executeFunction;
  std::optional<KernelDeclFunctionAdaptor> shapeFunction;
  std::optional<KernelDeclFunctionAdaptor> elementwiseFunction;
  std::optional<KernelDeclFunctionAdaptor> updateViewFunction;
  /// Struct-level ParamDeclAttrs shared by all functions of this kernel
  /// (e.g. `dtype`, `target`). Per-function `ParamDeclAttr`s are accessible
  /// via each function variant's owning `MOGG_KernelFunctionAttr`.
  mlir::ArrayAttr structParams;

  explicit KernelDeclAdaptor(KernelDeclOp decl);

  bool isElementwise() const { return elementwiseFunction.has_value(); }
  bool isViewKernel() const { return updateViewFunction.has_value(); }

  /// Return the function adaptor for a given MOGG function label
  /// (e.g. `kMOGGExecuteFunctionLabel`, `kMOGGShapeFunctionLabel`,
  ///  `kMOGGUpdateViewFunctionLabel`, `kMOGGElementwiseFunctionLabel`).
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

// MojoKernelAdaptor (legacy) printers.

template <typename StreamType>
StreamType &operator<<(StreamType &os, const MojoKernelAdaptor &adaptor) {
  os << "@register(" << adaptor.registeredOpName.str() << ")\n";
  os << "struct " << adaptor.originalStructName.str();
  if (adaptor.executeFunction.has_value()) {
    os << "\n";
    adaptor.executeFunction->printNested(os, "  ");
  }

  if (adaptor.shapeFunction.has_value()) {
    os << "\n";
    adaptor.shapeFunction->printNested(os, "  ");
  }

  return os;
}

template <typename StreamType>
StreamType &operator<<(StreamType &os, const MojoKernelFunctionAdaptor &func) {
  return func.printNested(os, "");
}

template <typename StreamType>
StreamType &operator<<(StreamType &os,
                       const MojoKernelOperandAdaptor &operand) {
  return operand.printNested(os, "");
}

template <typename StreamType>
StreamType &operator<<(StreamType &os, const MojoKernelOperandVariant &var) {
  std::visit([&](auto &&arg) { os << arg; }, var);
  return os;
}

template <typename StreamType>
StreamType &operator<<(StreamType &os,
                       const MojoKernelOperandSourceDescriptor &desc) {
  return os << desc.sourceName.str() << ": ";
}

static inline const char *bool2str(bool val) { return val ? "True" : "False"; }

template <typename StreamType>
StreamType &operator<<(StreamType &os, const TensorOperandAdaptor &tensor) {
  os << "Tensor[mut=" << bool2str(tensor.mut)
     << ", fused=" << bool2str(tensor.fused) << "]";
  return os;
}

template <typename StreamType>
StreamType &operator<<(StreamType &os,
                       const VariadicTensorOperandAdaptor &tensors) {
  os << "*" << *static_cast<const TensorOperandAdaptor *>(&tensors);
  return os;
}

template <typename StreamType>
StreamType &operator<<(StreamType &os, const ScalarOperandAdaptor &) {
  os << "Scalar";
  return os;
}

template <typename StreamType>
StreamType &operator<<(StreamType &os, const DeviceContextOperandAdaptor &) {
  os << "Context";
  return os;
}

template <typename StreamType>
StreamType &operator<<(StreamType &os,
                       const DeviceContextListOperandAdaptor &) {
  os << "*Context";
  return os;
}

template <typename StreamType>
StreamType &operator<<(StreamType &os, const OpaqueOperandAdaptor &opaque) {
  os << "<" << opaque.typeName.str() << ">";
  return os;
}

template <typename StreamType>
StreamType &operator<<(StreamType &os,
                       const UnsupportedOperandAdaptor &opaque) {
  os << "Unsupported";
  return os;
}

// KernelDeclAdaptor (current) printers.

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
