//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
#ifndef GENERICML_INTERFACES_EMITMOJOSTATE_H
#define GENERICML_INTERFACES_EMITMOJOSTATE_H

#include "GraphCompiler/GraphCompiler/MGPDialect/Dialect/MGPTypes.h"
#include "GraphCompiler/GraphCompiler/MODialect/MOAttrs.h"
#include "GraphCompiler/GraphCompiler/MODialect/MOTypes.h"
#include "GraphCompiler/GraphCompiler/MOGGDialect/MOGGEnums.h"
#include "GraphCompiler/GraphCompiler/MOGGDialect/MOGGTypes.h"
#include "GraphCompiler/GraphCompiler/MOGGDialect/Support/MojoBuilder.h"
#include "GraphCompiler/GraphCompiler/MOGGDialect/Support/MojoIdentifiers.h"

#include "KGEN/KGENDialect/KGENParameters.h"
#include "KGEN/KGENDialect/KGENTypes.h"
#include "Support/AssertStream.h"

namespace M::MOGG {
// Forward declarations to avoid circular includes:
// MOGGOps.h -> EmitsMojoInterface.h -> EmitMojoState.h ->
// {KernelDeclAdaptor,MOGGOps}.h -> MOGGOps.h
class MojoRegistry;
struct KernelDeclFunctionAdaptor;
struct KernelDeclAdaptor;
class PrimitiveDeclOp;
class ShapeFunctionDeclOp;

/// This struct holds all the necessary state to lower a MOGG operation into
/// Mojo text.
struct EmitMojoState {
  EmitMojoState(MojoRegistry &registry, bool enableOpTraceComments = false,
                bool enableComputeLambdas = false,
                bool enableDebugDumpFusedTensorSpecs = false)
      : registry(registry), enableComputeLambdas(enableComputeLambdas),
        enableOpTraceComments(enableOpTraceComments),
        enableDebugDumpFusedTensorSpecs(enableDebugDumpFusedTensorSpecs) {}

  /// The Mojo builder used to generate the text.
  MojoBuilder builder;
  /// Separate builder for module-scope struct definitions (fusion structs).
  /// Struct definitions are discovered mid-function during body emission
  /// (e.g. from BindLambdaOp), so they cannot be written to the main builder
  /// which is mid-function at that point. The output is concatenated after the
  /// main builder's output; Mojo allows file-scope declarations in any order.
  MojoBuilder structDefsBuilder;

  /// True once the trait-lane dense tensor-argument helper definitions
  /// (`_MoggTensorIO` + `_MoggDenseTensorArg`) have been emitted into
  /// `structDefsBuilder`. `TensorArgInitOp::emitMojo` emits them inline on
  /// first use and references them thereafter, so they live only inside the
  /// emitted Mojo module (no committed library type yet). Guards against
  /// re-emitting the definitions for every trait tensor argument.
  bool emittedDenseTensorArgDefs = false;
  /// One-per-module guard for `_MoggTensorBufferCapture` (region→stub
  /// boundary).
  bool emittedTensorBufferCaptureDefs = false;
  /// Same one-per-module guard for the trait-lane FUSED input-load definitions
  /// (`kInputLoadFusionDefs`: the `InputLoadFusion` trait +
  /// `_MoggInputLoadFusionTensor` wrapper), emitted by
  /// `emitInputLoadFusionFunctor` when a bind feeds a trait kernel's fused-load
  /// operand (Phase 5).
  bool emittedInputLoadFusionArgDefs = false;
  /// Same one-per-module guard for the trait-lane FUSED output-store
  /// definitions
  /// (`kOutputStoreFusionDefs`: the `OutputStoreFusion` trait +
  /// `_MoggOutputStoreFusionTensor` wrapper), emitted by
  /// `emitOutputStoreFusionFunctor` (Phase 5).
  bool emittedOutputStoreFusionArgDefs = false;
  /// Same one-per-module guard for the trait-lane FUSED output-transform
  /// (compute-output) definitions (`kTransformFusionDefs`: the
  /// `TransformFusion` trait + dual-nature `_MoggTransformFusionTensor`
  /// wrapper), emitted by `emitTransformFusionFunctor` (Phase 5).
  bool emittedTransformFusionArgDefs = false;
  /// One-per-module import of trait marker traits for shape-function stubs that
  /// still use generic `DenseTensor` stub parameters.
  bool emittedTraitMarkerImports = false;
  /// The actual return values of the `mogg.generic.compiled.region`. This is
  /// used by ops that feed into `mogg.output` (ex: AsyncReadyOp or AsyncPack*).
  SmallVector<MojoValue> mojoReturnValues;
  /// Kernels that dense resources will get these in the void** input after
  /// inputs and outputs.
  SmallVector<std::tuple<MojoValue, MojoValue>> mojoConstantResources;

  MojoRegistry &registry;

  /// In some cases values must be kept alive using `_ = someVar` at the end of
  /// the function. This is because mojo Compiler isn't smart enough in some
  /// cases to detect the right lifetime. This is mostly around lambdas, and
  /// should be fixed by the new closures work.
  SmallVector<MojoValue> keepAliveValues;

  /// Tracks the emitted Mojo variable names for each TaskHandle value.
  /// Populated by CallStubAsyncOp::emitMojo, consumed by AsyncJoinOp::emitMojo.
  struct TaskHandleInfo {
    MojoValue task;  // The Task variable from create_task()
    MojoValue error; // The Optional[Error] variable
  };
  llvm::DenseMap<Value, TaskHandleInfo> taskHandleInfos;

  /// Keep track of the mojo arguments / parameters for the current function
  /// being lowered.
  FunctionBody currentFunctionBody;

  /// If true, generate compute lambdas when possible.
  bool enableComputeLambdas;

  /// The top-level module being lowered. Used to resolve the per-device
  /// `mo.device_info_mapping` (e.g. the `tile_based_fusion` tile-codegen
  /// attribute).
  mlir::ModuleOp moduleOp;

  /// Name of the entry point of the current kernel
  std::string currentKernelName;

  /// Maps a BindLambdaOp output value to how it was lowered (a fusion struct).
  mlir::DenseMap<Value, EmittedLambdaLowering> lambdaLowerings;

  /// Record a fusion struct lowering for a BindLambdaOp output value.
  void setFusionStructLoweringFor(Value val, MojoValue instance,
                                  StringLiteral bindMethod, MojoValue typeName);

private:
  /// A mapping between mlir::Value to MojoValue (text).
  mlir::DenseMap<Value, MojoValue> mlirValueToMojoValue;

  /// Enables the emission of additional comments detailing IR ops that are
  /// currently being processed
  bool enableOpTraceComments;

  /// Enable the emission of print statements inside the fused lambda to debug
  /// the fused tensor specs.
  bool enableDebugDumpFusedTensorSpecs;

public:
  bool isDebugDumpFusedTensorSpecsEnabled() const {
    return enableDebugDumpFusedTensorSpecs;
  }
  // Helper to create a kernel wrapper function with a given body
  using KernelBodyBuilder =
      llvm::function_ref<void(const MOGG::FunctionBody &)>;

  /// Get the Mojovalue mapped to `val`.
  const MojoValue &getMojoValueFor(Value val) const {
    auto it = mlirValueToMojoValue.find(val);
    ASSERT_STREAM(it != mlirValueToMojoValue.end(),
                  << "value not bound to any MojoValue");
    return it->second;
  }

  /// Set the mapping from `val` to `mojoVal`.
  void setMojoValueFor(Value val, MojoValue mojoVal) {
    [[maybe_unused]] auto inserted =
        mlirValueToMojoValue.insert(std::make_pair(val, mojoVal)).second;
    ASSERT_STREAM(inserted, << "value already bound to a MojoValue");
  }

  /// Mark a value as being kept alive.
  /// Will generate `_ = value` at the end of the function.
  void markAsKeepAlive(MojoValue value) { keepAliveValues.push_back(value); }

  /// Generate all keep alive statements for the previous calls to
  /// `markAsKeepAlive`.
  void genAllKeepAliveStatements() {
    for (const MojoValue &val : keepAliveValues)
      builder.keepAlive(val);
    keepAliveValues.clear();
  }

  /// Try to lower `op` to Mojo using the EmitMojoInterface.
  /// Returns an error if lowering failed.
  LogicalResult lowerOpToMojo(Operation *op);

  /// Convert an mlir::Type to a MojoValue representing the type in Mojo text.
  /// For parameterized opaque types, this returns LegacyOpaquePointer.
  /// Use mlirOpaqueTypeToMojoType for the full parameterized type string.
  static MojoValue mlirTypeToMojoType(Type mlirType) {
    if (isa<MGP::DeviceBufferType>(mlirType)) {
      // Every `!mgp.buffer` value lowers to the storage-carrying composite;
      // producers build it and consumers unwrap its `.view` internally.
      return kStructOwnedByteBuffer;
    } else if (isa<MEF::ChainType>(mlirType)) {
      return kComptimeOpaquePointer;
    } else if (isa<MGP::DeviceContextType, ContextType>(mlirType)) {
      return kStructDeviceContext;
    } else if (isa<MGP::DeviceGraphType>(mlirType)) {
      return kStructDeviceGraph;
    } else if (isa<MGP::DeviceGraphBuilderType>(mlirType)) {
      return kStructDeviceGraphBuilder;
    } else if (isa<MGP::DeviceGraphNodeType>(mlirType)) {
      return kStructDeviceGraphNode;
    } else if (auto integerTy = dyn_cast<IntegerType>(mlirType)) {
      if (integerTy.getWidth() == 1)
        return kStructBool;
      return kStructInt;
    } else if (auto tensorType = dyn_cast<MO::TensorType>(mlirType)) {
      return mlirTensorTypeToMojoType(tensorType);
    } else if (auto tensorSpecType = dyn_cast<MGP::TensorSpecType>(mlirType)) {
      return MojoValue(kStructIndexList)[int64_t(tensorSpecType.getRank())];
    } else if (isa<MGP::ContextType>(mlirType)) {
      return kStructStateContext;
    } else if (isa<IndexType>(mlirType)) {
      return kStructInt;
    } else if (auto scalarTy = dyn_cast<MO::ScalarType>(mlirType)) {
      // Fusion struct fields can capture non-tensor operands from the fused
      // op body (e.g. integer/bool constants from ConstantScalarOp).  The
      // SIMDType branch below only covers KGEN-level types; this handles
      // MO-dialect scalars that appear before lowering.
      DType dtype = scalarTy.getDtype();
      if (dtype.isBool())
        return kStructBool;
      assert(dtype.isInt() &&
             "only bool/integer MO::ScalarType expected in fusion captures; "
             "extend this to handle float/other dtypes if needed");
      return kStructInt;
    } else if (auto simdType = dyn_cast<KGEN::SIMDType>(mlirType)) {
      /// FIXME(GEX-2867)
      // The only reason we would be manipulating SIMD types in the IR is for
      // doing shape calculations which all operate on index/Int.
      if (simdType.getResolvedDType()->isBool())
        return kStructBool;
      return kStructInt;
    } else if (auto shapeResultType = dyn_cast<ShapeFuncResultType>(mlirType)) {
      MOGG::MojoValue rankParam = MOGG::MojoValue(shapeResultType.getRank());
      return MOGG::MojoValue(MOGG::kStructIndexList)[rankParam];
    } else if (isa<MGP::OpaqueType, MO::OpaqueType>(mlirType)) {
      // Opaque types are represented as OpaquePointer in Mojo
      return kComptimeOpaquePointer;
    } else {
      ASSERT_STREAM(false, << "Unsupported mlir type " << mlirType);
      llvm_unreachable("Unsupported mlir type");
    }
  }

  /// Convert an MO::OpaqueType to its fully qualified Mojo type name.
  /// Returns just the qualified name without parameters.
  /// Example: !mo.opaque<"root::SIMDPair", ...> becomes "root.SIMDPair"
  /// Use getOpaqueTypeParameters() to get the parameters as MojoValues,
  /// then apply them using MojoValue::operator[].
  static MojoValue mlirOpaqueTypeToMojoType(MO::OpaqueType opaqueType);

  /// Extract parameters from an MO::OpaqueType as MojoValues.
  /// Returns empty vector if no parameters.
  /// Example: !mo.opaque<"SIMDPair", {S0 = 4, S1 = 8}> returns [4, 8]
  static SmallVector<MojoValue>
  getOpaqueTypeParameters(MO::OpaqueType opaqueType);

  static MojoValue mlirDTypeToMojoType(DType dtype) {
    KGEN::KGENDType kgenDtype(dtype);
    return MojoValue(kStructDType).get(kgenDtype.getAsString(/*libForm*/ true));
  }

  static MojoValue mlirTensorTypeToMojoType(MO::TensorType tensorType) {
    MO::IOKindAttr ioKind = tensorType.getIOKindAttr();
    auto tensorTypeName = [&]() {
      if (!ioKind)
        return kStructManagedTensorSlice;
      else if (ioKind.getValue() == MO::IOKind::Input) {
        BoolAttr isMutable = tensorType.getIsMutableAttr();
        if (isMutable && isMutable.getValue()) {
          return kComptimeMutableInputTensor;
        } else {
          return kComptimeInputTensor;
        }
      } else if (ioKind.getValue() == MO::IOKind::Output)
        return kComptimeOutputTensor;
      else {
        ASSERT_STREAM(false, << "Unsupported ioKind " << ioKind);
        return kStructManagedTensorSlice;
      }
    }();
    auto rank = tensorType.getRank();
    // Always use rank 1 for scalar tensors.
    if (rank == 0)
      rank = 1;
    auto dtype = mlirDTypeToMojoType(tensorType.getDtype()).text;
    return llvm::formatv("{}[dtype = {}, rank = {}, ...]", tensorTypeName,
                         dtype, rank)
        .str();
  }

  /// Given an array of dimensions, represent it as an IndexList Mojo value.
  /// Dynamic dimensions (kDynamic) are mapped to -1.
  static MojoValue getAsIndexListMojoValue(ArrayRef<int64_t> dims) {
    SmallVector<MojoValue> dimsAsMojoValues;
    // Map kDynamic to -1, keep others as-is.
    for (auto dim : dims) {
      if (dim == mlir::ShapedType::kDynamic)
        dimsAsMojoValues.push_back(-1);
      else
        dimsAsMojoValues.push_back(std::to_string(dim));
    }
    // Rank-0 tensors become rank-1.
    if (dimsAsMojoValues.empty())
      dimsAsMojoValues.push_back(1);
    MojoValue rank = dimsAsMojoValues.size();
    return MojoValue(kStructIndexList)[rank]({}, dimsAsMojoValues);
  }

  /// Given an array of dimensions, represent it as an IntTuple Mojo value.
  /// Dynamic dimensions (kDynamic) are mapped to -1 (UNKNOWN_VALUE).
  static MojoValue getAsIntTupleMojoValue(ArrayRef<int64_t> dims) {
    SmallVector<MojoValue> dimsAsMojoValues;
    // Map kDynamic to -1 (UNKNOWN_VALUE), keep others as-is.
    for (auto dim : dims) {
      if (dim == mlir::ShapedType::kDynamic)
        dimsAsMojoValues.push_back(-1);
      else
        dimsAsMojoValues.push_back(std::to_string(dim));
    }
    // Rank-0 tensors become rank-1.
    if (dimsAsMojoValues.empty())
      dimsAsMojoValues.push_back(1);
    return MojoValue(kStructIntTuple)({}, dimsAsMojoValues);
  }

  /// Given a tensor type, represent its static layout as a TileLayout Mojo
  /// value using `_IndexListToTileLayout`.
  static MojoValue
  getStaticTileLayoutFromIndexListsMojoValue(MO::TensorType type) {
    MojoValue shapeIL = getAsIndexListMojoValue(type.getIntDims());
    MojoValue stridesIL = getAsIndexListMojoValue(
        MO::getStridesForTensor(type, /*unknown=*/false));
    return MojoValue(kAliasIndexListToTileLayout)[{shapeIL, stridesIL}];
  }

  MojoValue getShapeAsIndexListMojoValue(MOSH::ShapeAttr shapeAttr,
                                         ValueRange dynamicDims = {});

  /// Emit the shape as a `Coord`, encoding static dims as compile-time `Idx[N]`
  /// components and dynamic dims as runtime `Int(<dim>)` components.
  MojoValue getShapeAsCoordMojoValue(MOSH::ShapeAttr shapeAttr,
                                     ValueRange dynamicDims = {});

  /// Helper to simply forward a list of mlir::Value into their MojoValue mapped
  /// equivalents.
  SmallVector<MojoValue> getMojoValuesFor(ValueRange operands) {
    return llvm::map_to_vector(
        operands, [&](Value operand) { return getMojoValueFor(operand); });
  }

  /// Split a fully-resolved function name (e.g. "max.kernels.AddOp.execute")
  /// into its struct prefix ("max.kernels.AddOp") and function suffix
  /// ("execute"), stripping any leading "__init__." prefix from the struct
  /// name.
  static std::pair<MojoValue, MojoValue>
  splitFunctionName(llvm::StringRef fullyResolvedName) {
    std::string fullName = fullyResolvedName.str();
    size_t lastDot = fullName.rfind('.');
    std::string structName = fullName.substr(0, lastDot);
    std::string fnName = fullName.substr(lastDot + 1);
    // Strip leading "__init__." package prefix from the struct name.
    constexpr llvm::StringLiteral kInitPrefix = "__init__.";
    if (llvm::StringRef(structName).starts_with(kInitPrefix))
      structName.erase(0, kInitPrefix.size());
    return {MojoValue(std::move(structName)), MojoValue(std::move(fnName))};
  }

  // Returns the corresponding SIMD type given its dtpype and its width
  static MojoValue getSIMDType(DType dtype, MojoValue width) {
    return MojoValue(kStructSIMD)[{mlirDTypeToMojoType(dtype), width}];
  }

  /// Looks at the named result-parameter mappings of a function adaptor.
  /// Returns the corresponding parameter name if found.
  static std::optional<StringRef>
  getResultParamNameFor(const KernelDeclFunctionAdaptor &fnAdaptor,
                        StringRef paramName);

  struct InferredParameters {
    llvm::SmallVector<MojoValue> structParameters;
    llvm::SmallVector<MojoValue> functionParameters;
  };

  /// Fill and return the list of inferred parameters for a struct kernel
  /// function (execute, shape, elementwise, or update_view). Uses the stored
  /// struct-level and fn-level ParamDeclAttrs from the KernelDeclAdaptor.
  /// Non-const because reading the function signature's pog list metadata
  /// is non-const-callable on `FnTypeGeneratorType`.
  static ErrorOr<InferredParameters> inferKernelParameters(
      KernelDeclAdaptor &adaptor, llvm::StringRef funcLabel,
      std::function<ErrorOr<MojoValue>(StringRef)> &&inferCallback);

  /// Fill and return the list of inferred parameters for a standalone
  /// primitive function (mogg.primitive.decl). No struct-level params.
  static ErrorOr<InferredParameters> inferKernelParameters(
      PrimitiveDeclOp primDecl,
      std::function<ErrorOr<MojoValue>(StringRef)> &&inferCallback);

  /// Fill and return the list of inferred parameters for a standalone shape
  /// function (mogg.shape_func.decl registered via @register_shape_function).
  /// No struct-level params.
  static ErrorOr<InferredParameters> inferKernelParameters(
      ShapeFunctionDeclOp shapeDecl,
      std::function<ErrorOr<MojoValue>(StringRef)> &&inferCallback);

  /// Base function to infer all parameter values of a kernel.
  ErrorOr<MojoValue>
  inferDefaultKernelParamValue(Operation *callOp, StringRef paramName,
                               DictionaryAttr customParamDict = nullptr);

  /// Debug print the spec of `tensorValue` if enabled.
  void addDebugPrintTensorSpecs(const MojoValue &tensorValue,
                                const MojoValue &indicesValue, bool isInput,
                                unsigned operandIndex);

  /// Debug print the parameters of a fused lambda if enabled.
  void addDebugPrintLambdaParam(MOGG::BindLambdaKind lambdaKind,
                                const MojoValue &widthParam,
                                const MojoValue &alignmentParam,
                                const MojoValue &indicesValue);
};
} // namespace M::MOGG

#endif // GENERICML_INTERFACES_EMITMOJOSTATE_H
