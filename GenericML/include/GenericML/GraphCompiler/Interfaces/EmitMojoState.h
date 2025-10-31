//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
#ifndef GENERICML_INTERFACES_EMITMOJOSTATE_H
#define GENERICML_INTERFACES_EMITMOJOSTATE_H

#include "GenericML/GraphCompiler/MOGGDialect/Analysis/MojoLibraryAnalysis.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGTypes.h"
#include "GenericML/GraphCompiler/MOGGDialect/Support/KGENUtils.h"
#include "GenericML/GraphCompiler/MOGGDialect/Support/MojoBuilder.h"
#include "GenericML/MGPDialect/MGPTypes.h"

#include "KGEN/MOGGPreElab/MOGGPreElabDecorators.h"
#include "Support/AssertStream.h"

namespace M::MOGG {
/// TODO(GEX-2488): The following are Mojo symbols that are hardcoded for
/// convenience. The end goal is to remove them all and either register them in
/// a Mojo file via `@compiler_register` or determine them programatically
/// (packages). Ideally, name changes or file movements should not break
/// anything in C++ land and be contained in Mojo (ex: stdlib decide to rename
/// IndexList to something else, all the changes will be in Mojo only).

namespace HardcodedMojoSymbols {
constexpr StringLiteral struct_IO = "io_spec.IO";
constexpr StringLiteral struct_IndexList = "IndexList";
constexpr StringLiteral struct_StaticTensorSpec = "directives.StaticTensorSpec";
constexpr StringLiteral struct_UnsafePointer = "unsafe_pointer.UnsafePointer";
constexpr StringLiteral struct_String = "string.String";
constexpr StringLiteral struct_DeviceContextPtr = "asyncrt.DeviceContextPtr";
constexpr StringLiteral struct_ManagedTensorSlice =
    "managed_tensor_slice.ManagedTensorSlice";
constexpr StringLiteral struct_Int = "Int";
constexpr StringLiteral struct_UInt = "UInt";
constexpr StringLiteral struct_Bool = "Bool";
constexpr StringLiteral struct_DimList = "dimlist.DimList";
constexpr StringLiteral struct_Dim = "Dim";
constexpr StringLiteral struct_AddressSpace = "pointer.AddressSpace";
constexpr StringLiteral struct_DType = "DType";
constexpr StringLiteral struct_SIMD = "SIMD";
constexpr StringLiteral struct_StateContext = "MOGGPrimitives.StateContext";
constexpr StringLiteral struct_NDBuffer = "buffer.NDBuffer";
constexpr StringLiteral struct_StaticTuple = "StaticTuple";
constexpr StringLiteral struct_MoggAsyncPackHelper =
    "MOGGPrimitives.MoggAsyncPackHelper";

constexpr StringLiteral enum_AddressSpace_GENERIC = "GENERIC";

constexpr StringLiteral func_DimList_create_unknown = "create_unknown";
constexpr StringLiteral func_ManagedTensorslice_shape = "shape";
constexpr StringLiteral func_stack_allocation = "stack_allocation";
constexpr StringLiteral func_to_managed_tensor_slice =
    "MOGGPrimitives.to_managed_tensor_slice";
constexpr StringLiteral func_UnsafePointer_bitcast = "bitcast";
constexpr StringLiteral func_simd_load_from_managed_tensor_slice =
    "managed_tensor_slice.simd_load_from_managed_tensor_slice";
constexpr StringLiteral func_rebuild_static_tensor_specs_with_output_lambda =
    "managed_tensor_slice.rebuild_static_tensor_specs_with_output_lambda";
constexpr StringLiteral
    func_rebuild_static_tensor_specs_with_compute_output_lambda =
        "managed_tensor_slice.rebuild_static_tensor_specs_with_compute_output_"
        "lambda";
constexpr StringLiteral func_simd_store_into_managed_tensor_slice =
    "managed_tensor_slice.simd_store_into_managed_tensor_slice";
constexpr StringLiteral func_rebuild_static_tensor_specs_with_input_lambda =
    "managed_tensor_slice.rebuild_static_tensor_specs_with_input_lambda";
constexpr StringLiteral func_reshape_contiguous_buffer =
    "MOGGPrimitives.tmp_reshape_contiguous_buffer";
constexpr StringLiteral func_unpack_tensor_spec =
    "MOGGPrimitives.unpack_tensor_spec";
constexpr StringLiteral func_get_simd_width_for_dtypes =
    "MOGGPrimitives.get_simd_width_for_dtypes";
constexpr StringLiteral func_unpack_buffer_ref =
    "MOGGPrimitives.unpack_buffer_ref";
constexpr StringLiteral func_buffer_constant =
    "MOGGPrimitives.mgp_buffer_constant";
constexpr StringLiteral func_unpack_device_ctx =
    "MOGGPrimitives.unpack_device_ctx";

constexpr StringLiteral alias_OpaquePointer = "unsafe_pointer.OpaquePointer";
constexpr StringLiteral alias_InputTensor = "managed_tensor_slice.InputTensor";
constexpr StringLiteral alias_OutputTensor =
    "managed_tensor_slice.OutputTensor";
constexpr StringLiteral alias_FusedOutputTensor =
    "managed_tensor_slice._FusedOutputTensor";
constexpr StringLiteral alias_FusedComputeOutputTensor =
    "managed_tensor_slice._FusedComputeOutputTensor";
constexpr StringLiteral alias_FusedInputTensor =
    "managed_tensor_slice._FusedInputTensor";
constexpr StringLiteral alias_UInt8 = "UInt8";
constexpr StringLiteral alias_MutableAnyOrigin = "MutableAnyOrigin";

/// TODO(GEX-2488): Do this programmatically by checking the Modular config for
/// all included packages and packages provided by the user. This is better than
/// relying on an `__init__.mojo` file that imports things or harcoding a
/// package that could change/be moved around.
constexpr StringLiteral packagesToImport[] = {
    "MOGGPrimitives", "MOGGKernelAPI", "tensor", "compiler_internal", "utils",
    "runtime",        "memory",        "buffer", "collections"};

/// TODO(GEX-2488): Find another way to find that a type is an IOSpec without
/// relying on package / type name.
inline bool isMojoIOSpecTypeName(ArrayRef<FlatSymbolRefAttr> symbolRefs) {
  constexpr StringLiteral mojoPackageName = "io_spec";
  constexpr StringLiteral mojoTypeName = "IOSpec";
  return symbolRefs.size() == 2 &&
         symbolRefs[0].getValue() == mojoPackageName &&
         symbolRefs[1].getValue() == mojoTypeName;
}

}; // namespace HardcodedMojoSymbols

/// This struct holds all the necessary state to lower a MOGG operation into
/// Mojo text.
struct EmitMojoState {
  EmitMojoState(MojoLibraryAnalysis &analysis,
                bool enableOpTraceComments = false,
                bool enableComputeLambdas = false)
      : analysis(analysis), enableComputeLambdas(enableComputeLambdas),
        enableOpTraceComments(enableOpTraceComments) {}

  /// The Mojo builder used to generate teh text.
  MojoBuilder builder;
  /// The actual return values of the `mogg.generic.compiled.region`. This is
  /// used by ops that feed into `mogg.output` (ex: AsyncReadyOp or AsyncPack*).
  SmallVector<MojoValue> mojoReturnValues;
  MojoLibraryAnalysis &analysis;

  /// In some cases values must be kept alive using `_ = someVar` at the end of
  /// the function. This is because mojo Compiler isn't smart enough in some
  /// cases to detect the right lifetime. This is mostly around lambdas, and
  /// should be fixed by the new closures work.
  SmallVector<MojoValue> keepAliveValues;

  /// Keep track of the mojo arguments / parameters for the current function
  /// being lowered.
  FunctionBody currentFunctionBody;

  /// If true, generate compute lambdas when possible.
  bool enableComputeLambdas;

private:
  /// A mapping between mlir::Value to MojoValue (text).
  mlir::DenseMap<Value, MojoValue> mlirValueToMojoValue;

  /// Enables the emission of additional comments detailing IR ops that are
  /// currently being processed
  bool enableOpTraceComments;

public:
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
  static MojoValue mlirTypeToMojoType(Type mlirType) {
    if (isa<MGP::DeviceBufferType, MEF::ChainType>(mlirType)) {
      return HardcodedMojoSymbols::alias_OpaquePointer;
    } else if (isa<MGP::DeviceContextType, ContextType>(mlirType)) {
      return HardcodedMojoSymbols::struct_DeviceContextPtr;
    } else if (auto integerTy = dyn_cast<IntegerType>(mlirType)) {
      if (integerTy.getWidth() == 1)
        return HardcodedMojoSymbols::struct_Bool;
      return HardcodedMojoSymbols::struct_Int;
    } else if (auto tensorType = dyn_cast<MO::TensorType>(mlirType)) {
      return mlirTensorTypeToMojoType(tensorType);
    } else if (auto tensorSpecType = dyn_cast<MGP::TensorSpecType>(mlirType)) {
      return MojoValue(HardcodedMojoSymbols::struct_IndexList)[int64_t(
          tensorSpecType.getRank())];
    } else if (isa<MGP::ContextType>(mlirType)) {
      return HardcodedMojoSymbols::struct_StateContext;
    } else if (isa<IndexType>(mlirType)) {
      return HardcodedMojoSymbols::struct_Int;
    } else if (auto shapeResultType = dyn_cast<ShapeFuncResultType>(mlirType)) {
      MOGG::MojoValue rankParam = MOGG::MojoValue(shapeResultType.getRank());
      return MOGG::MojoValue(
          MOGG::HardcodedMojoSymbols::struct_IndexList)[rankParam];
    } else {
      ASSERT_STREAM(false, << "Unsupported mlir type " << mlirType);
      llvm_unreachable("Unsupported mlir type");
    }
  }

  static MojoValue mlirDTypeToMojoType(DType dtype) {
    KGEN::KGENDType kgenDtype(dtype);
    return MojoValue(HardcodedMojoSymbols::struct_DType)
        .get(kgenDtype.getAsString(/*libForm*/ true));
  }

  static MojoValue mlirTensorTypeToMojoType(MO::TensorType tensorType) {
    MO::IOKindAttr ioKind = tensorType.getIOKindAttr();
    auto tensorTypeName = [&]() {
      if (!ioKind)
        return HardcodedMojoSymbols::struct_ManagedTensorSlice;
      else if (ioKind.getValue() == MO::IOKind::Input)
        return HardcodedMojoSymbols::alias_InputTensor;
      else if (ioKind.getValue() == MO::IOKind::Output)
        return HardcodedMojoSymbols::alias_OutputTensor;
      else {
        ASSERT_STREAM(false, << "Unsupported ioKind " << ioKind);
        return HardcodedMojoSymbols::struct_ManagedTensorSlice;
      }
    }();
    auto rank = tensorType.getRank();
    // Always use rank 1 for scalar tensors.
    if (rank == 0)
      rank = 1;
    auto dtype = mlirDTypeToMojoType(tensorType.getDtype()).text;
    return llvm::formatv("{}[dtype = {}, rank = {}]", tensorTypeName, dtype,
                         rank)
        .str();
  }

  /// Given an array of dimensions, represent it in Mojo using `DimList`.
  static MojoValue getAsDimListMojoValue(ArrayRef<int64_t> dims) {
    if (llvm::all_of(dims, [](int64_t dim) {
          return dim == mlir::ShapedType::kDynamic;
        })) {

      return MojoValue(HardcodedMojoSymbols::struct_DimList)
          .get(HardcodedMojoSymbols::func_DimList_create_unknown)(
              {std::to_string(std::max<int64_t>(1, dims.size()))});
    }

    SmallVector<MojoValue> dimsAsMojoValues;
    for (auto dim : dims) {
      if (dim == mlir::ShapedType::kDynamic) {
        dimsAsMojoValues.push_back(
            MojoValue(HardcodedMojoSymbols::struct_Dim)());
      } else {
        dimsAsMojoValues.push_back(std::to_string(dim));
      }
    }

    return MojoValue(HardcodedMojoSymbols::struct_DimList)({},
                                                           dimsAsMojoValues);
  }

  static MojoValue getStaticShapeAsMojoValue(MO::TensorType type) {
    return getAsDimListMojoValue(type.getIntDims());
  }

  static MojoValue getStaticStridesAsMojoValue(MO::TensorType type) {
    return getAsDimListMojoValue(
        MOGG::getStridesForTensor(type, /*unknown=*/false));
  }

  MojoValue getShapeAsIndexListMojoValue(MO::TensorType tensorTy,
                                         ValueRange dynamicDims = {});

  /// Helper to simply forward a list of mlir::Value into their MojoValue mapped
  /// equivalents.
  SmallVector<MojoValue> getMojoValuesFor(ValueRange operands) {
    return llvm::map_to_vector(
        operands, [&](Value operand) { return getMojoValueFor(operand); });
  }

  /// Returns the original Mojo name of the registered function. The
  /// MOGGPreElab annotation will always tell you which .mojo file contains the
  /// generator. This is convenient for imports. Only packages need to be
  /// imported, not the .mojo file.
  static MojoValue getOriginalMojoNameFor(KGEN::LIT::FnOp op) {
    // TODO: update the signature to only take a LIT::FuncOp when the move to
    // LIT is complete.
    auto nameAttr =
        op->getAttrOfType<StringAttr>(KGEN::MOGGPreElab::mojoSymbolName);
    ASSERT_STREAM(nameAttr,
                  "Could not find original mojo symbol name attribute");
    return nameAttr.str();
  }

  // Returns the corresponding SIMD type given its dtpype and its width
  static MojoValue getSIMDType(DType dtype, MojoValue width) {
    return MojoValue(
        HardcodedMojoSymbols::struct_SIMD)[{mlirDTypeToMojoType(dtype), width}];
  }

  /// Looks at the named parameters of the output value of the function.
  /// Returns the corresponding parameter name of the function if found.
  static std::optional<MojoValue>
  getOriginalResultParamNameFor(KGEN::LIT::FnOp op, StringRef paramName) {
    auto resultParams = op->getAttrOfType<ArrayAttr>(
        KGEN::MOGGPreElab::kKernelResultParameterAttrName);
    if (!resultParams)
      return std::nullopt;

    for (auto resultParam : resultParams) {
      auto resultParamDict = cast<DictionaryAttr>(resultParam);
      if (auto funcParamName =
              resultParamDict.getAs<KGEN::ParamDeclRefAttr>(paramName))
        return funcParamName.getName().str();
    }

    return std::nullopt;
  }

  /// Fill and return the list of inferred parameters to call `function`.
  /// Call `inferCallback` for each parameter that needs to be inferred.
  static ErrorOr<llvm::SmallVector<MojoValue>> inferKernelParameters(
      KGEN::LIT::FnOp function,
      std::function<ErrorOr<MojoValue>(StringRef)> &&inferCallback);

  /// Base function to infer all parameter values of a kernel.
  ErrorOr<MojoValue>
  inferDefaultKernelParamValue(Operation *callOp, StringRef paramName,
                               DictionaryAttr customParamDict = nullptr);
};
} // namespace M::MOGG

#endif // GENERICML_INTERFACES_EMITMOJOSTATE_H
