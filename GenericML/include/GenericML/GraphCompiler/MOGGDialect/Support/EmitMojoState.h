//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
#ifndef GENERICML_GRAPH_COMPILER_MOGGSUPPORT_EMITMOJOSTATE_H
#define GENERICML_GRAPH_COMPILER_MOGGSUPPORT_EMITMOJOSTATE_H

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
constexpr StringLiteral struct_DeviceContextPtr = "asyncrt.DeviceContextPtr";
constexpr StringLiteral struct_ManagedTensorSlice =
    "managed_tensor_slice.ManagedTensorSlice";
constexpr StringLiteral struct_Int = "Int";
constexpr StringLiteral struct_DimList = "dimlist.DimList";
constexpr StringLiteral struct_Dim = "Dim";
constexpr StringLiteral struct_AddressSpace = "pointer.AddressSpace";
constexpr StringLiteral struct_DType = "DType";

constexpr StringLiteral enum_AddressSpace_GENERIC = "GENERIC";

constexpr StringLiteral func_DimList_create_unknown = "create_unknown";
constexpr StringLiteral func_ManagedTensorslice_shape = "shape";

constexpr StringLiteral alias_OpaquePointer = "unsafe_pointer.OpaquePointer";
constexpr StringLiteral alias_InputTensor = "managed_tensor_slice.InputTensor";
constexpr StringLiteral alias_OutputTensor =
    "managed_tensor_slice.OutputTensor";

/// TODO(GEX-2488): Do this programmatically by checking the Modular config for
/// all included packages and packages provided by the user. This is better than
/// relying on an `__init__.mojo` file that imports things or harcoding a
/// package that could change/be moved around.
constexpr StringLiteral packagesToImport[] = {
    "MOGGPrimitives", "MOGGKernelAPI", "tensor_internal", "compiler_internal",
    "utils",          "runtime",       "memory",          "buffer"};
}; // namespace HardcodedMojoSymbols

/// This struct holds all the necessary state to lower a MOGG operation into
/// Mojo text.
struct EmitMojoState {
  EmitMojoState(MojoLibraryAnalysis &analysis) : analysis(analysis) {}

  /// The Mojo builder used to generate teh text.
  MojoBuilder builder;
  /// A mapping between mlir::Value to MojoValue (text).
  mlir::DenseMap<Value, MojoValue> mlirValueToMojoValue;
  /// The return actual values of the `mogg.generic.compiled.region`. This is
  /// only used by `AsyncReadyOp::emitMojo`.
  SmallVector<MojoValue> mojoReturnValues;
  MojoLibraryAnalysis &analysis;

  /// Convert an mlir::Type to a MojoValue representing the type in Mojo text.
  static MojoValue mlirTypeToMojoType(Type mlirType) {
    if (isa<MGP::DeviceBufferType, MEF::ChainType>(mlirType)) {
      return HardcodedMojoSymbols::alias_OpaquePointer;
    } else if (isa<MGP::DeviceContextType, ContextType>(mlirType)) {
      return HardcodedMojoSymbols::struct_DeviceContextPtr;
    } else if (isa<IntegerType>(mlirType)) {
      return HardcodedMojoSymbols::struct_Int;
    } else if (auto tensorType = dyn_cast<MO::TensorType>(mlirType)) {
      return mlirTensorTypeToMojoType(tensorType);
    } else {
      ASSERT_STREAM(false, << "Unsupported mlir type " << mlirType);
    }
  }

  static MojoValue mlirDTypeToMojoType(DType dtype) {
    KGEN::KGENDType kgenDtype(dtype);
    return MojoValue(HardcodedMojoSymbols::struct_DType)
        .get(kgenDtype.getAsString(/*libForm*/ true));
  }

  static MojoValue mlirTensorTypeToMojoType(MO::TensorType tensorType) {
    auto rank = tensorType.getRank();
    // Always use rank 1 for scalar tensors.
    if (rank == 0)
      rank = 1;
    auto dtype = mlirDTypeToMojoType(tensorType.getDtype()).text;
    return llvm::formatv("{}[dtype = {}, rank = {}]",
                         HardcodedMojoSymbols::struct_ManagedTensorSlice, dtype,
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

  /// Helper to simply forward a list of mlir::Value into their MojoValue mapped
  /// equivalents.
  SmallVector<MojoValue> getMojoValuesFor(ValueRange operands) {
    return llvm::map_to_vector(
        operands, [&](Value operand) { return mlirValueToMojoValue[operand]; });
  }

  /// Returns the original Mojo name of the registered generator. The
  /// MOGGPreElab annotation will always tell you which .mojo file contains the
  /// generator. This is convenient for imports. Only packages need to be
  /// imported, not the .mojo file.
  static MojoValue getOriginalMojoNameFor(KGEN::GeneratorOp op) {
    auto nameAttr =
        op->getAttrOfType<StringAttr>(KGEN::MOGGPreElab::mojoSymbolName);
    ASSERT_STREAM(nameAttr,
                  "Could not find original mojo symbol name attribute");
    return nameAttr.str();
  }
};
} // namespace M::MOGG

#endif
