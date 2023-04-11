//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares the operation classes for the MO dialect.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MODIALECT_MOOPS_H
#define GENERICML_GRAPHCOMPILER_MODIALECT_MOOPS_H

#include "GenericML/GraphCompiler/MODialect/MODialect.h"
#include "GenericML/GraphCompiler/MODialect/MOInterfaces.h"
#include "GenericML/GraphCompiler/MODialect/MOTypes.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENTypes.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/STLExtras.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir {
class ModuleOp;
} // namespace mlir

// Needed because the ODS definition of some KGEN ops don't have the namespace.
using M::KGEN::ParamDeclArrayAttr;
using M::KGEN::ParamDeclAttr;
using M::KGEN::ParameterExprArrayAttr;
using M::KGEN::SignatureType;

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "GenericML/GraphCompiler/MODialect/MO.h.inc"

//===----------------------------------------------------------------------===//
// MO Utilities
//===----------------------------------------------------------------------===//

namespace M::MO {

// TODO: Perhaps these should be in a MOUtils.h instead of MOOps.h?
template <typename inT, typename outT>
static SmallVector<outT>
denseResourceAttrToVec(DenseResourceElementsAttr attr) {
  ArrayRef<inT> data = attr.getRawHandle().getBlob()->getDataAs<inT>();
  return SmallVector<outT>(data.begin(), data.end());
}

/// If `attr` represents a blob of constants, return them in a vector,
/// else return None.
template <typename T>
static std::optional<SmallVector<T>> tryGetConstantBlob(mlir::ElementsAttr attr,
                                                        DType dType) {
  if (auto floatElemsAttr = dyn_cast<FloatArrayElementsAttr>(attr)) {
    SmallVector<double> blob =
        map_to_vector(floatElemsAttr.getValues(), [](const llvm::APFloat &dim) {
          return dim.convertToDouble();
        });
    return SmallVector<T>(blob.begin(), blob.end());
  }

  if (auto intElemsAttr = dyn_cast<IntArrayElementsAttr>(attr)) {
    SmallVector<int64_t> blob =
        map_to_vector(intElemsAttr.getValues(), [](const llvm::APInt &dim) {
          return dim.getSExtValue();
        });
    return SmallVector<T>(blob.begin(), blob.end());
  }

  if (auto axisValueAttr = dyn_cast<DenseResourceElementsAttr>(attr)) {
    return dType.dispatch<std::optional<SmallVector<T>>>()
        .template when<M::DType::si32>(
            [&]() { return denseResourceAttrToVec<int32_t, T>(axisValueAttr); })
        .template when<M::DType::si64>(
            [&]() { return denseResourceAttrToVec<int64_t, T>(axisValueAttr); })
        .template when<M::DType::f32>(
            [&]() { return denseResourceAttrToVec<float, T>(axisValueAttr); })
        .template when<M::DType::f64>(
            [&]() { return denseResourceAttrToVec<double, T>(axisValueAttr); })
        .otherwise([&]() { return std::nullopt; });
  }

  return std::nullopt;
}

/// If `val` comes from a constant tensor (defined by a ConstantOp or
/// RebindOp that takes in a ConstantOp), return the defining op
MO::ConstantOp tryGetConstantOpOfRank(TensorValue val, size_t rank);

/// If `val` is defined by a ConstantOp or a RebindOp that takes in a
/// ConstantOp with given rank, convert it to an vector of specificed type,
/// else return None.
template <typename T>
static std::optional<SmallVector<T>> tryGetConstantOfRank(TensorValue val,
                                                          size_t rank) {
  if (ConstantOp constantOp = tryGetConstantOpOfRank(val, rank)) {
    return tryGetConstantBlob<T>(constantOp.getValueAttr(),
                                 constantOp.getType().getDtype());
  }
  return std::nullopt;
}

/// If `val` comes from a constant integer scalar tensor (defined by a
/// ConstantOp or RebindOp that takes in a ConstantOp), convert it to a raw
/// integer, else return None.
std::optional<int64_t> tryGetConstantInt(TensorValue val);

/// If `val` comes from a constant integer 1D tensor (defined by a ConstantOp or
/// RebindOp that takes in a ConstantOp), convert it to an integer vector, else
/// return None.
std::optional<SmallVector<int64_t>> tryGetConstantIntVector(TensorValue val);

/// Returns a parameter constant attribute through a variant of ToTensor ops if
/// possible. Otherwise returns a null attribute.
template <typename ToTensorOpTy>
static TypedAttr getParamConstThrough(TensorValue inputVal) {
  if (auto toTensorOp = inputVal.template getDefiningOp<ToTensorOpTy>()) {
    if (auto constOp = toTensorOp.getInput()
                           .template getDefiningOp<KGEN::ParamConstantOp>()) {
      return constOp.getValue();
    }
  }
  return {};
}
} // namespace M::MO

#endif // GENERICML_GRAPHCOMPILER_MODIALECT_MOOPS_H
