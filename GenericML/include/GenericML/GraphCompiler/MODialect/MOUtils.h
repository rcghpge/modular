//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares utility function that don't depend on MO directly, but
// users of MO (including the dialect itself) may find useful. It is expected
// that some of these utilities may move to more accessible places (e.g.
// `Support`) over time.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MODIALECT_MOUTILS_H
#define GENERICML_GRAPHCOMPILER_MODIALECT_MOUTILS_H

#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/MDialect/MAttrs.h"
#include "Support/ML/DType.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "llvm/ADT/SmallVectorExtras.h"

//===----------------------------------------------------------------------===//
// MO Utilities
//===----------------------------------------------------------------------===//

namespace M::MO {

namespace detail {
template <typename inT, typename outT>
static SmallVector<outT>
denseResourceAttrToVec(DenseResourceElementsAttr attr) {
  ArrayRef<inT> data = attr.getRawHandle().getBlob()->getDataAs<inT>();
  return SmallVector<outT>(data.begin(), data.end());
}
} // namespace detail

/// If `attr` represents a blob of constants, return them in a vector,
/// else return None.
template <typename T>
static std::optional<SmallVector<T>> tryGetConstantBlob(mlir::ElementsAttr attr,
                                                        DType dType) {
  using namespace detail;
  if (auto floatElemsAttr = dyn_cast<FloatArrayElementsAttr>(attr)) {
    SmallVector<double> blob = llvm::map_to_vector(
        floatElemsAttr.getValues(),
        [](const llvm::APFloat &dim) { return dim.convertToDouble(); });
    return SmallVector<T>(blob.begin(), blob.end());
  }

  if (auto intElemsAttr = dyn_cast<IntArrayElementsAttr>(attr)) {
    SmallVector<int64_t> blob = llvm::map_to_vector(
        intElemsAttr.getValues(),
        [](const llvm::APInt &dim) { return dim.getSExtValue(); });
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

} // namespace M::MO

#endif // GENERICML_GRAPHCOMPILER_MODIALECT_MOUTILS_H
