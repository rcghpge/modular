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

#include <type_traits>

//===----------------------------------------------------------------------===//
// MO Utilities
//===----------------------------------------------------------------------===//

namespace M::MO {

namespace detail {
template <typename inT, typename outT>
static SmallVector<outT>
denseResourceAttrToVecDirect(DenseResourceElementsAttr attr) {
  ArrayRef<inT> data = attr.getRawHandle().getBlob()->getDataAs<inT>();
  return SmallVector<outT>(data.begin(), data.end());
}

/**
 * @brief Reinterpret the bits of attr as the given float semantic.
 *
 * @param [in] uintStorageType Storage type whose bits will be reinterpretted.
 * @param [in] attr The attribute to process.
 * @param [in] semantic The APFloat specification to interpret the bits as.
 * @param [out] outputType The output type to give the final output.
 *
 * @return The return value of the method.
 */
template <typename uintStorageType, typename outputType>
static SmallVector<outputType>
denseResourceAttrToVecSpecialFloat(DenseResourceElementsAttr attr,
                                   const llvm::fltSemantics &semantic) {
  // For FP types with no builtin C/C++ type use APFloat semantic
  constexpr int numBitsIn = sizeof(uintStorageType) * CHAR_BIT;
  static_assert(std::is_integral_v<uintStorageType>,
                "Please use uintXX_t to handle special floats.");
  static_assert(std::is_unsigned_v<uintStorageType>,
                "Please use uintXX_t to handle special floats.");
  static_assert(numBitsIn <= 64, "Support only up to 64 bit types");

  ArrayRef<uintStorageType> data =
      attr.getRawHandle().getBlob()->getDataAs<uintStorageType>();
  SmallVector<outputType> out =
      llvm::map_to_vector(data, [&](const uintStorageType &dim) {
        // Note this will fail an assert if the semantic float cannot always
        // convert to a double losslessly
        double result =
            APFloat(semantic, APInt(numBitsIn, dim)).convertToDouble();
        return static_cast<outputType>(result);
      });
  return out;
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
        .template when<M::DType::si32>([&]() {
          return denseResourceAttrToVecDirect<int32_t, T>(axisValueAttr);
        })
        .template when<M::DType::si64>([&]() {
          return denseResourceAttrToVecDirect<int64_t, T>(axisValueAttr);
        })
        .template when<M::DType::bf16>([&]() {
          return denseResourceAttrToVecSpecialFloat<uint16_t, T>(
              axisValueAttr, APFloat::BFloat());
        })
        .template when<M::DType::f16>([&]() {
          return denseResourceAttrToVecSpecialFloat<uint16_t, T>(
              axisValueAttr, APFloat::IEEEhalf());
        })
        .template when<M::DType::f32>([&]() {
          return denseResourceAttrToVecDirect<float, T>(axisValueAttr);
        })
        .template when<M::DType::f64>([&]() {
          return denseResourceAttrToVecDirect<double, T>(axisValueAttr);
        })
        .otherwise([&]() { return std::nullopt; });
  }

  return std::nullopt;
}

} // namespace M::MO

#endif // GENERICML_GRAPHCOMPILER_MODIALECT_MOUTILS_H
