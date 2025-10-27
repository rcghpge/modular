//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/MODialect/MOUtils.h"
#include "GenericML/GraphCompiler/MODialect/MOTypes.h"

using namespace M;
using namespace MO;

template <>
std::optional<SmallVector<APInt>>
M::MO::tryGetConstantBlob(mlir::ElementsAttr attr, DType dType) {
  using namespace M::MO::Detail;
  if (auto intElemsAttr = dyn_cast<IntArrayElementsAttr>(attr)) {
    SmallVector<APInt> blob = llvm::map_to_vector(
        intElemsAttr.getValues(), [](const llvm::APInt &elem) { return elem; });
    return blob;
  }

  if (auto denseAttr = dyn_cast<DenseResourceElementsAttr>(attr)) {
    ArrayRef<APInt> data =
        denseAttr.getRawHandle().getBlob()->getDataAs<APInt>();
    return SmallVector<APInt>(data.begin(), data.end());
  }

  return std::nullopt;
}

template <>
std::optional<SmallVector<APFloat>>
M::MO::tryGetConstantBlob(mlir::ElementsAttr attr, DType dType) {
  using namespace M::MO::Detail;
  if (auto floatElemsAttr = dyn_cast<FloatArrayElementsAttr>(attr)) {
    SmallVector<APFloat> blob =
        llvm::map_to_vector(floatElemsAttr.getValues(),
                            [](const llvm::APFloat &elem) { return elem; });
    return blob;
  }

  if (auto denseAttr = dyn_cast<DenseResourceElementsAttr>(attr)) {
    ArrayRef<APFloat> data =
        denseAttr.getRawHandle().getBlob()->getDataAs<APFloat>();
    return SmallVector<APFloat>(data.begin(), data.end());
  }

  return std::nullopt;
}

bool M::MO::isMutableOperand(Value val) {
  return isa<MO::BufferType, MO::OpaqueType>(val.getType());
}
