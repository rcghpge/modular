//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MODialect/MOUtils.h"
#include "GraphCompiler/MODialect/MOTypes.h"

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

LogicalResult M::MO::checkReshapeShapesCompatible(
    MOSH::ShapeAttr inShape, MOSH::ShapeAttr outShape,
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError) {
  std::optional<int64_t> inNumEles =
      MOSH::ShapeAttr::getStaticDim(inShape.getSymbolicNumElements());
  std::optional<int64_t> outNumEles =
      MOSH::ShapeAttr::getStaticDim(outShape.getSymbolicNumElements());
  if (!inNumEles.has_value() || !outNumEles.has_value())
    return success();

  // Allow one -1 dimension for now, which is a fancy parameter.
  if (*inNumEles < 0 || *outNumEles < 0)
    return success();

  if (*inNumEles != *outNumEles)
    return emitError() << "input and output elements do not match: "
                       << *inNumEles << " vs " << *outNumEles;
  return success();
}
