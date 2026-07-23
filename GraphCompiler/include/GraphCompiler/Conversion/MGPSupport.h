//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_CONVERSION_MGPSUPPORT_H
#define GENERICML_GRAPHCOMPILER_CONVERSION_MGPSUPPORT_H

#include "GraphCompiler/MGPDialect/Dialect/MGPOps.h"
#include "GraphCompiler/MODialect/MOTypes.h"
#include "GraphCompiler/MOSHDialect/MOSHAttrs.h"

namespace M::MGP {

inline MGP::TensorSpecType createTensorSpecType(MO::TensorType tensor) {
  if (tensor.hasRank()) {
    SmallVector<int64_t> shape{}; // Defaults to empty shape
    if (isa<MOSH::ShapeAttr>(tensor.getShapeAttr()))
      shape = tensor.getIntDims();
    return MGP::TensorSpecType::get(shape, tensor.getElementType());
  } else {
    return MGP::TensorSpecType::get(tensor.getElementType());
  }
}

inline MGP::TensorSpecType createTensorSpecType(mlir::ShapedType tensor) {
  if (tensor.hasRank())
    return MGP::TensorSpecType::get(tensor.getShape(), tensor.getElementType());
  else
    return MGP::TensorSpecType::get(tensor.getElementType());
}

} // namespace M::MGP

#endif // GENERICML_GRAPHCOMPILER_CONVERSION_MGPSUPPORT_H
