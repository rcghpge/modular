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
  SmallVector<int64_t> shape{}; // Defaults to empty shape

  if (auto shapeAttr = dyn_cast<MOSH::ShapeAttr>(tensor.getShapeAttr()))
    shape = shapeAttr.getIntDims();
  return MGP::TensorSpecType::get(shape, tensor.getElementType());
}

} // namespace M::MGP

#endif // GENERICML_GRAPHCOMPILER_CONVERSION_MGPSUPPORT_H
