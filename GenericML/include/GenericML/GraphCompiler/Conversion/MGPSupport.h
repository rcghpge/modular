//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_CONVERSION_MGPSUPPORT_H
#define GENERICML_GRAPHCOMPILER_CONVERSION_MGPSUPPORT_H

#include "GenericML/GraphCompiler/MODialect/MOTypes.h"
#include "GenericML/GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "GenericML/MGPDialect/MGPOps.h"

namespace M::MGP {

inline MGP::CpuTensorSpecType createCpuTensorSpec(MO::TensorType tensor) {
  if (tensor.hasRank()) {
    SmallVector<int64_t> shape{}; // Defaults to empty shape
    if (isa<MOSH::ShapeAttr>(tensor.getShapeAttr()))
      shape = tensor.getIntDims();
    return MGP::CpuTensorSpecType::get(shape, tensor.getElementType());
  } else {
    return MGP::CpuTensorSpecType::get(tensor.getElementType());
  }
}

inline MGP::CpuTensorSpecType createCpuTensorSpec(mlir::ShapedType tensor) {
  if (tensor.hasRank())
    return MGP::CpuTensorSpecType::get(tensor.getShape(),
                                       tensor.getElementType());
  else
    return MGP::CpuTensorSpecType::get(tensor.getElementType());
}

} // namespace M::MGP

#endif // GENERICML_GRAPHCOMPILER_CONVERSION_MGPSUPPORT_H
