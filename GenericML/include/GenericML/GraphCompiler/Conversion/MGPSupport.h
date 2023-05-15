//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_CONVERSION_MGPSUPPORT_H
#define GENERICML_GRAPHCOMPILER_CONVERSION_MGPSUPPORT_H

#include "GenericML/GraphCompiler/MODialect/MOTypes.h"
#include "GenericML/MGPDialect/MGPOps.h"

namespace M::MGP {

inline MGP::CpuTensorSpecType createCpuTensorSpec(MO::TensorType tensor) {
  // TODO(#12114): Enable dynamic rank
  assert(tensor.hasRank() && "dynamic rank not supported");
  SmallVector<int64_t> shape{}; // Defaults to unknown shape.
  if (isa<MOSH::ShapeAttr>(tensor.getShapeAttr()))
    shape = tensor.getIntDims();
  return MGP::CpuTensorSpecType::get(shape, tensor.getElementType());
}

inline MGP::CpuTensorSpecType createCpuTensorSpec(mlir::ShapedType tensor) {
  // TODO(#12114): Enable dynamic rank
  assert(tensor.hasRank() && "dynamic rank not supported");
  ArrayRef<int64_t> shape = tensor.getShape();
  return MGP::CpuTensorSpecType::get(shape, tensor.getElementType());
}

} // namespace M::MGP

#endif // GENERICML_GRAPHCOMPILER_CONVERSION_MGPSUPPORT_H
