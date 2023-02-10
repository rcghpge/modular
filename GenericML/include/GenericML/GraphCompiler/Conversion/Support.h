//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_CONVERSION_SUPPORT_H
#define GENERICML_GRAPHCOMPILER_CONVERSION_SUPPORT_H

#include "GenericML/MGPDialect/MGPOps.h"

namespace M::MGP {

template <typename T>
static MGP::CpuTensorSpecType createCpuTensorSpec(const T &tensor) {
  // TODO: Check for dynamic shape
  ArrayRef<int64_t> shape = tensor.getShape();
  return MGP::CpuTensorSpecType::get(shape, tensor.getElementType());
}

} // namespace M::MGP

#endif // GENERICML_GRAPHCOMPILER_CONVERSION_SUPPORT_H
