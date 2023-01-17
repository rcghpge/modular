//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_CONVERSION_SUPPORT_H
#define GENERICML_CONVERSION_SUPPORT_H

#include "GenericML/MGPDialect/MGPOps.h"

namespace M::MGP {

template <typename T>
static MGP::CpuTensorSpecType createCpuTensorSpec(const T &tensor) {
  // TODO: Check for dynamic shape
  ArrayRef<int64_t> shape = tensor.getShape();
  // Convert scalars to rank 1 tensors
  // make a local value for one so it doesn't disappear as an rvalue
  // when creating the ArrayRef
  int64_t one = 1;
  if (tensor.hasRank() && tensor.getRank() == 0)
    shape = ArrayRef<int64_t>(one);
  // Output tensor type.
  return MGP::CpuTensorSpecType::get(shape, tensor.getElementType());
}

} // namespace M::MGP

#endif // GENERICML_CONVERSION_SUPPORT_H
