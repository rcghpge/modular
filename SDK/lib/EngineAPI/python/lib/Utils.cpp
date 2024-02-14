//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
#include "SDK/EngineAPI/python/Utils.h"
#include "max/c/common.h"

using namespace M;

std::vector<int64_t>
core::convertToShape(const std::vector<std::optional<int64_t>> &optionalShape) {
  std::vector<int64_t> shape;
  shape.reserve(optionalShape.size());
  for (std::optional<int64_t> optionalDim : optionalShape) {
    if (optionalDim.has_value())
      shape.push_back(optionalDim.value());
    else
      shape.push_back(M_getDynamicDimensionValue());
  }
  return shape;
}
