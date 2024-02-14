//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "SDK/EngineAPI/python/TorchInputSpec.h"
#include "GenericML/API/max/c/tensor.h"
#include "GenericML/API/max/c/types.h"
#include "GenericML/API/max/cpp/aliases.h"
#include "SDK/EngineAPI/python/Utils.h"
#include <optional>
#include <string>
#include <vector>

using namespace M;
using namespace core;

core::TorchInputSpec::TorchInputSpec(
    const core::TorchInputSpec::InputShapeTy &shapeOr, M_Dtype dtype)
    : specPtr({nullptr, &M_freeTorchInputSpec}), specShape(shapeOr),
      specDtype(dtype) {
  if (specShape) {
    const std::vector<std::optional<int64_t>> &shape = *specShape;
    std::vector<int64_t> convertedShape = convertToShape(shape);
    specPtr.reset(M_newTorchInputSpec(convertedShape.data(),
                                      convertedShape.size(), dtype));
  } else {
    specPtr.reset(M_newTorchInputSpec(nullptr, M_getDynamicRankValue(), dtype));
  }
}
