//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "SDK/EngineAPI/python/TorchInputSpec.h"
#include "GenericML/API/max/c/types.h"
#include "GenericML/API/max/cpp/aliases.h"
#include "SDK/EngineAPI/python/Utils.h"
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

using namespace M;
using namespace core;

core::TorchInputSpec::TorchInputSpec(
    const core::TorchInputSpec::InputShapeTy &shapeOr, M::DType dtype)
    : core::TorchInputSpec(shapeOr, dtype, "") {}

core::TorchInputSpec::TorchInputSpec(
    const core::TorchInputSpec::InputShapeTy &shapeOr, M::DType dtype,
    std::string device)
    : specPtr(nullptr), specShape(shapeOr), specDtype(dtype),
      specDevice(device) {
  StatusPtr status{M_newStatus()};
  if (!status)
    throw std::invalid_argument("Unable to create status");
  if (specShape) {
    std::vector<int64_t> convertedShape = convertToShape(*specShape);
    std::vector<std::optional<std::string>> dimNames =
        convertToDimNames(*specShape);
    std::vector<const char *> dimNamePtrs;
    dimNamePtrs.reserve(dimNames.size());
    for (const auto &optionalDimName : dimNames)
      dimNamePtrs.push_back(optionalDimName.has_value()
                                ? optionalDimName.value().c_str()
                                : nullptr);
    specPtr.reset(M_newTorchInputSpec(
        convertedShape.data(), dimNamePtrs.data(), convertedShape.size(),
        static_cast<M_Dtype>(dtype.getValue()), device.c_str(), status.get()));
  } else {
    specPtr.reset(M_newTorchInputSpec(
        /*shape=*/nullptr, /*dimNames=*/nullptr, M_getDynamicRankValue(),
        static_cast<M_Dtype>(dtype.getValue()), device.c_str(), status.get()));
  }
  if (M_isError(status.get()))
    throw std::invalid_argument(M_getError(status.get()));
}
