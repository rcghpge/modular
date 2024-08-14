//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
#include "SDK/EngineAPI/python/Utils.h"
#include "max/c/common.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"

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

std::vector<int64_t> core::convertToShape(
    const std::vector<std::optional<std::variant<int64_t, std::string>>>
        &optionalShape) {
  std::vector<int64_t> shape;
  shape.reserve(optionalShape.size());
  for (const auto &optionalDim : optionalShape) {
    if (optionalDim.has_value() &&
        std::holds_alternative<int64_t>(optionalDim.value()))
      shape.push_back(std::get<int64_t>(optionalDim.value()));
    else
      shape.push_back(M_getDynamicDimensionValue());
  }
  return shape;
}

std::vector<std::optional<std::string>> core::convertToDimNames(
    const std::vector<std::optional<std::variant<int64_t, std::string>>>
        &optionalShape) {
  std::vector<std::optional<std::string>> dimNames;
  dimNames.reserve(optionalShape.size());
  for (const auto &optionalDim : optionalShape) {
    if (optionalDim.has_value() &&
        std::holds_alternative<std::string>(optionalDim.value()))
      dimNames.push_back(std::get<std::string>(optionalDim.value()));
    else
      dimNames.push_back(std::nullopt);
  }
  return dimNames;
}

void *core::getUnderlyingPtr(const py::object &obj) {
  auto *wrapper = reinterpret_cast<py::detail::instance *>(obj.ptr());
  wrapper->owned = false;
  void *&valueptr =
      py::detail::values_and_holders(wrapper).begin()->value_ptr();
  return valueptr;
}

void *core::getUnderlyingCapsulePtr(const py::object &obj, const char *name) {
  py::object capsule = mlirApiObjectToCapsule(obj);
  return PyCapsule_GetPointer(capsule.ptr(), name);
}
