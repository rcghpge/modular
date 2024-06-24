//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
#ifndef SDK_ENGINEAPI_PYTHON_UTILS_H
#define SDK_ENGINEAPI_PYTHON_UTILS_H

#include "pybind11/pybind11.h"
#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace py = pybind11;

namespace M::core {

std::vector<int64_t>
convertToShape(const std::vector<std::optional<int64_t>> &optionalShape);
std::vector<int64_t> convertToShape(
    const std::vector<std::optional<std::variant<int64_t, std::string>>>
        &optionalShape);
std::vector<std::optional<std::string>> convertToDimNames(
    const std::vector<std::optional<std::variant<int64_t, std::string>>>
        &optionalShape);

/// Extract the underlying data pointer of a pybind11 class.
void *getUnderlyingPtr(const py::object &obj);

/// Wrapping cast where the `obj` will be unpacked, the pybind11 class pointer
/// extracted, cast to the underlying type (which doesn't have to be defined or
/// declared in the current context) and stored in the PIMPL abstraction of
/// T. T needs to declare a constructor that takes a `void *` and is aware of
/// the underlying type. The method has no type safety.
template <typename T>
T unsafeWrapCast(const py::object &obj) {
  return T(getUnderlyingPtr(obj));
}

/// Wrap cast with custom conversion function.
template <typename T>
T unsafeWrapCast(const py::object &obj, std::function<T(void *)> fn) {
  return fn(getUnderlyingPtr(obj));
}

} // namespace M::core

#endif // SDK_ENGINEAPI_PYTHON_UTILS_H
