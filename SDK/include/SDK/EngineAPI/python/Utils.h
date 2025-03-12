//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
#ifndef SDK_ENGINEAPI_PYTHON_UTILS_H
#define SDK_ENGINEAPI_PYTHON_UTILS_H

#include "nanobind/nanobind.h"
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace nb = nanobind;

namespace M::core {

std::vector<int64_t>
convertToShape(const std::vector<std::optional<int64_t>> &optionalShape);
std::vector<int64_t> convertToShape(
    const std::vector<std::optional<std::variant<int64_t, std::string>>>
        &optionalShape);
std::vector<std::optional<std::string>> convertToDimNames(
    const std::vector<std::optional<std::variant<int64_t, std::string>>>
        &optionalShape);

void *getUnderlyingCapsulePtr(const nb::handle &obj, const char *name);

} // namespace M::core

#endif // SDK_ENGINEAPI_PYTHON_UTILS_H
