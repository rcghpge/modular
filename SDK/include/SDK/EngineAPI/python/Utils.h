//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
#ifndef SDK_ENGINEAPI_PYTHON_UTILS_H
#define SDK_ENGINEAPI_PYTHON_UTILS_H

#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace M::core {

std::vector<int64_t>
convertToShape(const std::vector<std::optional<int64_t>> &optionalShape);
std::vector<int64_t> convertToShape(
    const std::vector<std::optional<std::variant<int64_t, std::string>>>
        &optionalShape);
std::vector<std::optional<std::string>> convertToDimNames(
    const std::vector<std::optional<std::variant<int64_t, std::string>>>
        &optionalShape);

} // namespace M::core

#endif // SDK_ENGINEAPI_PYTHON_UTILS_H
