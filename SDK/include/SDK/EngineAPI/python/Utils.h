//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
#ifndef SDK_ENGINEAPI_PYTHON_UTILS_H
#define SDK_ENGINEAPI_PYTHON_UTILS_H

#include <cstdint>
#include <optional>
#include <vector>

namespace M::core {

std::vector<int64_t>
convertToShape(const std::vector<std::optional<int64_t>> &optionalShape);
}

#endif // SDK_ENGINEAPI_PYTHON_UTILS_H
