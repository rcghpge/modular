//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "MLRT/AsyncRT/NIXL/utils/common/str_tools.h"

using namespace M::AsyncRT;

std::vector<std::string_view> NIXL::str_split_substr(std::string_view str,
                                                     std::string_view delim) {
  std::vector<std::string_view> parts;
  std::size_t start = 0;

  // Find each occurrence.
  for (std::size_t end;
       (end = str.find(delim, start)) != std::string_view::npos;
       start = end + delim.size()) {
    parts.emplace_back(str.substr(start, end - start));
  }

  // Tail.
  parts.emplace_back(str.substr(start));
  return parts;
}
