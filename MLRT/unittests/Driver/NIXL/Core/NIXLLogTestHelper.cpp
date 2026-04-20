//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "MLRT/Driver/NIXL/utils/common/nixl_log.h"

#include <cstdio>
#include <cstring>

using namespace M::MLRT;

int main(int argc, char **argv) {
  if (argc != 2) {
    return 2;
  }

  if (std::strcmp(argv[1], "default-warn") == 0) {
    NIXL_INFO << "info-hidden";
    NIXL_WARN << "warn-visible";
    return 0;
  }

  if (std::strcmp(argv[1], "error-only") == 0) {
    NIXL_WARN << "warn-hidden";
    NIXL_ERROR << "error-visible";
    return 0;
  }

  if (std::strcmp(argv[1], "debug-enabled") == 0) {
    NIXL_DEBUG << "debug-visible";
    NIXL_TRACE << "trace-hidden";
    return 0;
  }

  if (std::strcmp(argv[1], "short-circuit") == 0) {
    int counter = 0;
    auto sideEffect = [&]() {
      ++counter;
      return "debug-side-effect";
    };
    NIXL_DEBUG << sideEffect();
    std::fprintf(stdout, "counter=%d\n", counter);
    return 0;
  }

  if (std::strcmp(argv[1], "fatal-visible") == 0) {
    NIXL_FATAL << "fatal-visible";
    return 0;
  }

  return 3;
}
