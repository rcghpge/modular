//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "modular/c/experimental/custom_ops.h"
#include "GenericML/API/c/detail/types.h"

#include "../detail/helpers.h"

#include <filesystem>

using namespace M;

void M_useKernelsFrom(M_CompileConfig *compileConfig, const char *path,
                      M_Status *status) {

  auto customOpsPath = std::filesystem::path(path);
  std::error_code ec;
  if (!std::filesystem::exists(customOpsPath, ec)) {
    detail::setError(status,
                     Error::getStaticString("invalid path to custom ops"));
    return;
  }

  auto &config = compileConfig->config;
  config.setCustomOpsPath(customOpsPath);
}
