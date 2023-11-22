//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "modular/c/custom_ops.h"
#include "GenericML/API/c/detail/types.h"

#include "detail/helpers.h"

#include <filesystem>

using namespace M;

void M_useKernelsFrom(M_CompileConfig *compileConfig, const char *path,
                      M_Status *status) {

  auto kernelLibraryPath = std::filesystem::path(std::string(path));
  std::error_code ec;
  if (!std::filesystem::exists(kernelLibraryPath, ec)) {
    detail::setError(status,
                     Error::getStaticString("invalid path to kernel library"));
    return;
  }

  auto &config = compileConfig->config;
  config.setKernelLibraryPath(std::move(kernelLibraryPath));
}
