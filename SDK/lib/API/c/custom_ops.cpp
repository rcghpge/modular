//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "max/c/custom_ops.h"
#include "SDK/API/c/detail/types.h"

#include "SDK/API/c/detail/helpers.h"

#include <filesystem>

using namespace M;

void M_useKernelsFrom(M_CompileConfig *compileConfig, const char *path,
                      M_Status *status) {
  std::string strPath = path;
  auto kernelLibraryPath = std::filesystem::path(strPath);
  std::error_code ec;
  if (!std::filesystem::exists(kernelLibraryPath, ec)) {
    M::Detail::setError(status,
                        Error("invalid path to kernel library: " + strPath));
    return;
  }

  auto &config = compileConfig->config;
  config.addKernelLibraryPath(std::move(kernelLibraryPath));
}
