//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "MLRT/AsyncRT/NIXL/utils/ucx/config.h"
#include "MLRT/AsyncRT/NIXL/utils/common/nixl_log.h"
#include "llvm/Support/FormatVariadic.h"

#include <cstdlib>

using namespace M;
using namespace M::AsyncRT::NIXL;

ErrorOr<UCX::Config> UCX::Config::create() {
  auto configOrError = readUcpConfig();
  if (configOrError.isError()) {
    return configOrError.takeError();
  }
  return Config(configOrError.takeValue());
}

void UCX::Config::modify(std::string_view key, std::string_view value) const {
  std::string envVarName = llvm::formatv("UCX_{0}", key).str();
  const char *envVal = std::getenv(envVarName.c_str());
  if (envVal) {
    NIXL_DEBUG << "UCX env var has already been set: " << key << "=" << envVal;
  } else {
    modifyAlways(key, value);
  }
}

void UCX::Config::modifyAlways(std::string_view key,
                               std::string_view value) const {
  const auto status =
      ucp_config_modify(config_.get(), key.data(), value.data());
  if (status != UCS_OK) {
    NIXL_WARN << "Failed to modify UCX config: " << key << "=" << value << ": "
              << ucs_status_string(status);
  } else {
    NIXL_DEBUG << "Modified UCX config: " << key << "=" << value;
  }
}

ErrorOr<ucp_config_t *> UCX::Config::readUcpConfig() {
  ucp_config_t *config = nullptr;
  const auto status = ucp_config_read(nullptr, nullptr, &config);
  if (status != UCS_OK) {
    const auto errStr = std::string("Failed to create UCX config: ") +
                        ucs_status_string(status);
    NIXL_ERROR << errStr;
    return Error(errStr);
  }

  return config;
}
