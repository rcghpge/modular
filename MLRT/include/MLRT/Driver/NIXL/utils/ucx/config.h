//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef MLRT_DRIVER_NIXL_UTILS_UCX_CONFIG_H
#define MLRT_DRIVER_NIXL_UTILS_UCX_CONFIG_H

#include "Support/ErrorOr.h"
#include <memory>
#include <string_view>

extern "C" {
#include <ucp/api/ucp.h>
}

namespace M::AsyncRT::NIXL::UCX {

class Config {
public:
  // Factory method to create Config with proper error handling.
  [[nodiscard]] static ErrorOr<Config> create();

  // Move constructor and assignment.
  Config(Config &&) = default;
  Config &operator=(Config &&) = default;

  // Delete copy constructor and assignment.
  Config(const Config &) = delete;
  Config &operator=(const Config &) = delete;

  ~Config() = default;

  [[nodiscard]] ucp_config_t *getUcpConfig() const { return config_.get(); }

  // Modify the config if it is not already set via environment variable
  void modify(std::string_view key, std::string_view value) const;

  // Modify the config always
  void modifyAlways(std::string_view key, std::string_view value) const;

private:
  // Private constructor - use create() factory method
  explicit Config(ucp_config_t *config)
      : config_(config, &ucp_config_release) {}

  [[nodiscard]] static ErrorOr<ucp_config_t *> readUcpConfig();

  std::unique_ptr<ucp_config_t, void (*)(ucp_config_t *)> config_;
};

} // namespace M::AsyncRT::NIXL::UCX

#endif // MLRT_DRIVER_NIXL_UTILS_UCX_CONFIG_H
