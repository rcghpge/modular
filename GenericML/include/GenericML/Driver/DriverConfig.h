//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_DRIVER_DRIVERCONFIG_H
#define GENERICML_DRIVER_DRIVERCONFIG_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"

#include <filesystem>

namespace M {
class DriverConfig {
public:
  enum OptionKind { kTF, kTorch, kMOF, kONNX, kCustomOps };

  DriverConfig(OptionKind kind, llvm::StringLiteral name)
      : Kind(kind), configName(name) {}

  OptionKind getKind() const { return Kind; }

  llvm::StringLiteral getConfigName() const { return configName; }

  virtual ~DriverConfig() = default;

private:
  const OptionKind Kind;
  const llvm::StringLiteral configName;
};

class CustomOpsDriverConfig : public DriverConfig {

public:
  CustomOpsDriverConfig(std::filesystem::path path)
      : DriverConfig(OptionKind::kCustomOps, "custom-ops-path"),
        customOpsPath(std::move(path)) {}

  static bool classof(const DriverConfig *config) {
    return config->getKind() == OptionKind::kCustomOps;
  }

  const std::filesystem::path &getPath() const { return customOpsPath; }

private:
  std::filesystem::path customOpsPath;
};

} // namespace M

#endif // GENERICML_DRIVER_DRIVERCONFIG_H
