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
  enum OptionKind { kTF, kTorch, kMOF, kONNX, kKernelLibrary };

  DriverConfig(OptionKind kind, llvm::StringLiteral name)
      : Kind(kind), configName(name) {}

  OptionKind getKind() const { return Kind; }

  llvm::StringLiteral getConfigName() const { return configName; }

  virtual ~DriverConfig() = default;

private:
  const OptionKind Kind;
  const llvm::StringLiteral configName;
};

class KernelLibraryDriverConfig : public DriverConfig {

public:
  KernelLibraryDriverConfig(std::filesystem::path path)
      : DriverConfig(OptionKind::kKernelLibrary, "kernel-library-path"),
        kernelLibraryPath(std::move(path)) {}

  static bool classof(const DriverConfig *config) {
    return config->getKind() == OptionKind::kKernelLibrary;
  }

  const std::filesystem::path &getPath() const { return kernelLibraryPath; }

private:
  std::filesystem::path kernelLibraryPath;
};

} // namespace M

#endif // GENERICML_DRIVER_DRIVERCONFIG_H
