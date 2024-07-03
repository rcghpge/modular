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
  enum OptionKind {
    kTF,
    kTorch,
    kMOF,
    kONNX,
    kKernelLibrary,
    kVisualization,
    kDevice
  };

  DriverConfig(OptionKind kind, llvm::StringLiteral name)
      : Kind(kind), configName(name) {}

  OptionKind getKind() const { return Kind; }

  llvm::StringLiteral getConfigName() const { return configName; }

  virtual ~DriverConfig() = default;

  virtual std::unique_ptr<DriverConfig> clone() {
    return std::make_unique<DriverConfig>(*this);
  }

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

  std::unique_ptr<DriverConfig> clone() override {
    return std::make_unique<KernelLibraryDriverConfig>(*this);
  }

private:
  std::filesystem::path kernelLibraryPath;
};

class VisualizationConfig : public DriverConfig {

public:
  VisualizationConfig(std::filesystem::path path)
      : DriverConfig(OptionKind::kVisualization, "viz-spec-path"),
        visualizationPath(std::move(path)) {}

  static bool classof(const DriverConfig *config) {
    return config->getKind() == OptionKind::kVisualization;
  }

  const std::filesystem::path &getPath() const { return visualizationPath; }

  std::unique_ptr<DriverConfig> clone() override {
    return std::make_unique<VisualizationConfig>(*this);
  }

private:
  std::filesystem::path visualizationPath;
};

class DeviceConfig : public DriverConfig {
public:
  DeviceConfig(std::string device, size_t deviceID)
      : DriverConfig(OptionKind::kDevice, "device-configuration"),
        deviceLabel(std::move(device)), deviceID(deviceID) {}

  static bool classof(const DriverConfig *config) {
    return config->getKind() == OptionKind::kDevice;
  }

  const std::string &getDeviceLabel() const { return deviceLabel; }
  size_t getDeviceID() const { return deviceID; }

  std::unique_ptr<DriverConfig> clone() override {
    return std::make_unique<DeviceConfig>(*this);
  }

private:
  std::string deviceLabel;
  size_t deviceID;
};

} // namespace M

#endif // GENERICML_DRIVER_DRIVERCONFIG_H
