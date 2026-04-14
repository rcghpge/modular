//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

// Adapted by Modular.
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#ifndef MLRT_DRIVER_NIXL_PLUGIN_MANAGER_H
#define MLRT_DRIVER_NIXL_PLUGIN_MANAGER_H

#include "MLRT/Driver/NIXL/backend/backend_plugin.h"
#include "MLRT/Driver/NIXL/nixl_types.h"
#include "MLRT/Driver/NIXL/telemetry/telemetry_plugin.h"

#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace M::MLRT::NIXL {

// Forward declarations
class nixlBackendEngine;
struct nixlBackendInitParams;

/**
 * This class represents a NIXL plugin and is used to create plugin instances.
 * nixlPluginHandle attributes are modified only in the constructor and
 * destructor and remain unchanged during normal operation, e.g., query
 * operations and plugin instance creation. This allows using it in
 * multi-threading environments without lock protection.
 */
class nixlPluginHandle {
public:
  explicit nixlPluginHandle(void *handle) : handle_(handle) {}
  virtual ~nixlPluginHandle() = default;

  nixlPluginHandle(const nixlPluginHandle &) = delete;
  nixlPluginHandle &operator=(const nixlPluginHandle &) = delete;
  nixlPluginHandle(nixlPluginHandle &&) = delete;
  nixlPluginHandle &operator=(nixlPluginHandle &&) = delete;

  virtual const char *getName() const = 0;
  virtual const char *getVersion() const = 0;

protected:
  void *handle_;
};

class nixlBackendPluginHandle : public nixlPluginHandle {
public:
  nixlBackendPluginHandle(void *handle, nixlBackendPlugin *plugin);
  ~nixlBackendPluginHandle() override;

  nixlBackendEngine *
  createEngine(const nixlBackendInitParams *initParams) const;
  void destroyEngine(nixlBackendEngine *engine) const;
  const char *getName() const override;
  const char *getVersion() const override;
  nixl_b_params_t getBackendOptions() const;
  nixl_mem_list_t getBackendMems() const;

private:
  nixlBackendPlugin *plugin_;
};

struct nixlBackendStaticPluginInfo {
  std::string name;
  nixlStaticPluginCreatorFunc createFunc;
};

class nixlTelemetryPluginHandle : public nixlPluginHandle {
public:
  nixlTelemetryPluginHandle(void *handle, nixlTelemetryPlugin *plugin);
  ~nixlTelemetryPluginHandle() override;

  std::unique_ptr<nixlTelemetryExporter>
  createExporter(const nixlTelemetryExporterInitParams &initParams) const;
  const char *getName() const override;
  const char *getVersion() const override;

private:
  nixlTelemetryPlugin *plugin_;
};

struct nixlTelemetryStaticPluginInfo {
  std::string name;
  nixlTelemetryStaticPluginCreatorFunc createFunc;
};

using nixlPluginLoaderFunc = std::shared_ptr<const nixlPluginHandle> (*)(
    void *handle, const std::string &pluginPath);

class nixlPluginManager {
public:
  static nixlPluginManager &getInstance();

  nixlPluginManager(const nixlPluginManager &) = delete;
  nixlPluginManager &operator=(const nixlPluginManager &) = delete;
  nixlPluginManager(nixlPluginManager &&) = delete;
  nixlPluginManager &operator=(nixlPluginManager &&) = delete;

  void loadPluginsFromList(const std::string &filename);

  std::shared_ptr<const nixlBackendPluginHandle>
  loadBackendPlugin(const nixl_backend_t &pluginName);
  std::shared_ptr<const nixlTelemetryPluginHandle>
  loadTelemetryPlugin(const nixl_telemetry_plugin_t &pluginName);

  void unloadBackendPlugin(const nixl_backend_t &pluginName);
  void unloadTelemetryPlugin(const nixl_telemetry_plugin_t &pluginName);

  std::shared_ptr<const nixlBackendPluginHandle>
  getBackendPlugin(const nixl_backend_t &pluginName);
  std::shared_ptr<const nixlTelemetryPluginHandle>
  getTelemetryPlugin(const nixl_telemetry_plugin_t &pluginName);

  std::vector<nixl_backend_t> getLoadedBackendPluginNames();
  std::vector<nixl_telemetry_plugin_t> getLoadedTelemetryPluginNames();

  void addPluginDirectory(const std::string &directory);

  const std::vector<nixlBackendStaticPluginInfo> &getBackendStaticPlugins();
  const std::vector<nixlTelemetryStaticPluginInfo> &getTelemetryStaticPlugins();

  nixl_b_params_t getBackendOptions(const nixl_backend_t &type);

private:
  std::map<nixl_backend_t, std::shared_ptr<const nixlBackendPluginHandle>>
      loadedBackendPlugins_;
  std::map<nixl_telemetry_plugin_t,
           std::shared_ptr<const nixlTelemetryPluginHandle>>
      loadedTelemetryPlugins_;
  std::vector<std::string> pluginDirs_;
  std::vector<nixlBackendStaticPluginInfo> backendStaticPlugins_;
  std::vector<nixlTelemetryStaticPluginInfo> telemetryStaticPlugins_;
  std::mutex lock;

  nixlPluginManager();

  void registerBuiltinPlugins();
  void registerBackendStaticPlugin(const std::string &name,
                                   nixlStaticPluginCreatorFunc creator);
  void
  registerTelemetryStaticPlugin(const std::string &name,
                                nixlTelemetryStaticPluginCreatorFunc creator);

  void discoverPluginsFromDir(const std::filesystem::path &dirpath);
  void discoverBackendPlugin(const std::string &filename);
  void discoverTelemetryPlugin(const std::string &filename);

  std::shared_ptr<const nixlPluginHandle>
  loadPluginFromPath(const std::string &pluginPath,
                     nixlPluginLoaderFunc loader);

  std::string composePluginPath(const std::string &dir,
                                const std::string &pluginPrefix,
                                const std::string &pluginName);
};

} // namespace M::MLRT::NIXL

#endif // MLRT_DRIVER_NIXL_PLUGIN_MANAGER_H
