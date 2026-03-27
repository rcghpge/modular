//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

// Adapted by Modular.
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#include "MLRT/Driver/NIXL/plugin_manager.h"

#include "MLRT/Driver/NIXL/nixl.h"
#include "MLRT/Driver/NIXL/utils/common/nixl_log.h"

#include <algorithm>
#include <cctype>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <string>

namespace M::AsyncRT::NIXL {
namespace {

using lock_guard = const std::lock_guard<std::mutex>;

inline constexpr char backendPluginPrefix[] = "libplugin_";
inline constexpr char telemetryPluginPrefix[] = "libtelemetry_exporter_";
inline constexpr char kPluginSuffix[] = ".so";

std::string normalizePluginDirectory(const std::string &directory) {
  return std::filesystem::path(directory).lexically_normal().string();
}

std::string canonicalPluginName(const char *name) {
  std::string pluginName = name ? std::string(name) : std::string();
  std::transform(pluginName.begin(), pluginName.end(), pluginName.begin(),
                 [](unsigned char ch) { return std::tolower(ch); });
  return pluginName;
}

std::shared_ptr<const nixlPluginHandle>
backendLoader(void *handle, const std::string &pluginPath) {
  using init_func_t = nixlBackendPlugin *(*)();
  init_func_t init =
      reinterpret_cast<init_func_t>(dlsym(handle, "nixl_plugin_init"));
  if (!init) {
    NIXL_ERROR << "Failed to find nixl_plugin_init in " << pluginPath << ": "
               << dlerror();
    dlclose(handle);
    return nullptr;
  }

  nixlBackendPlugin *plugin = init();
  if (!plugin) {
    NIXL_ERROR << "Plugin initialization failed for " << pluginPath;
    dlclose(handle);
    return nullptr;
  }

  if (plugin->api_version != NIXL_PLUGIN_API_VERSION) {
    NIXL_ERROR << "Plugin API version mismatch for " << pluginPath
               << ": expected " << NIXL_PLUGIN_API_VERSION << ", got "
               << plugin->api_version;
    dlclose(handle);
    return nullptr;
  }

  return std::make_shared<const nixlBackendPluginHandle>(handle, plugin);
}

std::shared_ptr<const nixlPluginHandle>
telemetryLoader(void *handle, const std::string &pluginPath) {
  using init_func_t = nixlTelemetryPlugin *(*)();
  init_func_t init = reinterpret_cast<init_func_t>(
      dlsym(handle, "nixl_telemetry_plugin_init"));
  if (!init) {
    NIXL_ERROR << "Failed to find nixl_telemetry_plugin_init in " << pluginPath
               << ": " << dlerror();
    dlclose(handle);
    return nullptr;
  }

  nixlTelemetryPlugin *plugin = init();
  if (!plugin) {
    NIXL_ERROR << "Plugin initialization failed for " << pluginPath;
    dlclose(handle);
    return nullptr;
  }

  if (plugin->api_version != nixlTelemetryPluginApiVersionV1) {
    NIXL_ERROR << "Plugin API version mismatch for " << pluginPath
               << ": expected "
               << static_cast<int>(nixlTelemetryPluginApiVersionV1) << ", got "
               << static_cast<int>(plugin->api_version);
    dlclose(handle);
    return nullptr;
  }

  return std::make_shared<const nixlTelemetryPluginHandle>(handle, plugin);
}

std::map<nixl_backend_t, std::string>
loadPluginList(const std::string &filename) {
  std::map<nixl_backend_t, std::string> plugins;
  std::ifstream file(filename);
  if (!file.is_open()) {
    NIXL_ERROR << "Failed to open plugin list file: " << filename;
    return plugins;
  }

  std::string line;
  while (std::getline(file, line)) {
    if (line.empty() || line[0] == '#') {
      continue;
    }

    size_t pos = line.find('=');
    if (pos == std::string::npos) {
      continue;
    }

    std::string name = line.substr(0, pos);
    std::string path = line.substr(pos + 1);
    auto trim = [](std::string &s) {
      s.erase(0, s.find_first_not_of(" \t"));
      s.erase(s.find_last_not_of(" \t") + 1);
    };
    trim(name);
    trim(path);
    plugins[name] = path;
  }

  return plugins;
}

std::string getPluginDir() {
  if (const char *pluginDir = std::getenv("MODULAR_NIXL_PLUGIN_DIR")) {
    return normalizePluginDirectory(pluginDir);
  }

  Dl_info info;
  if (!dladdr(reinterpret_cast<void *>(&getPluginDir), &info)) {
    NIXL_ERROR << "Failed to get plugin directory from dladdr";
    return "";
  }
  return (std::filesystem::path(info.dli_fname).parent_path() / "plugins")
      .string();
}

bool startsWith(const std::string &str, const std::string &prefix) {
  return str.size() >= prefix.size() &&
         std::equal(prefix.begin(), prefix.end(), str.begin());
}

bool endsWith(const std::string &str, const std::string &suffix) {
  return str.size() >= suffix.size() &&
         std::equal(suffix.rbegin(), suffix.rend(), str.rbegin());
}

std::string extractPluginName(const std::string &filename,
                              const std::string &prefix) {
  return filename.substr(prefix.size(), filename.size() - prefix.size() -
                                            (sizeof(kPluginSuffix) - 1));
}

} // namespace

nixlBackendPluginHandle::nixlBackendPluginHandle(void *handle,
                                                 nixlBackendPlugin *plugin)
    : nixlPluginHandle(handle), plugin_(plugin) {}

nixlBackendPluginHandle::~nixlBackendPluginHandle() {
  if (handle_) {
    using fini_func_t = void (*)();
    fini_func_t fini =
        reinterpret_cast<fini_func_t>(dlsym(handle_, "nixl_plugin_fini"));
    if (fini) {
      fini();
    }
    dlclose(handle_);
    handle_ = nullptr;
    plugin_ = nullptr;
  }
}

nixlBackendEngine *nixlBackendPluginHandle::createEngine(
    const nixlBackendInitParams *initParams) const {
  if (plugin_ && plugin_->create_engine) {
    return plugin_->create_engine(initParams);
  }
  return nullptr;
}

void nixlBackendPluginHandle::destroyEngine(nixlBackendEngine *engine) const {
  if (plugin_ && plugin_->destroy_engine && engine) {
    plugin_->destroy_engine(engine);
  }
}

const char *nixlBackendPluginHandle::getName() const {
  if (plugin_ && plugin_->get_plugin_name) {
    return plugin_->get_plugin_name();
  }
  return "unknown";
}

const char *nixlBackendPluginHandle::getVersion() const {
  if (plugin_ && plugin_->get_plugin_version) {
    return plugin_->get_plugin_version();
  }
  return "unknown";
}

nixl_b_params_t nixlBackendPluginHandle::getBackendOptions() const {
  nixl_b_params_t params;
  if (plugin_ && plugin_->get_backend_options) {
    return plugin_->get_backend_options();
  }
  return params;
}

nixl_mem_list_t nixlBackendPluginHandle::getBackendMems() const {
  nixl_mem_list_t mems;
  if (plugin_ && plugin_->get_backend_mems) {
    return plugin_->get_backend_mems();
  }
  return mems;
}

nixlTelemetryPluginHandle::nixlTelemetryPluginHandle(
    void *handle, nixlTelemetryPlugin *plugin)
    : nixlPluginHandle(handle), plugin_(plugin) {}

nixlTelemetryPluginHandle::~nixlTelemetryPluginHandle() {
  if (handle_) {
    using fini_func_t = void (*)();
    fini_func_t fini = reinterpret_cast<fini_func_t>(
        dlsym(handle_, "nixl_telemetry_plugin_fini"));
    if (fini) {
      fini();
    }
    dlclose(handle_);
    handle_ = nullptr;
    plugin_ = nullptr;
  }
}

std::unique_ptr<nixlTelemetryExporter>
nixlTelemetryPluginHandle::createExporter(
    const nixlTelemetryExporterInitParams &initParams) const {
  if (plugin_ && plugin_->create_exporter) {
    return plugin_->create_exporter(initParams);
  }
  return nullptr;
}

const char *nixlTelemetryPluginHandle::getName() const {
  if (plugin_) {
    return plugin_->getName().c_str();
  }
  return "unknown";
}

const char *nixlTelemetryPluginHandle::getVersion() const {
  if (plugin_) {
    return plugin_->getVersion().c_str();
  }
  return "unknown";
}

std::shared_ptr<const nixlPluginHandle>
nixlPluginManager::loadPluginFromPath(const std::string &pluginPath,
                                      nixlPluginLoaderFunc loader) {
  void *handle =
      dlopen(pluginPath.c_str(), RTLD_NOW | RTLD_LOCAL | RTLD_NODELETE);
  if (!handle) {
    NIXL_INFO << "Failed to load plugin from " << pluginPath << ": "
              << dlerror();
    return nullptr;
  }

  return loader(handle, pluginPath);
}

void nixlPluginManager::loadPluginsFromList(const std::string &filename) {
  auto plugins = loadPluginList(filename);
  lock_guard lg(lock);

  for (const auto &pair : plugins) {
    auto pluginHandle = loadPluginFromPath(pair.second, backendLoader);
    if (pluginHandle) {
      auto backendPlugin =
          std::static_pointer_cast<const nixlBackendPluginHandle>(pluginHandle);
      loadedBackendPlugins_[canonicalPluginName(pair.first.c_str())] =
          backendPlugin;
    }
  }
}

nixlPluginManager::nixlPluginManager() {
  // `libplugin_ucx_(cuda|rocm)_verbs.so` depends on `libibverbs.so` and
  // `libmlx5.so`. `libibverbs.so` is linked at build time, but `libmlx5.so`
  // must already be present in the process before we `dlopen` the verbs
  // plugin. Only one UCX backend variant should be loaded in a process, and
  // systems without rdma-core `-dev` packages often only ship the versioned
  // soname.
  void *dlLibMlx5 = dlopen("libmlx5.so.1", RTLD_NOW | RTLD_GLOBAL);

  // CUDA UCX backends rely on both the CUDA driver and NVML being available.
  void *dlLibCuda = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
  void *dlLibNvml = dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_GLOBAL);

  // ROCm UCX backends rely on the HSA runtime being available.
  void *dlLibHsaRuntime64 =
      dlopen("libhsa-runtime64.so.1", RTLD_NOW | RTLD_GLOBAL);

  // IB devices are only relevant when deciding whether to try the verbs
  // variants first.
  const char *ibDevices = std::getenv("UCX_NET_DEVICES");

  std::string pluginDir = getPluginDir();
  if (!pluginDir.empty()) {
    NIXL_DEBUG << "Loading plugins from: " << pluginDir;
    pluginDirs_.insert(pluginDirs_.begin(), pluginDir);
    discoverPluginsFromDir(pluginDir);
  }

  registerBuiltinPlugins();

  if (dlLibHsaRuntime64 != nullptr) {
    if (ibDevices != nullptr && dlLibMlx5 != nullptr &&
        loadBackendPlugin("ucx_rocm_verbs") != nullptr) {
      NIXL_INFO << "Loaded NIXL plugin_ucx_rocm_verbs";
      return;
    }
    if (loadBackendPlugin("ucx_rocm") != nullptr) {
      NIXL_INFO << "Loaded NIXL plugin_ucx_rocm";
      return;
    }
  }

  if (dlLibCuda != nullptr && dlLibNvml != nullptr) {
    if (ibDevices != nullptr && dlLibMlx5 != nullptr &&
        loadBackendPlugin("ucx_cuda_verbs") != nullptr) {
      NIXL_INFO << "Loaded NIXL plugin_ucx_cuda_verbs";
      return;
    }
    if (loadBackendPlugin("ucx_cuda") != nullptr) {
      NIXL_INFO << "Loaded NIXL plugin_ucx_cuda";
      return;
    }
  }

  if (loadBackendPlugin("ucx_cpu") != nullptr) {
    NIXL_INFO << "Loaded NIXL plugin_ucx_cpu";
  }
}

nixlPluginManager &nixlPluginManager::getInstance() {
  // Meyers singleton initialization is thread-safe in modern C++.
  static nixlPluginManager instance;
  return instance;
}

void nixlPluginManager::addPluginDirectory(const std::string &directory) {
  if (directory.empty()) {
    NIXL_ERROR << "Cannot add empty plugin directory";
    return;
  }

  std::string normalizedDirectory = normalizePluginDirectory(directory);
  if (!std::filesystem::exists(normalizedDirectory) ||
      !std::filesystem::is_directory(normalizedDirectory)) {
    NIXL_ERROR << "Plugin directory does not exist or is not readable: "
               << normalizedDirectory;
    return;
  }

  {
    lock_guard lg(lock);
    for (const auto &dir : pluginDirs_) {
      if (dir == normalizedDirectory) {
        NIXL_WARN << "Plugin directory already registered: "
                  << normalizedDirectory;
        return;
      }
    }
    pluginDirs_.insert(pluginDirs_.begin(), normalizedDirectory);
  }

  discoverPluginsFromDir(normalizedDirectory);
}

std::string
nixlPluginManager::composePluginPath(const std::string &dir,
                                     const std::string &pluginPrefix,
                                     const std::string &pluginName) {
  if (dir.empty()) {
    return "";
  }
  if (dir.back() == '/') {
    return dir + pluginPrefix + pluginName + kPluginSuffix;
  }
  return dir + "/" + pluginPrefix + pluginName + kPluginSuffix;
}

std::shared_ptr<const nixlBackendPluginHandle>
nixlPluginManager::loadBackendPlugin(const nixl_backend_t &pluginName) {
  lock_guard lg(lock);

  std::string canonicalName = canonicalPluginName(pluginName.c_str());

  auto it = loadedBackendPlugins_.find(canonicalName);
  if (it != loadedBackendPlugins_.end()) {
    return it->second;
  }

  for (const auto &pair : loadedBackendPlugins_) {
    // The shared library name can encode implementation details such as
    // `ucx_cuda` or `ucx_rocm_verbs`, while the plugin itself still reports
    // the logical backend name `ucx`.
    if (canonicalPluginName(pair.second->getName()) == canonicalName) {
      return pair.second;
    }
  }

  for (const auto &dir : pluginDirs_) {
    std::string pluginPath =
        composePluginPath(dir, backendPluginPrefix, pluginName);
    if (pluginPath.empty()) {
      continue;
    }
    if (!std::filesystem::exists(pluginPath)) {
      NIXL_WARN << "Plugin file does not exist: " << pluginPath;
      continue;
    }

    auto pluginHandle = loadPluginFromPath(pluginPath, backendLoader);
    if (pluginHandle) {
      auto backendPlugin =
          std::static_pointer_cast<const nixlBackendPluginHandle>(pluginHandle);
      loadedBackendPlugins_[canonicalName] = backendPlugin;
      return backendPlugin;
    }
  }

  NIXL_INFO << "Failed to load plugin '" << pluginName
            << "' from any directory";
  return nullptr;
}

std::shared_ptr<const nixlTelemetryPluginHandle>
nixlPluginManager::loadTelemetryPlugin(
    const nixl_telemetry_plugin_t &pluginName) {
  lock_guard lg(lock);

  auto it = loadedTelemetryPlugins_.find(pluginName);
  if (it != loadedTelemetryPlugins_.end()) {
    return it->second;
  }

  for (const auto &dir : pluginDirs_) {
    std::string pluginPath =
        composePluginPath(dir, telemetryPluginPrefix, pluginName);
    if (pluginPath.empty()) {
      continue;
    }
    if (!std::filesystem::exists(pluginPath)) {
      NIXL_WARN << "Plugin file does not exist: " << pluginPath;
      continue;
    }

    auto pluginHandle = loadPluginFromPath(pluginPath, telemetryLoader);
    if (pluginHandle) {
      auto telemetryPlugin =
          std::static_pointer_cast<const nixlTelemetryPluginHandle>(
              pluginHandle);
      loadedTelemetryPlugins_[pluginName] = telemetryPlugin;
      return telemetryPlugin;
    }
  }

  NIXL_INFO << "Failed to load plugin '" << pluginName
            << "' from any directory";
  return nullptr;
}

void nixlPluginManager::discoverBackendPlugin(const std::string &filename) {
  if (startsWith(filename, backendPluginPrefix) &&
      endsWith(filename, kPluginSuffix)) {
    std::string pluginName = extractPluginName(filename, backendPluginPrefix);
    auto plugin = loadBackendPlugin(pluginName);
    if (plugin) {
      NIXL_INFO << "Discovered and loaded backend plugin: " << pluginName;
    }
  }
}

void nixlPluginManager::discoverTelemetryPlugin(const std::string &filename) {
  if (startsWith(filename, telemetryPluginPrefix) &&
      endsWith(filename, kPluginSuffix)) {
    std::string pluginName = extractPluginName(filename, telemetryPluginPrefix);
    auto plugin = loadTelemetryPlugin(pluginName);
    if (plugin) {
      NIXL_INFO << "Discovered and loaded telemetry plugin: " << pluginName;
    }
  }
}

void nixlPluginManager::discoverPluginsFromDir(
    const std::filesystem::path &dirpath) {
  std::error_code ec;
  std::filesystem::directory_iterator dirIter(dirpath, ec);
  if (ec) {
    NIXL_ERROR << "Error accessing directory(" << dirpath
               << "): " << ec.message();
    return;
  }

  for (const auto &entry : dirIter) {
    std::string filename = entry.path().filename().string();
    discoverBackendPlugin(filename);
    discoverTelemetryPlugin(filename);
  }
}

void nixlPluginManager::unloadBackendPlugin(const nixl_backend_t &pluginName) {
  for (const auto &splugin : getBackendStaticPlugins()) {
    if (canonicalPluginName(splugin.name.c_str()) ==
        canonicalPluginName(pluginName.c_str())) {
      return;
    }
  }
  lock_guard lg(lock);
  const std::string canonicalName = canonicalPluginName(pluginName.c_str());
  loadedBackendPlugins_.erase(canonicalName);
  for (auto it = loadedBackendPlugins_.begin();
       it != loadedBackendPlugins_.end();) {
    if (canonicalPluginName(it->second->getName()) == canonicalName) {
      it = loadedBackendPlugins_.erase(it);
    } else {
      ++it;
    }
  }
}

void nixlPluginManager::unloadTelemetryPlugin(
    const nixl_telemetry_plugin_t &pluginName) {
  for (const auto &splugin : getTelemetryStaticPlugins()) {
    if (splugin.name == pluginName) {
      return;
    }
  }
  lock_guard lg(lock);
  loadedTelemetryPlugins_.erase(pluginName);
}

std::shared_ptr<const nixlBackendPluginHandle>
nixlPluginManager::getBackendPlugin(const nixl_backend_t &pluginName) {
  lock_guard lg(lock);
  const std::string canonicalName = canonicalPluginName(pluginName.c_str());
  auto it = loadedBackendPlugins_.find(canonicalName);
  if (it != loadedBackendPlugins_.end()) {
    return it->second;
  }
  for (const auto &pair : loadedBackendPlugins_) {
    if (canonicalPluginName(pair.second->getName()) == canonicalName) {
      return pair.second;
    }
  }
  return nullptr;
}

std::shared_ptr<const nixlTelemetryPluginHandle>
nixlPluginManager::getTelemetryPlugin(
    const nixl_telemetry_plugin_t &pluginName) {
  lock_guard lg(lock);
  auto it = loadedTelemetryPlugins_.find(pluginName);
  if (it != loadedTelemetryPlugins_.end()) {
    return it->second;
  }
  return nullptr;
}

std::vector<nixl_backend_t> nixlPluginManager::getLoadedBackendPluginNames() {
  lock_guard lg(lock);
  std::vector<nixl_backend_t> names;
  for (const auto &pair : loadedBackendPlugins_) {
    const std::string canonicalName =
        canonicalPluginName(pair.second->getName());
    if (std::find(names.begin(), names.end(), canonicalName) == names.end()) {
      names.push_back(canonicalName);
    }
  }
  return names;
}

std::vector<nixl_telemetry_plugin_t>
nixlPluginManager::getLoadedTelemetryPluginNames() {
  lock_guard lg(lock);
  std::vector<nixl_telemetry_plugin_t> names;
  for (const auto &pair : loadedTelemetryPlugins_) {
    names.push_back(pair.first);
  }
  return names;
}

void nixlPluginManager::registerBackendStaticPlugin(
    const std::string &name, nixlStaticPluginCreatorFunc creator) {
  lock_guard lg(lock);
  nixlBackendStaticPluginInfo info{name, creator};
  backendStaticPlugins_.push_back(info);
  nixlBackendPlugin *plugin = info.createFunc();
  NIXL_INFO << "Loading static plugin: " << name;
  if (plugin) {
    auto pluginHandle =
        std::make_shared<const nixlBackendPluginHandle>(nullptr, plugin);
    loadedBackendPlugins_[name] = pluginHandle;
  }
}

void nixlPluginManager::registerTelemetryStaticPlugin(
    const std::string &name, nixlTelemetryStaticPluginCreatorFunc creator) {
  lock_guard lg(lock);
  nixlTelemetryStaticPluginInfo info{name, creator};
  telemetryStaticPlugins_.push_back(info);
  nixlTelemetryPlugin *plugin = info.createFunc();
  NIXL_INFO << "Loading static plugin: " << name;
  if (plugin) {
    auto pluginHandle =
        std::make_shared<const nixlTelemetryPluginHandle>(nullptr, plugin);
    loadedTelemetryPlugins_[name] = pluginHandle;
  }
}

const std::vector<nixlBackendStaticPluginInfo> &
nixlPluginManager::getBackendStaticPlugins() {
  return backendStaticPlugins_;
}

const std::vector<nixlTelemetryStaticPluginInfo> &
nixlPluginManager::getTelemetryStaticPlugins() {
  return telemetryStaticPlugins_;
}

void nixlPluginManager::registerBuiltinPlugins() {
  extern nixlTelemetryPlugin *createStaticBUFFERPlugin();
  registerTelemetryStaticPlugin("BUFFER", createStaticBUFFERPlugin);
}

nixl_b_params_t
nixlPluginManager::getBackendOptions(const nixl_backend_t &type) {
  auto plugin = getBackendPlugin(type);
  if (!plugin) {
    plugin = loadBackendPlugin(type);
  }
  if (plugin) {
    return plugin->getBackendOptions();
  }
  return {};
}

} // namespace M::AsyncRT::NIXL
