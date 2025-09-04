//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

// Adapted by Modular.
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#include "AsyncRT/NIXL/nixl.h"
#include "AsyncRT/NIXL/plugin_manager.h"
#include "AsyncRT/NIXL/utils/common/nixl_log.h"
#include <cstdlib> // For getenv
#include <dirent.h>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <unistd.h> // For access() and F_OK

using namespace M::AsyncRT::NIXL;

using lock_guard = const std::lock_guard<std::mutex>;

// pluginHandle implementation
nixlPluginHandle::nixlPluginHandle(void *handle, nixlBackendPlugin *plugin)
    : handle_(handle), plugin_(plugin) {}

nixlPluginHandle::~nixlPluginHandle() {
  if (handle_) {
    // Call the plugin's cleanup function
    typedef void (*fini_func_t)();
    fini_func_t fini = (fini_func_t)dlsym(handle_, "nixl_plugin_fini");
    if (fini) {
      fini();
    }

    // Close the dynamic library
    dlclose(handle_);
    handle_ = nullptr;
    plugin_ = nullptr;
  }
}

nixlBackendEngine *
nixlPluginHandle::createEngine(const nixlBackendInitParams *init_params) const {
  if (plugin_ && plugin_->create_engine) {
    return plugin_->create_engine(init_params);
  }
  return nullptr;
}

void nixlPluginHandle::destroyEngine(nixlBackendEngine *engine) const {
  if (plugin_ && plugin_->destroy_engine && engine) {
    plugin_->destroy_engine(engine);
  }
}

const char *nixlPluginHandle::getName() const {
  if (plugin_ && plugin_->get_plugin_name) {
    return plugin_->get_plugin_name();
  }
  return "unknown";
}

const char *nixlPluginHandle::getVersion() const {
  if (plugin_ && plugin_->get_plugin_version) {
    return plugin_->get_plugin_version();
  }
  return "unknown";
}

std::shared_ptr<const nixlPluginHandle>
nixlPluginManager::loadPluginFromPath(const std::string &plugin_path) {
  // Open the plugin file
  void *handle = dlopen(plugin_path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!handle) {
    NIXL_ERROR << "Failed to load plugin from " << plugin_path << ": "
               << dlerror();
    return nullptr;
  }

  // Get the initialization function
  typedef nixlBackendPlugin *(*init_func_t)();
  init_func_t init = (init_func_t)dlsym(handle, "nixl_plugin_init");
  if (!init) {
    NIXL_ERROR << "Failed to find nixl_plugin_init in " << plugin_path << ": "
               << dlerror();
    dlclose(handle);
    return nullptr;
  }

  // Call the initialization function
  nixlBackendPlugin *plugin = init();
  if (!plugin) {
    NIXL_ERROR << "Plugin initialization failed for " << plugin_path;
    dlclose(handle);
    return nullptr;
  }

  // Check API version
  if (plugin->api_version != NIXL_PLUGIN_API_VERSION) {
    NIXL_ERROR << "Plugin API version mismatch for " << plugin_path
               << ": expected " << NIXL_PLUGIN_API_VERSION << ", got "
               << plugin->api_version;
    dlclose(handle);
    return nullptr;
  }

  // Create and store the plugin handle
  auto pluginHandle = std::make_shared<const nixlPluginHandle>(handle, plugin);

  return pluginHandle;
}

// PluginManager implementation
nixlPluginManager::nixlPluginManager() {
  // The only NIXL plugin we support is UCX.

  // libplugin_ucx_(cuda|rocm)_verbs.so depend on libibverbs.so and libmlx5.so.
  // libibverbs.so is dynamically linked at build time. libmlx5.so must be
  // dlopen'ed at runtime prior to dlopen'ing libplugin_ucx_*_verbs.so.
  //
  // Exactly one of these UCX plugin files should be loaded. Loading both of
  // them will cause issues.
  void *dlLibMlx5 = dlopen("libmlx5.so", RTLD_NOW | RTLD_GLOBAL);

  // CUDA Driver + NVML is needed for CUDA.
  void *dlLibCuda = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
  void *dlLibNvml = dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_GLOBAL);

  // AMD HSA Runtime is needed for ROCm.
  void *dlLibHsaRuntime64 =
      dlopen("libhsa-runtime64.so.1", RTLD_NOW | RTLD_GLOBAL);

  // Check whether IB network devices are specified in the environment variable
  const char *ibDevices = getenv("UCX_NET_DEVICES");

  if (dlLibHsaRuntime64 != nullptr) {
    if (ibDevices != nullptr && dlLibMlx5 != nullptr &&
        loadPlugin("ucx_rocm_verbs") != nullptr) {
      NIXL_INFO << "Loaded NIXL plugin_ucx_rocm_verbs";
      return;
    }
    if (loadPlugin("ucx_rocm") != nullptr) {
      NIXL_INFO << "Loaded NIXL plugin_ucx_rocm";
      return;
    }
  }

  if (dlLibCuda != nullptr && dlLibNvml != nullptr) {
    if (ibDevices != nullptr && dlLibMlx5 != nullptr &&
        loadPlugin("ucx_cuda_verbs") != nullptr) {
      NIXL_INFO << "Loaded NIXL plugin_ucx_cuda_verbs";
      return;
    }

    if (loadPlugin("ucx_cuda") != nullptr) {
      NIXL_INFO << "Loaded NIXL plugin_ucx_cuda";
      return;
    }
  }

  NIXL_FATAL << "Unable to load CUDA or ROCm UCX backend";
}

nixlPluginManager &nixlPluginManager::getInstance() {
  // Meyers singleton initialization is safe in multi-threaded environment.
  // Consult standard [stmt.dcl] chapter for details.
  static nixlPluginManager instance;

  return instance;
}

std::shared_ptr<const nixlPluginHandle>
nixlPluginManager::loadPlugin(const std::string &pluginName) {
  lock_guard lg(lock);

  // Look up a canonical plugin name.
  std::string canonPlugName = StringRef(pluginName).trim().lower();

  // Check if the plugin is already loaded
  // Static Plugins are preloaded so return handle
  auto it = loaded_plugins_.find(canonPlugName);
  if (it != loaded_plugins_.end()) {
    return it->second;
  }

  auto pluginDirEnv = getenv("MODULAR_NIXL_PLUGIN_DIR");
  if (!pluginDirEnv) {
    NIXL_ERROR << "MODULAR_NIXL_PLUGIN_DIR is not set";
    return nullptr;
  }
  std::string pluginDir = std::string(pluginDirEnv);
  NIXL_DEBUG << "MODULAR_NIXL_PLUGIN_DIR=\"" << pluginDir << "\"";
  // Clean up `.`s and `..`s
  const auto dirPath = std::filesystem::path(pluginDir).lexically_normal();
  std::string dir = dirPath.string();

  // Handle path joining correctly with or without trailing slash
  std::string pluginPath;
  if (dir.back() == '/') {
    pluginPath =
        (Twine(dir) + kPluginPrefix + canonPlugName + kPluginSuffix).str();
  } else {
    pluginPath =
        (Twine(dir) + "/" + kPluginPrefix + canonPlugName + kPluginSuffix)
            .str();
  }

  NIXL_DEBUG << "Loading plugin from path: " << pluginPath;

  // Check if the plugin file exists before attempting to load it
  if (!std::filesystem::exists(pluginPath)) {
    NIXL_WARN << "Plugin file does not exist: " << pluginPath;
    return nullptr;
  }

  std::shared_ptr<const nixlPluginHandle> pluginHandle =
      loadPluginFromPath(pluginPath);
  if (pluginHandle) {
    // Even though the plugin file may be
    // "libplugin_ucx_(cuda|rocm)(_verbs)?.so", the plugin is "ucx" and not
    // "ucx_..." .
    auto loadedPluginName = StringRef(pluginHandle->getName()).trim().lower();
    loaded_plugins_[loadedPluginName] = pluginHandle;
    return pluginHandle;
  }

  // Failed to load the plugin
  NIXL_ERROR << "Failed to load plugin '" << canonPlugName
             << "' from any directory";
  return nullptr;
}

void nixlPluginManager::unloadPlugin(const nixl_backend_t &plugin_name) {
  lock_guard lg(lock);

  loaded_plugins_.erase(plugin_name);
}

std::shared_ptr<const nixlPluginHandle>
nixlPluginManager::getPlugin(const nixl_backend_t &plugin_name) {
  lock_guard lg(lock);

  auto it = loaded_plugins_.find(plugin_name);
  if (it != loaded_plugins_.end()) {
    return it->second;
  }
  return nullptr;
}

nixl_b_params_t nixlPluginHandle::getBackendOptions() const {
  nixl_b_params_t params;
  if (plugin_ && plugin_->get_backend_options) {
    return plugin_->get_backend_options();
  }
  return params; // Return empty params if not implemented
}

nixl_mem_list_t nixlPluginHandle::getBackendMems() const {
  nixl_mem_list_t mems;
  if (plugin_ && plugin_->get_backend_mems) {
    return plugin_->get_backend_mems();
  }
  return mems; // Return empty mems if not implemented
}

std::vector<nixl_backend_t> nixlPluginManager::getLoadedPluginNames() {
  lock_guard lg(lock);

  std::vector<nixl_backend_t> names;
  for (const auto &pair : loaded_plugins_) {
    names.push_back(pair.first);
  }
  return names;
}
