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
    // Skip empty lines and comments
    if (line.empty() || line[0] == '#') {
      continue;
    }

    // Find the equals sign
    size_t pos = line.find('=');
    if (pos != std::string::npos) {
      std::string name = line.substr(0, pos);
      std::string path = line.substr(pos + 1);

      auto trim = [](std::string &s) {
        s.erase(0, s.find_first_not_of(" \t"));
        s.erase(s.find_last_not_of(" \t") + 1);
      };
      trim(name);
      trim(path);

      // Add to map
      plugins[name] = path;
    }
  }

  return plugins;
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
  auto plugin_handle = std::make_shared<const nixlPluginHandle>(handle, plugin);

  return plugin_handle;
}

void nixlPluginManager::loadPluginsFromList(const std::string &filename) {
  auto plugins = loadPluginList(filename);

  lock_guard lg(lock);

  for (const auto &[name, path] : plugins) {
    auto plugin_handle = loadPluginFromPath(path);
    if (plugin_handle)
      loaded_plugins_[name] = plugin_handle;
  }
}

// PluginManager implementation
nixlPluginManager::nixlPluginManager() {
  // Check for MODULAR_NIXL_PLUGIN_DIR environment variable
  const char *pluginDir = getenv("MODULAR_NIXL_PLUGIN_DIR");
  if (pluginDir) {
    NIXL_DEBUG << "Loading plugins from directory: " << pluginDir << "\n";
    // Clean up `.`s and `..`s
    const auto normalizedPluginDir =
        std::filesystem::path(pluginDir).lexically_normal();
    // Insert at the beginning for priority.
    plugin_dirs_.insert(plugin_dirs_.begin(), normalizedPluginDir.string());
    discoverPluginsFromDir(normalizedPluginDir);
  }

  registerBuiltinPlugins();
}

nixlPluginManager &nixlPluginManager::getInstance() {
  // Meyers singleton initialization is safe in multi-threaded environment.
  // Consult standard [stmt.dcl] chapter for details.
  static nixlPluginManager instance;

  return instance;
}

void nixlPluginManager::addPluginDirectory(
    const std::filesystem::path &directory) {
  if (directory.empty()) {
    NIXL_ERROR << "Cannot add empty plugin directory";
    return;
  }

  // Check if directory exists
  if (!std::filesystem::exists(directory) ||
      !std::filesystem::is_directory(directory)) {
    NIXL_ERROR << "Plugin directory does not exist or is not readable: "
               << directory;
    return;
  }

  {
    lock_guard lg(lock);

    // Check if directory is already in the list
    for (const auto &dir : plugin_dirs_) {
      if (dir == directory) {
        NIXL_WARN << "Plugin directory already registered: " << directory;
        return;
      }
    }

    // Prioritize the new directory by inserting it at the beginning
    plugin_dirs_.insert(plugin_dirs_.begin(), directory);
  }

  discoverPluginsFromDir(directory);
}

std::shared_ptr<const nixlPluginHandle>
nixlPluginManager::loadPlugin(const std::string &plugin_name) {
  lock_guard lg(lock);

  // Look up a canonical plugin name.
  std::string canonPlugName = StringRef(plugin_name).trim().lower();

  // Check if the plugin is already loaded
  // Static Plugins are preloaded so return handle
  auto it = loaded_plugins_.find(canonPlugName);
  if (it != loaded_plugins_.end()) {
    return it->second;
  }

  // Try to load the plugin from all registered directories
  for (const auto &dir : plugin_dirs_) {
    // Handle path joining correctly with or without trailing slash
    std::string plugin_path;
    if (dir.empty()) {
      continue;
    } else if (dir.back() == '/') {
      plugin_path =
          (Twine(dir) + kPluginPrefix + canonPlugName + kPluginSuffix).str();
    } else {
      plugin_path =
          (Twine(dir) + "/" + kPluginPrefix + canonPlugName + kPluginSuffix)
              .str();
    }

    // Check if the plugin file exists before attempting to load i
    if (!std::filesystem::exists(plugin_path)) {
      NIXL_WARN << "Plugin file does not exist: " << plugin_path;
      continue;
    }

    std::shared_ptr<const nixlPluginHandle> plugin_handle =
        loadPluginFromPath(plugin_path);
    if (plugin_handle) {
      loaded_plugins_[canonPlugName] = plugin_handle;
      return plugin_handle;
    }
  }

  // Failed to load the plugin
  NIXL_ERROR << "Failed to load plugin '" << canonPlugName
             << "' from any directory\n";
  return nullptr;
}

void nixlPluginManager::discoverPluginsFromDir(
    const std::filesystem::path &dirPath) {
  // Calculate the minimum length a filename must have to potentially match the
  // pattern (length of prefix + length of suffix). A filename shorter than this
  // cannot contain both the required prefix and suffix.
  constexpr size_t kMinPluginFilenameLength =
      kPluginPrefix.length() + kPluginSuffix.length();

  std::error_code ec;
  std::filesystem::directory_iterator directoryIter(dirPath, ec);
  if (ec) {
    NIXL_ERROR << "Error accessing plugin directory '" << dirPath
               << "': " << ec.message() << '\n';
    return;
  }

  for (const auto &entry : directoryIter) {
    // Check if the entry is a regular file before proceeding.
    // We only care about files, not subdirectories or other types.
    if (!entry.is_regular_file())
      continue;

    std::string filename = entry.path().filename().string();

    // Perform checks based on the expected plugin filename pattern.

    // 1. Check if the filename is long enough to contain the prefix and suffix.
    if (filename.length() < kMinPluginFilenameLength)
      continue;

    // 2. Check if the filename starts with the required prefix.
    if (!StringRef(filename).starts_with(kPluginPrefix))
      continue;

    // 3. Check if the filename ends with the required suffix.
    if (!StringRef(filename).ends_with(kPluginSuffix))
      continue;

    // Extract the plugin name.

    // If all checks passed, the filename matches the pattern
    // "libplugin_<plugin_name>.so".
    // The plugin name is the substring between the prefix and the suffix.
    size_t nameStartPos = kPluginPrefix.length();
    size_t nameLength = filename.length() - kMinPluginFilenameLength;

    // Ensure the extracted name is not empty (e.g., avoids processing
    // "libplugin_.so")
    if (nameLength == 0) {
      NIXL_WARN << "Skipping potential plugin file with empty name: "
                << filename;
      continue;
    }

    std::string pluginName = filename.substr(nameStartPos, nameLength);

    // Load the discovered plugin.

    // Attempt to load the plugin using its extracted name.
    // loadPlugin handles checking if it's already loaded.
    auto plugin = loadPlugin(pluginName);
    if (plugin) {
      // Use NIXL logging for consistency
      NIXL_INFO << "Discovered and loaded plugin '" << pluginName << "' from "
                << entry.path().string() << "\n";
    } else {
      NIXL_ERROR << "Failed to load discovered plugin: " << pluginName << "\n";
    }
  }
}

void nixlPluginManager::unloadPlugin(const nixl_backend_t &plugin_name) {
  // Skip unloading static plugins.
  for (const auto &splugin : getStaticPlugins()) {
    if (splugin.name == plugin_name)
      return;
  }

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

void nixlPluginManager::registerStaticPlugin(
    const char *name, nixlStaticPluginCreatorFunc creator) {
  lock_guard lg(lock);

  nixlStaticPluginInfo info;
  info.name = name;
  info.createFunc = creator;
  static_plugins_.push_back(info);

  // Static Plugins are considered pre-loaded
  nixlBackendPlugin *plugin = info.createFunc();
  if (plugin) {
    // Register the loaded plugin
    auto plugin_handle =
        std::make_shared<const nixlPluginHandle>(nullptr, plugin);
    loaded_plugins_[name] = plugin_handle;
  }
}

const std::vector<nixlStaticPluginInfo> &nixlPluginManager::getStaticPlugins() {
  return static_plugins_;
}

void nixlPluginManager::registerBuiltinPlugins() {}
