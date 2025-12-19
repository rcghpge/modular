//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

// Adapted by Modular.
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#ifndef MLRT_DRIVER_NIXL_BACKEND_BACKEND_PLUGIN_H
#define MLRT_DRIVER_NIXL_BACKEND_BACKEND_PLUGIN_H

#include "MLRT/Driver/NIXL/backend/backend_engine.h"

// Define the plugin API version
#define NIXL_PLUGIN_API_VERSION 1

namespace M::AsyncRT::NIXL {

// Define the plugin interface class
class nixlBackendPlugin {
public:
  int api_version;

  // Function pointer for creating a new backend engine instance
  nixlBackendEngine *(*create_engine)(const nixlBackendInitParams *init_params);

  // Function pointer for destroying a backend engine instance
  void (*destroy_engine)(nixlBackendEngine *engine);

  // Function to get the plugin name
  const char *(*get_plugin_name)();

  // Function to get the plugin version
  const char *(*get_plugin_version)();

  // Function to get backend options
  nixl_b_params_t (*get_backend_options)();

  // Function to get supported backend mem types
  nixl_mem_list_t (*get_backend_mems)();
};

// Macro to define exported C functions for the plugin
#define NIXL_PLUGIN_EXPORT __attribute__((visibility("default")))

// Plugin must implement these functions for dynamic loading
extern "C" {
// Initialize the plugin
NIXL_PLUGIN_EXPORT nixlBackendPlugin *nixl_plugin_init();

// Cleanup the plugin
NIXL_PLUGIN_EXPORT void nixl_plugin_fini();
}

} // namespace M::AsyncRT::NIXL

#endif // MLRT_DRIVER_NIXL_BACKEND_BACKEND_PLUGIN_H
