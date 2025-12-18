//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

// Adapted by Modular.
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#include "MLRT/AsyncRT/NIXL/backend/backend_plugin.h"
#include "MLRT/AsyncRT/NIXL/plugins/ucx/ucx_backend.h"

using namespace M::AsyncRT;
using namespace M::AsyncRT::NIXL;

// Plugin version information
static const char *PLUGIN_NAME = "UCX";
static const char *PLUGIN_VERSION = "0.1.0";

// Function to create a new UCX backend engine instance
static nixlBackendEngine *
create_ucx_engine(const nixlBackendInitParams *init_params) {
  return nixlUcxEngine::create(*init_params).release();
}

static void destroy_ucx_engine(nixlBackendEngine *engine) { delete engine; }

// Function to get the plugin name
static const char *get_plugin_name() { return PLUGIN_NAME; }

// Function to get the plugin version
static const char *get_plugin_version() { return PLUGIN_VERSION; }

// Function to get backend options
static nixl_b_params_t get_backend_options() {
  nixl_b_params_t params;
  params["ucx_devices"] = "";
  params["ucx_error_handling_mode"] = "none"; // or "peer"
  return params;
}

// Function to get supported backend mem types
static nixl_mem_list_t get_backend_mems() {
  nixl_mem_list_t mems;
  mems.push_back(nixl_mem_t::DRAM_SEG);
  mems.push_back(nixl_mem_t::VRAM_SEG);
  return mems;
}

// Static plugin structure
static nixlBackendPlugin plugin = {NIXL_PLUGIN_API_VERSION, create_ucx_engine,
                                   destroy_ucx_engine,      get_plugin_name,
                                   get_plugin_version,      get_backend_options,
                                   get_backend_mems};

#ifdef STATIC_PLUGIN_UCX

nixlBackendPlugin *NIXL::createStaticUcxPlugin() {
  return &plugin; // Return the static plugin instance
}

#else

// Plugin initialization function
extern "C" NIXL_PLUGIN_EXPORT nixlBackendPlugin *nixl_plugin_init() {
  return &plugin;
}

// Plugin cleanup function
extern "C" NIXL_PLUGIN_EXPORT void nixl_plugin_fini() {
  // Cleanup any resources if needed
}
#endif
