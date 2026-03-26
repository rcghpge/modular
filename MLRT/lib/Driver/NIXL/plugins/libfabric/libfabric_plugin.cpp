//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

// Adapted by Modular.
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// SPDX-FileCopyrightText: Copyright (c) 2025 Amazon.com, Inc. and affiliates.
// SPDX-License-Identifier: Apache-2.0

#include "MLRT/Driver/NIXL/backend/backend_plugin.h"
#include "MLRT/Driver/NIXL/plugins/libfabric/libfabric_backend.h"

// Plugin type alias for convenience
namespace M::AsyncRT::NIXL {
using enum nixl_mem_t;
using enum nixl_status_t;
using enum nixl_xfer_op_t;

static nixlBackendEngine *
createLibfabricEngine(const nixlBackendInitParams *init_params) {
  return new nixlLibfabricEngine(init_params);
}

static void destroyLibfabricEngine(nixlBackendEngine *engine) { delete engine; }

static const char *getLibfabricPluginName() { return "LIBFABRIC"; }

static const char *getLibfabricPluginVersion() { return "0.1.0"; }

static nixl_b_params_t getLibfabricBackendOptions() { return {}; }

static nixl_mem_list_t getLibfabricBackendMems() {
  // GPU memory support is determined by the runtime DeviceContext-backed
  // engine, so the plugin advertises the conservative baseline here.
  return {DRAM_SEG};
}

static nixlBackendPlugin *getLibfabricPlugin() {
  static nixlBackendPlugin plugin = {
      .api_version = NIXL_PLUGIN_API_VERSION,
      .create_engine = createLibfabricEngine,
      .destroy_engine = destroyLibfabricEngine,
      .get_plugin_name = getLibfabricPluginName,
      .get_plugin_version = getLibfabricPluginVersion,
      .get_backend_options = getLibfabricBackendOptions,
      .get_backend_mems = getLibfabricBackendMems,
  };
  return &plugin;
}

extern "C" NIXL_PLUGIN_EXPORT nixlBackendPlugin *nixl_plugin_init() {
  return getLibfabricPlugin();
}

extern "C" NIXL_PLUGIN_EXPORT void nixl_plugin_fini() {}

} // namespace M::AsyncRT::NIXL
