//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

// Adapted by Modular.
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#ifndef MLRT_DRIVER_NIXL_NIXL_PARAMS_H
#define MLRT_DRIVER_NIXL_NIXL_PARAMS_H

#include "nixl_types.h"
#include <cstdint>

namespace M::AsyncRT::NIXL {

/**
 * @class nixlAgentConfig
 * @brief Per Agent configuration information, such as if progress thread should
 * be used. Other configs such as assigned IP/port or device access can be
 * added.
 */
struct nixlAgentConfig {
  /** @var Enable progress thread */
  bool useProgThread;
  /** @var Enable listener thread */
  bool useListenThread;
  /** @var Port for listener thread to use */
  int listenPort;
  /** @var synchronization mode for multi-threaded environment execution */
  nixl_thread_sync_t syncMode;

  /**
   * @var Progress thread event waiting timeout.
   *      Defines a delay between periodic main loop iterations that progress
   * every worker unconditionally of any pending event signals.
   */
  uint64_t pthrDelay;
  /**
   * @var Listener thread frequency knob (in us)
   *      Listener thread sleeps in a similar way to progress thread, desrcibed
   * previously. These will be combined into a unified NIXL Thread API in a
   * future version.
   */
  uint64_t lthrDelay;

  /**
   * @brief  Agent configuration constructor for enabling various features.
   * @param use_prog_thread    flag to determine use of progress thread
   * @param use_listen_thread  flag to determine use of listener thread
   * @param port               specify port for listener thread to listen on
   * @param sync_mode          Thread synchronization mode
   * @param pthr_delay_us      Optional delay for pthread in us
   * @param lthr_delay_us      Optional delay for listener thread in us
   */
  nixlAgentConfig(const bool use_prog_thread,
                  const bool use_listen_thread = false, const int port = 0,
                  nixl_thread_sync_t sync_mode =
                      nixl_thread_sync_t::NIXL_THREAD_SYNC_DEFAULT,
                  const uint64_t pthr_delay_us = 0,
                  const uint64_t lthr_delay_us = 100000)
      : useProgThread(use_prog_thread), useListenThread(use_listen_thread),
        listenPort(port), syncMode(sync_mode), pthrDelay(pthr_delay_us),
        lthrDelay(lthr_delay_us) {}

  friend class nixlAgent;
  friend class nixlAgentData;
};

} // namespace M::AsyncRT::NIXL

#endif // MLRT_DRIVER_NIXL_NIXL_PARAMS_H
