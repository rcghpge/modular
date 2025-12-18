//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

// Adapted by Modular.
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#ifndef ASYNCRT_NIXL_UTILS_STREAM_METADATA_STREAM_H
#define ASYNCRT_NIXL_UTILS_STREAM_METADATA_STREAM_H

#include "MLRT/AsyncRT/NIXL/nixl_types.h"
#include <arpa/inet.h>
#include <cstring>
#include <netinet/in.h>
#include <string>
#include <sys/un.h>
#include <thread>
#include <unistd.h>

constexpr int RECV_BUFFER_SIZE = 16384;

namespace M::AsyncRT::NIXL {

class nixlMetadataStream {
protected:
  int port;
  int socketFd;
  // TODO(GENAI-170): Replace with an IP address type.
  std::string listenerAddress;
  sockaddr_in listenerAddr;

  bool setupStream();
  void closeStream();

public:
  nixlMetadataStream(int port);
  ~nixlMetadataStream();

  nixlMetadataStream(const nixlMetadataStream &) = delete;
  nixlMetadataStream &operator=(const nixlMetadataStream &) = delete;
  nixlMetadataStream(nixlMetadataStream &&) = delete;
  nixlMetadataStream &operator=(nixlMetadataStream &&) = delete;
};

class nixlMDStreamListener : public nixlMetadataStream {
private:
  std::thread listenerThread;
  int csock;

  void acceptClientsAsync();
  void recvFromClients(int clientSocket);

public:
  nixlMDStreamListener(int port);
  ~nixlMDStreamListener();

  nixlMDStreamListener(const nixlMDStreamListener &) = delete;
  nixlMDStreamListener &operator=(const nixlMDStreamListener &) = delete;
  nixlMDStreamListener(nixlMDStreamListener &&) = delete;
  nixlMDStreamListener &operator=(nixlMDStreamListener &&) = delete;

  int acceptClient();
  void setupListener();
  void startListenerForClients();
  void startListenerForClient();
  std::string recvFromClient();
};

class nixlMDStreamClient : public nixlMetadataStream {
private:
  std::string listenerAddress;
  bool setupClient();

public:
  nixlMDStreamClient(const std::string &listenerAddress, int port);
  ~nixlMDStreamClient();

  nixlMDStreamClient(const nixlMDStreamClient &) = delete;
  nixlMDStreamClient &operator=(const nixlMDStreamClient &) = delete;
  nixlMDStreamClient(nixlMDStreamClient &&) = delete;
  nixlMDStreamClient &operator=(nixlMDStreamClient &&) = delete;

  bool connectListener();
  void sendData(const blob_ref data);
  std::string recvData();
};

} // namespace M::AsyncRT::NIXL

#endif // ASYNCRT_NIXL_UTILS_STREAM_METADATA_STREAM_H
