//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

// Adapted by Modular.
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#include "MLRT/AsyncRT/NIXL/agent_data.h"
#include "MLRT/AsyncRT/NIXL/nixl.h"
#include "MLRT/AsyncRT/NIXL/nixl_types.h"
#include "MLRT/AsyncRT/NIXL/utils/common/nixl_log.h"
#include "MLRT/AsyncRT/NIXL/utils/common/nixl_time.h"
#include "MLRT/AsyncRT/NIXL/utils/common/str_tools.h"
#include "llvm/Support/ErrorHandling.h"
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <poll.h>
#include <string_view>
#include <sys/uio.h>

using namespace M::AsyncRT::NIXL;
using namespace std::string_view_literals;

// fix SOCK_NONBLOCK for e.g. macOS
#ifndef SOCK_NONBLOCK
#define SOCK_NONBLOCK O_NONBLOCK
#endif

static int connectToIP(const std::string &ip_addr, int port) {

  struct sockaddr_in listenerAddr;
  listenerAddr.sin_port = htons(port);
  listenerAddr.sin_family = AF_INET;

  if (inet_pton(AF_INET, ip_addr.c_str(), &listenerAddr.sin_addr) <= 0) {
    NIXL_ERROR << "inet_pton failed for ip_addr: " << ip_addr;
    return -1;
  }

  // Create a non-blocking socket
  int ret_fd = socket(AF_INET, SOCK_STREAM | SOCK_NONBLOCK, 0);
  if (ret_fd == -1) {
    NIXL_ERROR << "socket creation failed for ip_addr: " << ip_addr
               << " and port: " << port;
    return -1;
  }

  // Connect will return immediately with EINPROGRESS
  int ret =
      connect(ret_fd, (struct sockaddr *)&listenerAddr, sizeof(listenerAddr));
  if (ret < 0 && errno != EINPROGRESS) {
    close(ret_fd);
    return -1;
  }

  // Use poll to wait for connection with timeout
  struct pollfd pfd;
  pfd.fd = ret_fd;
  pfd.events = POLLOUT;
  pfd.revents = 0;

  ret = poll(&pfd, 1, 1000);
  if (ret <= 0) {
    if (ret < 0) {
      NIXL_ERROR << "poll failed for ip_addr: " << ip_addr
                 << " and port: " << port << " with error: " << strerror(errno);
    } else {
      NIXL_ERROR << "poll timed out for ip_addr: " << ip_addr
                 << " and port: " << port;
    }
    close(ret_fd);
    return -1;
  }

  // Check if connection was successful
  int error = 0;
  socklen_t len = sizeof(error);
  if (getsockopt(ret_fd, SOL_SOCKET, SO_ERROR, &error, &len) < 0 ||
      error != 0) {
    NIXL_ERROR << "getsockopt failed for ip_addr: " << ip_addr
               << " and port: " << port << " with error: " << strerror(error);
    close(ret_fd);
    return -1;
  }

  return ret_fd;
}

static void sendCommMessage(int fd, std::string_view msg) {
  size_t size = msg.size();
  constexpr size_t iov_size = 2;
  struct iovec iov[iov_size] = {{&size, sizeof(size)},
                                {const_cast<char *>(msg.data()), msg.size()}};

  for (size_t i = 0, offset = 0, sent = 0; i < iov_size;) {
    auto bytes = send(fd, static_cast<char *>(iov[i].iov_base) + offset,
                      iov[i].iov_len - offset, 0);
    if (bytes < 0) {
      if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK) {
        continue;
      }

      NIXL_ERROR << "sendCommMessage(fd=" << fd << ") " << sent << "/"
                 << (size + sizeof(size)) << " bytes failed, errno=" << errno;
      return;
    }

    offset += bytes;
    sent += bytes;
    if (offset == iov[i].iov_len) {
      offset = 0;
      ++i;
    }
  }
}

static bool recvCommMessageType(int fd, void *data, size_t size,
                                bool force = false) {
  for (size_t received = 0; received < size;) {
    auto bytes =
        recv(fd, static_cast<char *>(data) + received, size - received, 0);
    if (bytes > 0) {
      received += bytes;
      continue;
    }
    if (bytes == 0 && received == 0 && !force) {
      return false;
    }

    if (bytes < 0) {
      if (errno == EINTR) {
        continue;
      }

      if (errno == EAGAIN || errno == EWOULDBLOCK) {
        if (!force && received == 0) {
          return false; // nothing to read yet
        }
        continue;
      }
    }

    NIXL_ERROR << "recvCommMessage(fd=" << fd << ") " << received << "/" << size
               << " bytes failed ret=" << bytes << " errno=" << errno;
    return false;
  }

  return true;
}

static bool recvCommMessage(int fd, std::string &msg) {
  size_t size;
  if (!recvCommMessageType(fd, &size, sizeof(size))) {
    return false;
  }

  msg.resize(size);
  return recvCommMessageType(fd, msg.data(), size, true);
}

void nixlAgentData::commWorker(nixlAgent *myAgent) {
  NIXL_INFO << "Listener thread started for agent: " << name
            << " (useListenThread=" << config.useListenThread << ")";

  while (!(commThreadStop)) {
    std::vector<nixl_comm_req_t> work_queue;

    // first, accept new connections
    int new_fd = 0;

    while (new_fd != -1 && config.useListenThread) {
      new_fd = listener->acceptClient();
      nixl_socket_peer_t accepted_client;

      if (new_fd != -1) {
        // need to convert fd to IP address and add to client map
        sockaddr_in client_address;
        socklen_t client_addrlen = sizeof(client_address);
        if (getpeername(new_fd, (sockaddr *)&client_address, &client_addrlen) ==
            0) {
          char client_ip[INET_ADDRSTRLEN];
          inet_ntop(AF_INET, &client_address.sin_addr, client_ip,
                    INET_ADDRSTRLEN);
          accepted_client.first = std::string(client_ip);
          accepted_client.second = client_address.sin_port;
        } else {
          // TODO(GENAI-170): return ErrorOr here.
          NIXL_FATAL << "getpeername failed\n";
          std::abort();
        }
        remoteSockets[accepted_client] = new_fd;

        // make new socket nonblocking
        int new_flags = fcntl(new_fd, F_GETFL, 0) | O_NONBLOCK;

        if (fcntl(new_fd, F_SETFL, new_flags) == -1) {
          // TODO(GENAI-170): return ErrorOr here.
          NIXL_FATAL << "fcntl accept failed\n";
          std::abort();
        }
      }
    }

    // second, do agent commands
    getCommWork(work_queue);

    // TODO(GENAI-170): Clean up with future IP type.
    for (const auto &[req_command, req_ip, req_port, my_MD] : work_queue) {
      nixl_socket_peer_t req_sock = std::make_pair(req_ip, req_port);

      // use remote IP for socket lookup
      auto client = remoteSockets.find(req_sock);
      int client_fd;

      // Create connection for all socket commands if not already connected
      if (req_command < nixl_comm_t::SOCK_MAX) {
        if (client == remoteSockets.end()) {
          int new_client = connectToIP(req_ip, req_port);
          if (new_client == -1) {
            NIXL_ERROR << "Listener thread could not connect to IP " << req_ip
                       << " and port " << req_port << "\n";
            continue;
          }
          remoteSockets[req_sock] = new_client;
          client_fd = new_client;
        } else {
          client_fd = client->second;
        }
      }

      switch (req_command) {
      case nixl_comm_t::SOCK_SEND: {
        sendCommMessage(client_fd, ("NIXLCOMM:LOAD" + Twine(my_MD)).str());
        break;
      }
      case nixl_comm_t::SOCK_FETCH: {
        sendCommMessage(client_fd, "NIXLCOMM:SEND"sv);
        break;
      }
      case nixl_comm_t::SOCK_INVAL: {
        sendCommMessage(client_fd, ("NIXLCOMM:INVL"sv + Twine(name)).str());
        break;
      }
      case nixl_comm_t::SOCK_MAX: {
        // Invalid command, should not reach here
        llvm_unreachable("invalid NIXL request command");
      }
      }
    }

    // third, do remote commands
    auto socket_iter = remoteSockets.begin();
    while (socket_iter != remoteSockets.end()) {
      std::string commands;
      if (!recvCommMessage(socket_iter->second, commands)) {
        ++socket_iter;
        continue;
      }

      std::vector<std::string_view> command_list =
          str_split_substr(commands, "NIXLCOMM:"sv);

      for (std::string_view command : command_list) {

        if (command.size() < 4)
          continue;

        // always just 4 chars:
        std::string_view header = command.substr(0, 4);

        if (header == "LOAD"sv) {
          // Everything past the header
          const blob_ref remoteMetaData{command.substr(4)};
          std::string remote_agent;
          nixl_status_t ret =
              myAgent->loadRemoteMD(remoteMetaData, remote_agent);
          if (ret != nixl_status_t::NIXL_SUCCESS) {
            NIXL_ERROR
                << "loadRemoteMD in listener thread failed for md from peer "
                << socket_iter->first.first << ":" << socket_iter->first.second
                << " with error " << to_underlying(ret);
            continue;
          }
          // not sure what to do with remote_agent
        } else if (header == "SEND"sv) {
          nixl_blob_t my_MD;
          myAgent->getLocalMD(my_MD);

          sendCommMessage(socket_iter->second, "NIXLCOMM:LOAD" + my_MD);
        } else if (header == "INVL") {
          std::string_view remote_agent = command.substr(4);
          myAgent->invalidateRemoteMD(std::string(remote_agent));
          break;
        } else {
          NIXL_ERROR << "Received socket message with bad header " << header
                     << " from peer " << socket_iter->first.first << ":"
                     << socket_iter->first.second;
        }
      }

      ++socket_iter;
    }

    nixlTime::us_t start = nixlTime::getUs();
    while ((start + config.lthrDelay) > nixlTime::getUs()) {
      std::this_thread::yield();
    }
  }
}

void nixlAgentData::enqueueCommWork(nixl_comm_req_t request) {
  std::lock_guard<std::mutex> lock(commLock);
  commQueue.push_back(std::move(request));
}

void nixlAgentData::getCommWork(std::vector<nixl_comm_req_t> &req_list) {
  std::lock_guard<std::mutex> lock(commLock);
  req_list = std::move(commQueue);
  commQueue.clear();
}
