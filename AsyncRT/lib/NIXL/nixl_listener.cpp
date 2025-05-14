//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

// Adapted by Modular.
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#include "AsyncRT/NIXL/agent_data.h"
#include "AsyncRT/NIXL/nixl.h"
#include "AsyncRT/NIXL/nixl_types.h"
#include "AsyncRT/NIXL/utils/common/nixl_log.h"
#include "AsyncRT/NIXL/utils/common/nixl_time.h"
#include "AsyncRT/NIXL/utils/common/str_tools.h"
#include "llvm/Support/ErrorHandling.h"
#include <fcntl.h>
#include <iostream>
#include <string_view>

using namespace M::AsyncRT::NIXL;
using namespace std::string_view_literals;

static int connectToIP(const std::string &ip_addr, int port) {

  struct sockaddr_in listenerAddr;
  listenerAddr.sin_port = htons(port);
  listenerAddr.sin_family = AF_INET;

  int ret_fd = socket(AF_INET, SOCK_STREAM | SOCK_NONBLOCK, 0);
  if (ret_fd == -1) {
    return -1;
  }

  if (inet_pton(AF_INET, ip_addr.c_str(), &listenerAddr.sin_addr) <= 0) {
    close(ret_fd);
    return -1;
  }

  // make connect block for now to avoid ambiguity in send right after
  int orig_flags = fcntl(ret_fd, F_GETFL, 0);
  int new_flags = orig_flags ^ O_NONBLOCK;

  fcntl(ret_fd, F_SETFL, new_flags);

  if (connect(ret_fd, (struct sockaddr *)&listenerAddr, sizeof(listenerAddr)) <
      0) {
    perror("async connect");
    close(ret_fd);
    return -1;
  }

  // make nonblocking again
  fcntl(ret_fd, F_SETFL, orig_flags);

  return ret_fd;
}

static size_t sendCommMessage(int fd, std::string_view msg) {
  size_t bytes;
  bytes = send(fd, msg.data(), msg.size(), 0);
  if (bytes < 0)
    std::cerr << "Cannot send on socket to fd " << fd << "\n";

  return bytes;
}

static std::pair<std::string, ssize_t> recvCommMessage(int fd) {
  char buffer[16384];
  ssize_t one_recv_bytes = 0;
  ssize_t recv_bytes = 0;
  std::string msg;

  do {
    one_recv_bytes = recv(fd, buffer, sizeof(buffer), 0);
    if (one_recv_bytes == -1) {
      if (errno == EAGAIN || errno == EWOULDBLOCK)
        return {msg, recv_bytes};
      // TODO(GENAI-170): Use logging utils.
      std::cerr << "Cannot recv on socket fd " << fd << "\n";
      return {msg, one_recv_bytes};
    }
    msg.append(buffer, one_recv_bytes);
    recv_bytes += one_recv_bytes;
  } while (one_recv_bytes > 0);

  return {msg, recv_bytes};
}

void nixlAgentData::commWorker(nixlAgent *myAgent) {
  while (!(commThreadStop)) {
    std::vector<nixl_comm_req_t> work_queue;

    // first, accept new connections
    int new_fd = 0;

    while (new_fd != -1 && config.useListenThread && !useEtcd) {
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

    for (nixl_comm_req_t request : work_queue) {
      // TODO(GENAI-170): Clean up with future IP type.
      nixl_comm_t req_command = std::get<0>(request);
      std::string req_ip = std::get<1>(request);
      int req_port = std::get<2>(request);
      std::string_view my_MD = std::get<3>(request);

      nixl_socket_peer_t req_sock = std::make_pair(req_ip, req_port);

      // use remote IP for socket lookup
      auto client = remoteSockets.find(req_sock);
      int client_fd;

      switch (req_command) {
      case nixl_comm_t::SOCK_SEND: {
        // not connected
        if (client == remoteSockets.end()) {
          int new_client = connectToIP(req_ip, req_port);
          if (new_client == -1) {
            NIXL_ERROR << "Listener thread could not connect to IP " << req_ip
                       << " and port " << req_port << "\n";
            break;
          }
          remoteSockets[req_sock] = new_client;
          client_fd = new_client;
        } else {
          client_fd = client->second;
        }

        sendCommMessage(client_fd, ("NIXLCOMM:LOAD" + Twine(my_MD)).str());
        break;
      }
      case nixl_comm_t::SOCK_FETCH: {
        if (client == remoteSockets.end()) {
          int new_client = connectToIP(req_ip, req_port);
          if (new_client == -1) {
            NIXL_ERROR << "Listener thread could not connect to IP " << req_ip;
            break;
          }
          remoteSockets[req_sock] = new_client;
          client_fd = new_client;
        } else {
          client_fd = client->second;
        }

        sendCommMessage(client_fd, "NIXLCOMM:SEND"sv);
        break;
      }
      case nixl_comm_t::SOCK_INVAL: {
        if (client == remoteSockets.end()) {
          // TODO(GENAI-170): return ErrorOr here.
          NIXL_FATAL << "invalidate on closed socket\n";
          std::abort();
        }
        client_fd = client->second;
        sendCommMessage(client_fd, ("NIXLCOMM:INVL"sv + Twine(name)).str());
        break;
      }
      }
      llvm_unreachable("invalid NIXL request command");
    }

    // third, do remote commands
    auto socket_iter = remoteSockets.begin();
    while (socket_iter != remoteSockets.end()) {
      auto [commands, recv_bytes] = recvCommMessage(socket_iter->second);
      if (recv_bytes == 0 || recv_bytes == -1) {
        socket_iter++;
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
          std::string_view remote_md = command.substr(4);
          std::string remote_agent;
          nixl_status_t ret =
              myAgent->loadRemoteMD(std::string(remote_md), remote_agent);
          if (ret != nixl_status_t::NIXL_SUCCESS) {
            // TODO(GENAI-170): return ErrorOr here.
            NIXL_FATAL << "loadRemoteMD in listener thread failed, critically "
                          "failing\n";
            std::abort();
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
          // TODO(GENAI-170): return ErrorOr here.
          NIXL_FATAL << "Received socket message with bad header" << header
                     << ", critically failing\n";
          std::abort();
        }
      }

      socket_iter++;
    }

    nixlTime::us_t start = nixlTime::getUs();
    while ((start + config.lthrDelay) > nixlTime::getUs()) {
      std::this_thread::yield();
    }
  }

  // Close remaining connections
  for (auto &[remote, fd] : remoteSockets) {
    shutdown(fd, SHUT_RDWR);
    close(fd);
  }
}

void nixlAgentData::enqueueCommWork(
    std::tuple<nixl_comm_t, std::string, int, std::string> request) {
  std::lock_guard<std::mutex> lock(commLock);
  commQueue.push_back(request);
}

void nixlAgentData::getCommWork(std::vector<nixl_comm_req_t> &req_list) {
  std::lock_guard<std::mutex> lock(commLock);
  req_list = commQueue;
  commQueue.clear();
}
