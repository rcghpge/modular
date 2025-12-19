//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

// Adapted by Modular.
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#include "MLRT/Driver/NIXL/utils/stream/metadata_stream.h"
#include "MLRT/Driver/NIXL/utils/common/nixl_log.h"
#include <arpa/inet.h>
#include <cstring>
#include <iostream>
#include <netinet/in.h>
#include <string_view>
#include <sys/socket.h>
#include <unistd.h>

using namespace M::AsyncRT::NIXL;
using namespace std::string_view_literals;

// fix SOCK_NONBLOCK for e.g. macOS
#ifndef SOCK_NONBLOCK
#include <fcntl.h>
#define SOCK_NONBLOCK O_NONBLOCK
#endif

nixlMetadataStream::nixlMetadataStream(int port) : port(port), socketFd(-1) {
  memset(&listenerAddr, 0, sizeof(listenerAddr));
}

nixlMetadataStream::~nixlMetadataStream() { closeStream(); }

bool nixlMetadataStream::setupStream() {
  socketFd = socket(AF_INET, SOCK_STREAM | SOCK_NONBLOCK, 0);
  if (socketFd == -1) {
    NIXL_PERROR << "failed to create stream socket for listener";
    return false;
  }

  listenerAddr.sin_family = AF_INET;
  listenerAddr.sin_addr.s_addr = INADDR_ANY;
  listenerAddr.sin_port = htons(port);

  return true;
}

void nixlMetadataStream::closeStream() {
  if (socketFd != -1) {
    close(socketFd);
  }
}

nixlMDStreamListener::nixlMDStreamListener(int port)
    : nixlMetadataStream(port) {}

nixlMDStreamListener::~nixlMDStreamListener() {
  if (listenerThread.joinable()) {
    listenerThread.join();
  }
  if (csock >= 0) {
    close(csock);
  }
}

void nixlMDStreamListener::setupListener() {
  if (!setupStream()) {
    // TODO(GENAI-170): Error propagation.
    return;
  }

  if (bind(socketFd, (sockaddr *)&listenerAddr, sizeof(listenerAddr)) < 0) {
    NIXL_PERROR << "Socket Bind failed while setting up listener for MD";
    closeStream();
    return;
  }

  if (listen(socketFd, 128) < 0) {
    NIXL_PERROR << "Listening failed for stream Socket: " << socketFd;
    closeStream();
    return;
  }
  NIXL_DEBUG << "MD listener is listening on port " << port << "...";
}

int nixlMDStreamListener::acceptClient() {
  csock = accept(socketFd, nullptr, nullptr);
  if (csock < 0 && errno != EAGAIN) {
    NIXL_PERROR << "Cannot accept client connection";
  }
  return csock;
}

void nixlMDStreamListener::acceptClientsAsync() {
  while (true) {
    int clientSocket = accept(socketFd, nullptr, nullptr);
    if (clientSocket < 0) {
      NIXL_PERROR << "Cannot accept client connection";
      continue;
    }
    NIXL_DEBUG << "Client connected.";
    std::thread clientThread(&nixlMDStreamListener::recvFromClients, this,
                             clientSocket);
    clientThread.detach();
  }
}

std::string nixlMDStreamListener::recvFromClient() {
  char buffer[RECV_BUFFER_SIZE];
  int bytes_read;
  std::string recvData;

  bytes_read = recv(csock, buffer, sizeof(buffer), 0);

  if (bytes_read > 0) {
    recvData = std::string(buffer, bytes_read);
  } else if (bytes_read == 0) {
    NIXL_DEBUG << "Client disconnected";
  } else {
    NIXL_ERROR << "Error receiving data";
  }
  return recvData;
}

void nixlMDStreamListener::recvFromClients(int clientSocket) {
  char buffer[RECV_BUFFER_SIZE];
  int bytes_read;

  while ((bytes_read = recv(clientSocket, buffer, sizeof(buffer), 0)) > 0) {
    buffer[bytes_read] = '\0';
    // Return ack
    std::string_view ack = "Message received"sv;
    send(clientSocket, ack.data(), ack.size(), 0);
    std::string_view recvMessage(buffer);
    NIXL_DEBUG << "Message Received" << recvMessage;
  }
  close(clientSocket);
  NIXL_DEBUG << "Client disconnected";
}

void nixlMDStreamListener::startListenerForClient() {
  setupListener();
  acceptClient();
}

void nixlMDStreamListener::startListenerForClients() {
  setupListener();
  listenerThread = std::thread(&nixlMDStreamListener::acceptClientsAsync, this);
}

nixlMDStreamClient::nixlMDStreamClient(const std::string &listenerAddress,
                                       int port)
    : nixlMetadataStream(port), listenerAddress(listenerAddress) {}

nixlMDStreamClient::~nixlMDStreamClient() { closeStream(); }

bool nixlMDStreamClient::setupClient() {
  setupStream();

  sockaddr_in listenerAddr;
  listenerAddr.sin_family = AF_INET;
  listenerAddr.sin_port = htons(port);

  if (inet_pton(AF_INET, listenerAddress.c_str(), &listenerAddr.sin_addr) <=
      0) {
    perror("Invalid address/ Address not supported");
    return false;
  }

  if (connect(socketFd, (sockaddr *)&listenerAddr, sizeof(listenerAddr)) < 0) {
    NIXL_PERROR << "Connection Failed";
    closeStream();
    return false;
  }
  NIXL_DEBUG << "Connected to listener at " << listenerAddress << ":" << port;
  return true;
}

bool nixlMDStreamClient::connectListener() { return setupClient(); }

void nixlMDStreamClient::sendData(const blob_ref data) {
  // TODO(GENAI-170): Fix logging and indicate error another way.
  if (send(socketFd, data.data, data.size, 0) < 0)
    NIXL_ERROR << "Send failed";
}

std::string nixlMDStreamClient::recvData() {
  char buffer[RECV_BUFFER_SIZE];
  int bytes_read = recv(socketFd, buffer, sizeof(buffer), 0);
  if (bytes_read > 0) {
    buffer[bytes_read] = '\0';
    return std::string(buffer);
  }
  return "";
}
