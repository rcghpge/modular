//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "MLRT/AsyncRT/ZeroMQ/zmq.hpp"

#include <iostream>

int main() {
  auto contextOr = zmq::context_t::create(1);
  if (contextOr.isError()) {
    std::cerr << "Failed to create context: " << contextOr.getError()
              << std::endl;
    return 1;
  }
  auto &context = contextOr.get();

  auto socketOr = zmq::socket_t::create(context, zmq::socket_type::req);
  if (socketOr.isError()) {
    std::cerr << "Failed to create socket: " << socketOr.getError()
              << std::endl;
    return 1;
  }
  auto &socket = socketOr.get();

  auto connectResult = socket.connect("tcp://localhost:5555");
  if (connectResult.isError()) {
    std::cerr << "Failed to connect socket: " << connectResult.getError()
              << std::endl;
    return 1;
  }

  auto requestOr = zmq::message_t::create(5);
  if (requestOr.isError()) {
    std::cerr << "Failed to create message: " << requestOr.getError()
              << std::endl;
    return 1;
  }
  auto &request = requestOr.get();

  std::memcpy(request.data(), "Hello", 5);
  if (auto sendResult = socket.send(request, zmq::send_flags::none);
      sendResult.isError()) {
    std::cerr << "Failed to send message: " << sendResult.getError()
              << std::endl;
    return 1;
  }
  zmq::message_t reply;

  if (auto recvResult = socket.recv(reply, zmq::recv_flags::none);
      recvResult.isError()) {
    std::cerr << "Failed to receive message: " << recvResult.getError()
              << std::endl;
    return 1;
  }

  std::cout << "Received reply: "
            << std::string(static_cast<char *>(reply.data()), reply.size())
            << std::endl;
  return 0;
}
