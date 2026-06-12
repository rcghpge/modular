//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "MLRT/Driver/NIXL/nixl.h"
#include "MLRT/Driver/NIXL/utils/libfabric/libfabric_common.h"

#include <algorithm>
#include <arpa/inet.h>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <netdb.h>
#include <netinet/in.h>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {

using namespace M::MLRT;
using namespace M::Driver;

constexpr std::string_view kBackend = "libfabric";
constexpr std::string_view kInitiator = "initiator";
constexpr std::string_view kTarget = "target";
constexpr uint8_t kTransferPattern = 0xab;
constexpr auto kPollSleep = std::chrono::microseconds(100);
constexpr auto kTargetVerifyDelay = std::chrono::seconds(30);
constexpr size_t kWarmupIterations = 1;
constexpr int kRetryLimit = 400;
constexpr std::string_view kPluginFileName = "libplugin_libfabric.so";

struct TargetOptions {
  int listenPort = 0;
  int gpu = 0;
  size_t bytes = 64ULL * 1024 * 1024;
};

struct InitiatorOptions {
  std::string targetHost;
  int targetPort = 0;
  int gpu = 0;
  size_t bytes = 64ULL * 1024 * 1024;
  size_t iterations = 1;
  // Number of transfer requests to post before waiting for any to complete.
  // Default 1 = the original sequential behaviour. Larger values force many
  // concurrent fi_writedata posts per rail, used to stress the EFA TX send
  // queue and probe the -FI_EAGAIN behaviour described in GEX-3539.
  size_t burst = 1;
  // Number of descriptors to pack into each transfer request. Default 1 is
  // the legacy single-desc behaviour. Setting this > 1 mimics the real KV
  // cache transfer shape where one postXferReq fans out to many per-page
  // fi_writedata calls through striping / round-robin.
  size_t descsPerXfer = 1;
};

struct TargetInfo {
  nixl_blob_t metadata;
  uint64_t baseAddr = 0;
  uint64_t bytes = 0;
  uint64_t deviceId = 0;
};

struct RegisteredVramBuffer {
  DeviceContextRef context;
  DeviceBufferRef buffer;
  nixl_reg_dlist_t regDescs;
  nixl_xfer_dlist_t xferDescs;
};

[[noreturn]] void fail(std::string_view message) {
  std::cerr << "error: " << message << std::endl;
  std::exit(EXIT_FAILURE);
}

[[noreturn]] void failWithDetail(std::string_view message,
                                 std::string_view detail) {
  std::cerr << "error: " << message;
  if (!detail.empty()) {
    std::cerr << ": " << detail;
  }
  std::cerr << std::endl;
  std::exit(EXIT_FAILURE);
}

template <typename T>
T takeValueOrDie(M::ErrorOr<T> valueOrError, std::string_view message) {
  if (valueOrError.isError()) {
    failWithDetail(message, valueOrError.getError());
  }
  return valueOrError.takeValue();
}

void checkSuccess(M::ErrorOrSuccess status, std::string_view message) {
  if (status.isError()) {
    failWithDetail(message, status.getError());
  }
}

void checkStatus(nixl_status_t status, std::string_view message) {
  if (status != nixl_status_t::NIXL_SUCCESS) {
    failWithDetail(message, NixlEnumStrings::statusStr(status));
  }
}

void checkNonNegativeStatus(nixl_status_t status, std::string_view message) {
  if (isErrorStatus(status)) {
    failWithDetail(message, NixlEnumStrings::statusStr(status));
  }
}

[[noreturn]] void failFromErrno(std::string_view message) {
  failWithDetail(message, std::strerror(errno));
}

size_t parseSize(std::string_view value) {
  errno = 0;
  char *end = nullptr;
  const std::string owned(value);
  const unsigned long long parsed = std::strtoull(owned.c_str(), &end, 10);
  if (errno != 0 || end == owned.c_str() || *end != '\0') {
    fail("invalid size value");
  }
  return static_cast<size_t>(parsed);
}

int parseInt(std::string_view value) {
  errno = 0;
  char *end = nullptr;
  const std::string owned(value);
  const long parsed = std::strtol(owned.c_str(), &end, 10);
  if (errno != 0 || end == owned.c_str() || *end != '\0') {
    fail("invalid integer value");
  }
  return static_cast<int>(parsed);
}

int parsePort(std::string_view value) {
  const int parsed = parseInt(value);
  if (parsed <= 0 || parsed > 65535) {
    fail("invalid port value");
  }
  return parsed;
}

void printUsage(const char *argv0) {
  std::cerr << "Usage:\n"
            << "  " << argv0 << " target <listen-port> [gpu] [bytes]\n"
            << "  " << argv0
            << " initiator <target-host> <target-port> [gpu] [bytes] [iters] "
               "[burst] [descs-per-xfer]\n";
}

void ensureEfaAvailable() {
  if (LibfabricUtils::getAvailableEfaDevices().empty()) {
    fail("no EFA devices available for libfabric backend");
  }
}

nixl_opt_args_t makeBackendArgs(nixlBackendH *backend) {
  nixl_opt_args_t backendArgs;
  backendArgs.backends = {backend};
  return backendArgs;
}

TargetOptions parseTargetArgs(int argc, char **argv) {
  if (argc < 3 || argc > 5) {
    printUsage(argv[0]);
    std::exit(EXIT_FAILURE);
  }

  TargetOptions opts;
  opts.listenPort = parsePort(argv[2]);
  opts.gpu = argc > 3 ? parseInt(argv[3]) : 0;
  opts.bytes = argc > 4 ? parseSize(argv[4]) : opts.bytes;
  return opts;
}

InitiatorOptions parseInitiatorArgs(int argc, char **argv) {
  if (argc < 4 || argc > 9) {
    printUsage(argv[0]);
    std::exit(EXIT_FAILURE);
  }

  InitiatorOptions opts;
  opts.targetHost = argv[2];
  opts.targetPort = parsePort(argv[3]);
  opts.gpu = argc > 4 ? parseInt(argv[4]) : 0;
  opts.bytes = argc > 5 ? parseSize(argv[5]) : opts.bytes;
  opts.iterations = argc > 6 ? parseSize(argv[6]) : opts.iterations;
  opts.burst = argc > 7 ? parseSize(argv[7]) : opts.burst;
  opts.descsPerXfer = argc > 8 ? parseSize(argv[8]) : opts.descsPerXfer;
  if (opts.iterations == 0) {
    fail("iters must be greater than zero");
  }
  if (opts.burst == 0) {
    fail("burst must be greater than zero");
  }
  if (opts.descsPerXfer == 0) {
    fail("descs-per-xfer must be greater than zero");
  }
  return opts;
}

std::string joinPlugins(const std::vector<nixl_backend_t> &plugins) {
  std::ostringstream os;
  for (size_t i = 0; i < plugins.size(); ++i) {
    if (i > 0) {
      os << ", ";
    }
    os << plugins[i];
  }
  return os.str();
}

struct AgentWithBackend {
  nixlAgent agent;
  nixlBackendH *backend;

  AgentWithBackend(std::string name, int gpu)
      : agent(std::move(name), nixlAgentConfig(/*use_prog_thread=*/true)),
        backend(nullptr) {
    std::vector<nixl_backend_t> plugins;
    checkStatus(agent.getAvailPlugins(plugins),
                "failed to query available NIXL plugins");
    if (std::none_of(
            plugins.begin(), plugins.end(),
            [](const nixl_backend_t &plugin) { return plugin == kBackend; })) {
      failWithDetail("libfabric plugin unavailable",
                     plugins.empty() ? std::string("no plugins found")
                                     : joinPlugins(plugins));
    }

    nixl_mem_list_t mems;
    nixl_b_params_t backendParams;
    checkStatus(
        agent.getPluginParams(std::string(kBackend), mems, backendParams),
        "failed to get libfabric plugin params");
    backendParams["gpu_device_id"] = std::to_string(gpu);
    checkStatus(
        agent.createBackend(std::string(kBackend), backendParams, backend),
        "failed to create libfabric backend");
    if (backend == nullptr) {
      fail("libfabric backend handle is null");
    }
  }
};

std::optional<std::filesystem::path>
findBundledPluginDirectory(const char *argv0) {
  if (argv0 == nullptr || argv0[0] == '\0') {
    return std::nullopt;
  }

  std::error_code ec;
  const auto executable = std::filesystem::absolute(argv0, ec);
  if (ec) {
    return std::nullopt;
  }

  const auto runfiles =
      std::filesystem::path(executable.string() + ".runfiles");
  if (!std::filesystem::exists(runfiles)) {
    for (auto ancestor = executable.parent_path(); !ancestor.empty();
         ancestor = ancestor.parent_path()) {
      if (std::filesystem::exists(ancestor / "Driver" / kPluginFileName)) {
        return ancestor / "Driver";
      }
      if (std::filesystem::exists(ancestor / kPluginFileName)) {
        return ancestor;
      }
      if (ancestor == ancestor.root_path()) {
        break;
      }
    }
    return std::nullopt;
  }

  const std::vector<std::filesystem::path> candidates = {
      runfiles / "_main" / "MLRT",
      runfiles / "_main" / "MLRT" / "Driver" / "plugin_libfabric",
      runfiles / "__main__" / "MLRT",
      runfiles / "__main__" / "MLRT" / "Driver" / "plugin_libfabric",
      runfiles / "modular" / "MLRT",
      runfiles / "modular" / "MLRT" / "Driver" / "plugin_libfabric",
  };
  for (const auto &candidate : candidates) {
    if (std::filesystem::exists(candidate / kPluginFileName)) {
      return candidate;
    }
  }

  for (std::filesystem::recursive_directory_iterator it(runfiles, ec), end;
       !ec && it != end; it.increment(ec)) {
    if (it->path().filename() == kPluginFileName) {
      return it->path().parent_path();
    }
  }
  return std::nullopt;
}

void configureRuntimeEnvironment(const char *argv0) {
  if (std::getenv("FI_EFA_USE_DEVICE_RDMA") == nullptr) {
    ::setenv("FI_EFA_USE_DEVICE_RDMA", "1", 0);
  }
  if (std::getenv("FI_LOG_LEVEL") == nullptr) {
    ::setenv("FI_LOG_LEVEL", "warn", 0);
  }
  if (std::getenv("MODULAR_NIXL_TELEMETRY_ENABLE") == nullptr) {
    ::setenv("MODULAR_NIXL_TELEMETRY_ENABLE", "y", 0);
  }
  if (std::getenv("MODULAR_NIXL_PLUGIN_DIR") != nullptr) {
    return;
  }

  if (auto pluginDir = findBundledPluginDirectory(argv0)) {
    ::setenv("MODULAR_NIXL_PLUGIN_DIR", pluginDir->c_str(), 0);
  }
}

RegisteredVramBuffer prepareRegisteredVramBuffer(int gpu, size_t bytes,
                                                 uint8_t fillValue,
                                                 size_t descsPerXfer = 1) {
  auto context = takeValueOrDie(DeviceContext::create(Device::gpuAPI, gpu),
                                "failed to create GPU device context");
  auto buffer = takeValueOrDie(context->enqueueCreateBuffer(bytes),
                               "failed to allocate GPU buffer");
  checkSuccess(context->enqueueSetMemory(*buffer, fillValue, sizeof(uint8_t)),
               "failed to initialize GPU buffer");
  checkSuccess(context->synchronize(),
               "failed to synchronize GPU buffer setup");

  // Register the whole buffer as a single MR (matches real KV cache usage).
  nixl_reg_dlist_t regDescs(nixl_mem_t::VRAM_SEG);
  regDescs.addDesc(nixlBlobDesc(reinterpret_cast<uintptr_t>(buffer->ptr()),
                                bytes, static_cast<uint64_t>(gpu), ""));

  // Build the transfer descriptor list. `descsPerXfer == 1` keeps the legacy
  // single-descriptor behaviour. Larger values partition the buffer into
  // equal-sized chunks so one postXferReq fans out to many per-descriptor
  // fi_writedata calls, which matches the real KV cache transfer shape.
  nixl_xfer_dlist_t xferDescs(nixl_mem_t::VRAM_SEG);
  const size_t numChunks = std::max<size_t>(1, descsPerXfer);
  const size_t chunkSize = bytes / numChunks;
  if (chunkSize == 0) {
    fail("descs-per-xfer larger than buffer size in bytes");
  }
  for (size_t i = 0; i < numChunks; ++i) {
    const uintptr_t offset = static_cast<uintptr_t>(i) * chunkSize;
    const size_t len = (i == numChunks - 1) ? (bytes - offset) : chunkSize;
    xferDescs.addDesc(
        nixlBasicDesc(reinterpret_cast<uintptr_t>(buffer->ptr()) + offset, len,
                      static_cast<uint64_t>(gpu)));
  }

  return {std::move(context), std::move(buffer), std::move(regDescs),
          std::move(xferDescs)};
}

bool bufferMatches(const RegisteredVramBuffer &buffer, uint8_t expected) {
  std::vector<uint8_t> host(buffer.buffer->byteSize());
  checkSuccess(
      buffer.context->enqueueCopyFromDevice(host.data(), *buffer.buffer),
      "failed to copy GPU buffer back to host");
  checkSuccess(buffer.context->synchronize(),
               "failed to synchronize GPU verification copy");
  return std::all_of(host.begin(), host.end(),
                     [&](uint8_t value) { return value == expected; });
}

void verifyBuffer(const RegisteredVramBuffer &buffer, uint8_t expected) {
  if (!bufferMatches(buffer, expected)) {
    fail("device buffer contents did not match expected pattern");
  }
}

void waitForTransferComplete(nixlAgent &agent, nixlXferReqH *request,
                             std::string_view phase) {
  nixl_status_t status = agent.postXferReq(request);
  checkNonNegativeStatus(status,
                         std::string("failed to post ") + std::string(phase));
  while (status != nixl_status_t::NIXL_SUCCESS) {
    status = agent.getXferStatus(request);
    checkNonNegativeStatus(status, std::string("failed to query ") +
                                       std::string(phase) + " status");
    if (status != nixl_status_t::NIXL_SUCCESS) {
      std::this_thread::sleep_for(kPollSleep);
    }
  }
}

// Post a transfer request WITHOUT waiting for completion. Used by burst mode
// to stack many in-flight posts and stress the EFA TX send queue. Returns the
// initial postXferReq status so the caller can track per-request errors.
nixl_status_t postTransferNoWait(nixlAgent &agent, nixlXferReqH *request,
                                 size_t index) {
  nixl_status_t status = agent.postXferReq(request);
  if (isErrorStatus(status)) {
    std::cerr << "postXferReq #" << index
              << " failed: " << NixlEnumStrings::statusStr(status) << std::endl;
  }
  return status;
}

// Wait for a single already-posted transfer request to report NIXL_SUCCESS.
void waitForPostedTransfer(nixlAgent &agent, nixlXferReqH *request,
                           size_t index) {
  nixl_status_t status = agent.getXferStatus(request);
  while (status != nixl_status_t::NIXL_SUCCESS) {
    if (isErrorStatus(status)) {
      failWithDetail(std::string("posted transfer #") + std::to_string(index) +
                         " failed during completion",
                     NixlEnumStrings::statusStr(status));
    }
    std::this_thread::sleep_for(kPollSleep);
    status = agent.getXferStatus(request);
  }
}

struct ScopedFd {
  int fd = -1;

  ScopedFd() = default;
  explicit ScopedFd(int rawFd) : fd(rawFd) {}
  ~ScopedFd() {
    if (fd != -1) {
      ::close(fd);
    }
  }

  ScopedFd(const ScopedFd &) = delete;
  ScopedFd &operator=(const ScopedFd &) = delete;

  ScopedFd(ScopedFd &&other) noexcept : fd(other.fd) { other.fd = -1; }

  ScopedFd &operator=(ScopedFd &&other) noexcept {
    if (this != &other) {
      reset();
      fd = other.fd;
      other.fd = -1;
    }
    return *this;
  }

  int get() const { return fd; }

  void reset(int rawFd = -1) {
    if (fd != -1) {
      ::close(fd);
    }
    fd = rawFd;
  }
};

void writeAll(int fd, const void *data, size_t bytes) {
  auto *cursor = static_cast<const char *>(data);
  size_t remaining = bytes;
  while (remaining > 0) {
    const ssize_t written = ::send(fd, cursor, remaining, 0);
    if (written == 0) {
      fail("control socket closed while writing payload");
    }
    if (written == -1) {
      if (errno == EINTR) {
        continue;
      }
      failFromErrno("failed to write control socket payload");
    }
    cursor += written;
    remaining -= static_cast<size_t>(written);
  }
}

void readAll(int fd, void *data, size_t bytes) {
  auto *cursor = static_cast<char *>(data);
  size_t remaining = bytes;
  while (remaining > 0) {
    const ssize_t readBytes = ::recv(fd, cursor, remaining, 0);
    if (readBytes == 0) {
      fail("unexpected EOF while reading control socket payload");
    }
    if (readBytes == -1) {
      if (errno == EINTR) {
        continue;
      }
      failFromErrno("failed to read control socket payload");
    }
    cursor += readBytes;
    remaining -= static_cast<size_t>(readBytes);
  }
}

uint64_t readUInt64(int fd) {
  uint64_t value = 0;
  readAll(fd, &value, sizeof(value));
  return value;
}

void writeUInt64(int fd, uint64_t value) {
  writeAll(fd, &value, sizeof(value));
}

ScopedFd createListeningSocket(int port) {
  // Bind the control listener on an IPv6 dual-stack socket so initiators can
  // reach us over either IPv4 or IPv6 (many cluster environments only route
  // one of the two between pods).
  ScopedFd server(::socket(AF_INET6, SOCK_STREAM, 0));
  if (server.get() == -1) {
    failFromErrno("failed to create control listener socket");
  }

  constexpr int kEnable = 1;
  if (::setsockopt(server.get(), SOL_SOCKET, SO_REUSEADDR, &kEnable,
                   sizeof(kEnable)) == -1) {
    failFromErrno("failed to configure control listener socket reuse");
  }
  int v6only = 0;
  (void)::setsockopt(server.get(), IPPROTO_IPV6, IPV6_V6ONLY, &v6only,
                     sizeof(v6only));

  sockaddr_in6 addr{};
  addr.sin6_family = AF_INET6;
  addr.sin6_addr = in6addr_any;
  addr.sin6_port = htons(static_cast<uint16_t>(port));
  if (::bind(server.get(), reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) ==
      -1) {
    failFromErrno("failed to bind control listener socket");
  }
  if (::listen(server.get(), 1) == -1) {
    failFromErrno("failed to listen on control socket");
  }
  return server;
}

ScopedFd acceptClientConnection(int serverFd) {
  ScopedFd client(::accept(serverFd, nullptr, nullptr));
  if (client.get() == -1) {
    failFromErrno("failed to accept control socket connection");
  }
  return client;
}

ScopedFd connectToTarget(std::string_view host, int port) {
  addrinfo hints{};
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;

  addrinfo *results = nullptr;
  const std::string portString = std::to_string(port);
  const int lookupStatus = ::getaddrinfo(std::string(host).c_str(),
                                         portString.c_str(), &hints, &results);
  if (lookupStatus != 0) {
    failWithDetail("failed to resolve control target host",
                   ::gai_strerror(lookupStatus));
  }

  ScopedFd connection;
  for (const addrinfo *result = results; result != nullptr;
       result = result->ai_next) {
    ScopedFd candidate(
        ::socket(result->ai_family, result->ai_socktype, result->ai_protocol));
    if (candidate.get() == -1) {
      continue;
    }
    if (::connect(candidate.get(), result->ai_addr, result->ai_addrlen) == 0) {
      connection = std::move(candidate);
      break;
    }
  }
  ::freeaddrinfo(results);

  if (connection.get() == -1) {
    failFromErrno("failed to connect to target control socket");
  }
  return connection;
}

void sendTargetInfo(int fd, const TargetInfo &handshake) {
  writeUInt64(fd, static_cast<uint64_t>(handshake.metadata.size()));
  writeAll(fd, handshake.metadata.data(), handshake.metadata.size());
  writeUInt64(fd, handshake.baseAddr);
  writeUInt64(fd, handshake.bytes);
  writeUInt64(fd, handshake.deviceId);
}

TargetInfo receiveTargetInfo(std::string_view host, int port) {
  ScopedFd connection = connectToTarget(host, port);

  TargetInfo handshake;
  const uint64_t metadataSize = readUInt64(connection.get());
  if (metadataSize >
      static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
    fail("control socket metadata payload is too large");
  }
  handshake.metadata.resize(static_cast<size_t>(metadataSize));
  readAll(connection.get(), handshake.metadata.data(),
          handshake.metadata.size());
  handshake.baseAddr = readUInt64(connection.get());
  handshake.bytes = readUInt64(connection.get());
  handshake.deviceId = readUInt64(connection.get());
  return handshake;
}

TargetInfo buildTargetInfo(const RegisteredVramBuffer &buffer, int gpu,
                           nixlAgent &agent) {
  nixl_blob_t metadata;
  checkStatus(agent.getLocalMD(metadata), "failed to get target metadata");

  TargetInfo handshake;
  handshake.metadata = std::move(metadata);
  handshake.baseAddr =
      static_cast<uint64_t>(reinterpret_cast<uintptr_t>(buffer.buffer->ptr()));
  handshake.bytes = static_cast<uint64_t>(buffer.buffer->byteSize());
  handshake.deviceId = static_cast<uint64_t>(gpu);
  return handshake;
}

void serveTargetInfo(const TargetInfo &handshake, int listenPort) {
  ScopedFd server = createListeningSocket(listenPort);
  std::cout << "Target control listener ready on port " << listenPort
            << std::endl;
  ScopedFd client = acceptClientConnection(server.get());
  sendTargetInfo(client.get(), handshake);
  std::cout << "Target metadata handshake sent" << std::endl;
}

TargetInfo fetchTargetInfo(std::string_view host, int port) {
  std::cout << "Connecting to target control socket at " << host << ":" << port
            << std::endl;
  return receiveTargetInfo(host, port);
}

void validateTargetInfo(const TargetInfo &handshake,
                        const InitiatorOptions &opts) {
  if (handshake.bytes != static_cast<uint64_t>(opts.bytes)) {
    fail("target handshake bytes did not match requested transfer size");
  }
}

nixl_xfer_dlist_t makeRemoteDescs(const TargetInfo &handshake,
                                  size_t descsPerXfer = 1) {
  nixl_xfer_dlist_t remoteDescs(nixl_mem_t::VRAM_SEG);
  const size_t numChunks = std::max<size_t>(1, descsPerXfer);
  const size_t chunkSize = static_cast<size_t>(handshake.bytes) / numChunks;
  for (size_t i = 0; i < numChunks; ++i) {
    const uintptr_t offset =
        static_cast<uintptr_t>(i) * static_cast<uintptr_t>(chunkSize);
    const size_t len = (i == numChunks - 1)
                           ? static_cast<size_t>(handshake.bytes) - offset
                           : chunkSize;
    remoteDescs.addDesc(
        nixlBasicDesc(static_cast<uintptr_t>(handshake.baseAddr) + offset, len,
                      handshake.deviceId));
  }
  return remoteDescs;
}

nixlXferReqH *createTransferRequest(nixlAgent &agent,
                                    const nixl_xfer_dlist_t &localDescs,
                                    const nixl_xfer_dlist_t &remoteDescs,
                                    const nixl_opt_args_t &backendArgs) {
  nixlXferReqH *request = nullptr;
  nixl_status_t status = nixl_status_t::NIXL_ERR_NOT_FOUND;
  for (int retry = 0;
       retry < kRetryLimit && status == nixl_status_t::NIXL_ERR_NOT_FOUND;
       ++retry) {
    status =
        agent.createXferReq(nixl_xfer_op_t::NIXL_WRITE, localDescs, remoteDescs,
                            std::string(kTarget), request, &backendArgs);
    if (status == nixl_status_t::NIXL_ERR_NOT_FOUND) {
      std::this_thread::sleep_for(kPollSleep);
    }
  }
  checkStatus(status, "failed to create transfer request");
  return request;
}

void waitForRemoteWrite(const RegisteredVramBuffer &buffer) {
  std::this_thread::sleep_for(kTargetVerifyDelay);
  verifyBuffer(buffer, kTransferPattern);
}

void runTarget(const TargetOptions &opts) {
  ensureEfaAvailable();

  AgentWithBackend target(std::string(kTarget), opts.gpu);
  const nixl_opt_args_t backendArgs = makeBackendArgs(target.backend);

  auto buffer = prepareRegisteredVramBuffer(opts.gpu, opts.bytes, 0);
  checkStatus(target.agent.registerMem(buffer.regDescs, &backendArgs),
              "failed to register target VRAM");

  serveTargetInfo(buildTargetInfo(buffer, opts.gpu, target.agent),
                  opts.listenPort);
  waitForRemoteWrite(buffer);

  checkStatus(target.agent.deregisterMem(buffer.regDescs, &backendArgs),
              "failed to deregister target VRAM");
  std::cout << "Target verification passed for " << opts.bytes
            << " bytes on GPU " << opts.gpu << std::endl;
}

void runInitiator(const InitiatorOptions &opts) {
  ensureEfaAvailable();

  AgentWithBackend initiator(std::string(kInitiator), opts.gpu);
  const nixl_opt_args_t backendArgs = makeBackendArgs(initiator.backend);

  auto buffer = prepareRegisteredVramBuffer(
      opts.gpu, opts.bytes, kTransferPattern, opts.descsPerXfer);
  checkStatus(initiator.agent.registerMem(buffer.regDescs, &backendArgs),
              "failed to register initiator VRAM");

  const TargetInfo targetInfo =
      fetchTargetInfo(opts.targetHost, opts.targetPort);
  validateTargetInfo(targetInfo, opts);

  std::string remoteAgent;
  checkStatus(
      initiator.agent.loadRemoteMD(blob_ref(targetInfo.metadata), remoteAgent),
      "failed to load target metadata");
  if (remoteAgent != kTarget) {
    failWithDetail("loaded metadata for unexpected remote agent", remoteAgent);
  }

  const nixl_xfer_dlist_t remoteDescs =
      makeRemoteDescs(targetInfo, opts.descsPerXfer);

  nixlXferReqH *warmupRequest = createTransferRequest(
      initiator.agent, buffer.xferDescs, remoteDescs, backendArgs);
  for (size_t warmup = 0; warmup < kWarmupIterations; ++warmup) {
    waitForTransferComplete(initiator.agent, warmupRequest, "warmup transfer");
  }
  checkStatus(initiator.agent.releaseXferReq(warmupRequest),
              "failed to release warmup transfer request");

  std::vector<nixlXferReqH *> requests;
  requests.reserve(opts.iterations);
  for (size_t iter = 0; iter < opts.iterations; ++iter) {
    requests.push_back(createTransferRequest(initiator.agent, buffer.xferDescs,
                                             remoteDescs, backendArgs));
  }

  const auto start = std::chrono::steady_clock::now();
  if (opts.burst <= 1) {
    for (auto *request : requests) {
      waitForTransferComplete(initiator.agent, request, "transfer request");
    }
  } else {
    // Burst mode: post `burst` requests back-to-back before collecting
    // completions, which forces many concurrent fi_writedata posts and
    // pressures the libfabric/EFA TX send queue. Posts that fail (e.g.
    // surface -FI_EAGAIN as NIXL_ERR_BACKEND) are counted and reported.
    std::cout << "Burst mode: posting in windows of " << opts.burst
              << " concurrent transfers" << std::endl;
    size_t postedTotal = 0;
    size_t postErrorsTotal = 0;
    for (size_t base = 0; base < requests.size(); base += opts.burst) {
      const size_t windowEnd = std::min(base + opts.burst, requests.size());
      size_t windowPosted = 0;
      size_t windowErrors = 0;
      for (size_t i = base; i < windowEnd; ++i) {
        nixl_status_t postStatus =
            postTransferNoWait(initiator.agent, requests[i], i);
        if (isErrorStatus(postStatus)) {
          ++windowErrors;
        } else {
          ++windowPosted;
        }
      }
      std::cout << "  window [" << base << ".." << (windowEnd - 1)
                << "]: posted=" << windowPosted << " errors=" << windowErrors
                << std::endl;
      postedTotal += windowPosted;
      postErrorsTotal += windowErrors;
      // Drain this window before posting the next.
      for (size_t i = base; i < windowEnd; ++i) {
        waitForPostedTransfer(initiator.agent, requests[i], i);
      }
    }
    std::cout << "Burst summary: posted=" << postedTotal
              << " postErrors=" << postErrorsTotal << std::endl;
    if (postErrorsTotal > 0) {
      fail("burst posts surfaced errors; see preceding log for details");
    }
  }
  const auto end = std::chrono::steady_clock::now();

  nixl_xfer_telem_t telemetry;
  checkStatus(initiator.agent.getXferTelemetry(requests.back(), telemetry),
              "failed to fetch transfer telemetry");

  const double elapsedSeconds =
      std::chrono::duration<double>(end - start).count();
  const double totalBytes =
      static_cast<double>(opts.bytes) * static_cast<double>(opts.iterations);
  const double gbPerSecond =
      elapsedSeconds > 0.0 ? totalBytes / elapsedSeconds / 1.0e9 : 0.0;

  std::cout << std::fixed << std::setprecision(2)
            << "Initiator transfer passed\n"
            << "  bytes:          " << opts.bytes << "\n"
            << "  warmup iters:   " << kWarmupIterations << "\n"
            << "  iterations:     " << opts.iterations << "\n"
            << "  wall clock GB/s:" << gbPerSecond << "\n"
            << "  telemetry bytes:" << telemetry.totalBytes << "\n"
            << "  telemetry descs:" << telemetry.descCount << "\n"
            << "  telemetry post: " << telemetry.postDuration.count() << " us\n"
            << "  telemetry xfer: " << telemetry.xferDuration.count() << " us"
            << std::endl;

  for (auto *request : requests) {
    checkStatus(initiator.agent.releaseXferReq(request),
                "failed to release transfer request");
  }
  checkStatus(initiator.agent.deregisterMem(buffer.regDescs, &backendArgs),
              "failed to deregister initiator VRAM");
}

} // namespace

int main(int argc, char **argv) {
  configureRuntimeEnvironment(argv[0]);

  if (argc < 2) {
    printUsage(argv[0]);
    return EXIT_FAILURE;
  }

  const std::string mode = argv[1];
  if (mode == "target") {
    runTarget(parseTargetArgs(argc, argv));
    return EXIT_SUCCESS;
  }
  if (mode == "initiator") {
    runInitiator(parseInitiatorArgs(argc, argv));
    return EXIT_SUCCESS;
  }

  printUsage(argv[0]);
  return EXIT_FAILURE;
}
