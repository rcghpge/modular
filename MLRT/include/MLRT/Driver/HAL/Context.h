//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file provides C++ bindings over the Hardware Abstraction Layer (HAL) C
// interface in SDK/Driver.
//
//===----------------------------------------------------------------------===//

#ifndef MLRT_DRIVER_HAL_CONTEXT_H
#define MLRT_DRIVER_HAL_CONTEXT_H

#include "Buffer.h"
#include "Event.h"
#include "Function.h"
#include "HAL/M_driver_api.h"
#include "Queue.h"
#include "Support/ErrorOr.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/RCRef.h"
#include "Support/ReferenceCounted.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Hashing.h"

#include <cstdint>
#include <mutex>
#include <shared_mutex>
#include <string>

namespace M::Driver::HAL {

class Device;
using DeviceRef = RCRef<Device>;

/// Cache key for loaded device functions, used to avoid redundant JIT
/// compilation.
struct FunctionCacheKey {
  std::string name;
  std::string debugLevel;
  int32_t maxDynamicSharedBytes;
  int32_t optimizationLevel;

  bool operator==(const FunctionCacheKey &other) const {
    return name == other.name && debugLevel == other.debugLevel &&
           maxDynamicSharedBytes == other.maxDynamicSharedBytes &&
           optimizationLevel == other.optimizationLevel;
  }
};

class Context : public ReferenceCounted<Context> {
public:
  ~Context();

  /// Retrieve the device for this context.
  DeviceRef getDevice() const;

  ErrorOr<BufferRef> allocBufferSync(size_t byteSize);
  ErrorOr<BufferRef> allocBufferPinned(size_t byteSize);
  ErrorOr<QueueRef> createQueue();
  ErrorOr<EventRef> createEvent();

  /// Load a named function from device code. Results are cached per-device
  /// to avoid redundant JIT compilation.
  ///
  /// \param moduleName    Module name used as cache key (if empty, the entire
  ///                      code string is used instead).
  /// \param functionName  Name of the kernel entry point.
  /// \param code          Device code image.
  /// \param maxDynamicSharedBytes  Max dynamic shared memory.
  /// \param debugLevel    JIT debug level.
  /// \param optimizationLevel  JIT optimization level.
  ErrorOr<FunctionRef>
  loadFunction(llvm::StringRef moduleName, llvm::StringRef functionName,
               llvm::StringRef code, int32_t maxDynamicSharedBytes = -1,
               llvm::StringRef debugLevel = "", int32_t optimizationLevel = -1);

private:
  explicit Context(DeviceRef device, M_driver_context_t handle);

  DeviceRef device;
  M_driver_context_t handle;

  std::shared_mutex funcCacheMutex;
  llvm::DenseMap<FunctionCacheKey, FunctionRef> funcCache;

  friend class Device;
  friend class Buffer;
  friend class Queue;
  friend class Function;
  friend class Event;
};

using ContextRef = RCRef<Context>;

} // namespace M::Driver::HAL

namespace llvm {
template <>
struct DenseMapInfo<M::Driver::HAL::FunctionCacheKey> {
  static inline M::Driver::HAL::FunctionCacheKey getEmptyKey() {
    return {"", "", 0, 0};
  }
  static inline M::Driver::HAL::FunctionCacheKey getTombstoneKey() {
    return {"", "", -1, -1};
  }
  static unsigned getHashValue(const M::Driver::HAL::FunctionCacheKey &val) {
    return hash_combine(val.name, val.debugLevel, val.maxDynamicSharedBytes,
                        val.optimizationLevel);
  }
  static bool isEqual(const M::Driver::HAL::FunctionCacheKey &lhs,
                      const M::Driver::HAL::FunctionCacheKey &rhs) {
    return lhs == rhs;
  }
};
} // namespace llvm

#endif // MLRT_DRIVER_HAL_CONTEXT_H
