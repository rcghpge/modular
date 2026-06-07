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
#include "MLRT/Driver/HAL/Bundle.h"
#include "Queue.h"
#include "Support/ErrorOr.h"
#include "Support/RCRef.h"
#include "Support/ReferenceCounted.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Hashing.h"

#include <cstdint>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <vector>

namespace M::Driver::HAL {

class Device;
using DeviceRef = RCRef<Device>;

/// Cache key for loaded device bundles, used to avoid redundant JIT
/// compilation.
struct BundleCacheKey {
  std::string name;
  std::string debugLevel;
  int32_t optimizationLevel;

  bool operator==(const BundleCacheKey &other) const {
    return name == other.name && debugLevel == other.debugLevel &&
           optimizationLevel == other.optimizationLevel;
  }
};

struct FuncCacheKey {
  M_driver_runtime_bundle_t bundle;
  std::string name;

  bool operator==(const FuncCacheKey &other) const {
    return bundle == other.bundle && name == other.name;
  }
};

class Context : public ReferenceCounted<Context> {
public:
  ~Context();

  DeviceRef getDevice() const;

  ErrorOr<BufferRef> allocBufferSync(size_t byteSize);
  ErrorOr<BufferRef> allocBufferPinned(size_t byteSize);
  ErrorOr<QueueRef> createQueue();

  ErrorOr<BundleRef> loadBundle(llvm::StringRef bundleName,
                                llvm::StringRef code, llvm::StringRef codeType,
                                llvm::StringRef debugLevel = "",
                                int32_t optimizationLevel = -1);

  /// Load a named function from device code. Results are cached per-device
  /// to avoid redundant JIT compilation.
  ///
  /// \param moduleName    Module name used as cache key (if empty, the entire
  ///                      code string is used instead).
  /// \param functionName  Name of the kernel entry point.
  /// \param code          Device code image.
  /// \param debugLevel    JIT debug level.
  /// \param optimizationLevel  JIT optimization level.
  ErrorOr<FunctionRef> loadFunction(BundleRef bundle,
                                    llvm::StringRef functionName);

private:
  explicit Context(DeviceRef device, M_driver_context_t handle);

  ErrorOr<EventRef> createEvent(M_driver_event_flags_t flags);

  void recycleEventHandle(M_driver_event_t handle,
                          M_driver_event_flags_t flags);

  DeviceRef device;
  M_driver_context_t handle;

  std::mutex eventRecyclerMutex;
  // Keyed by flags bitmask so a CPU_VISIBLE event is never handed out
  // when intra-GPU was requested, and vice versa.
  llvm::DenseMap<M_driver_event_flags_t, std::vector<M_driver_event_t>>
      eventRecycler;

  std::shared_mutex bundleCacheMutex;
  llvm::DenseMap<BundleCacheKey, BundleRef> bundleCache;

  std::shared_mutex funcCacheMutex;
  llvm::DenseMap<FuncCacheKey, FunctionRef> funcCache;

  friend class Device;
  friend class Buffer;
  friend class Queue;
  friend class Bundle;
  friend class Function;
  friend class Event;
};

using ContextRef = RCRef<Context>;

} // namespace M::Driver::HAL

namespace llvm {
template <>
struct DenseMapInfo<M::Driver::HAL::BundleCacheKey> {
  static inline M::Driver::HAL::BundleCacheKey getEmptyKey() {
    return {"", "", 0};
  }
  static inline M::Driver::HAL::BundleCacheKey getTombstoneKey() {
    return {"", "", -1};
  }
  static unsigned getHashValue(const M::Driver::HAL::BundleCacheKey &val) {
    return hash_combine(val.name, val.debugLevel, val.optimizationLevel);
  }
  static bool isEqual(const M::Driver::HAL::BundleCacheKey &lhs,
                      const M::Driver::HAL::BundleCacheKey &rhs) {
    return lhs == rhs;
  }
};

template <>
struct DenseMapInfo<M::Driver::HAL::FuncCacheKey> {
  static inline M::Driver::HAL::FuncCacheKey getEmptyKey() {
    return {.bundle = DenseMapInfo<M_driver_runtime_bundle_t>::getEmptyKey(),
            .name = ""};
  }
  static unsigned getHashValue(const M::Driver::HAL::FuncCacheKey &val) {
    return hash_combine(val.bundle, val.name);
  }
  static bool isEqual(const M::Driver::HAL::FuncCacheKey &lhs,
                      const M::Driver::HAL::FuncCacheKey &rhs) {
    return lhs == rhs;
  }
};
} // namespace llvm

#endif // MLRT_DRIVER_HAL_CONTEXT_H
