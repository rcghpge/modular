//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions:
// https://llvm.org/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//

#ifndef KGEN_COMPILERCACHES_H
#define KGEN_COMPILERCACHES_H

#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/DenseMap.h"

#include <array>
#include <memory>
#include <mutex>
#include <shared_mutex>

namespace M::KGEN {

/// Thread-safe cache from an attribute or type to its canonical form. Canonical
/// forms are pure functions of the uniqued value, so entries never go stale.
class CanonicalCache {
public:
  /// Returns the cached canonical form of `key`, or null.
  const void *lookup(const void *key) {
    Shard &shard = getShard(key);
    std::shared_lock<std::shared_mutex> lock(shard.mutex);
    return shard.map.lookup(key);
  }

  /// Records that `result` is the canonical form of `key`, and of itself.
  void insert(const void *key, const void *result) {
    insertOne(key, result);
    if (key != result)
      insertOne(result, result);
  }

private:
  // Sharded to avoid one global lock.
  static constexpr size_t numShards = 64;
  struct Shard {
    std::shared_mutex mutex;
    llvm::DenseMap<const void *, const void *> map;
  };

  Shard &getShard(const void *key) {
    return shards[llvm::DenseMapInfo<const void *>::getHashValue(key) %
                  numShards];
  }

  void insertOne(const void *key, const void *value) {
    Shard &shard = getShard(key);
    std::unique_lock<std::shared_mutex> lock(shard.mutex);
    shard.map.try_emplace(key, value);
  }

  std::array<Shard, numShards> shards;
};

/// Caches for pure helpers of the parameter machinery. Every parser state on an
/// `MLIRContext` shares one instance, owned through the `shared_ptr` that
/// `getOrCreate` returns; it lives until the last owner releases it. Helpers
/// that only have the context find it with `get`, and compute without caching
/// when no owner is alive.
class CompilerCaches {
public:
  /// Returns the caches of `context`, or null if no owner is alive.
  static CompilerCaches *get(mlir::MLIRContext *context);

  /// Returns the caches shared by all owners on `context`, creating them for
  /// the first one.
  static std::shared_ptr<CompilerCaches>
  getOrCreate(mlir::MLIRContext *context);

  /// Canonical forms, keyed by the value's opaque pointer.
  CanonicalCache canonical;

private:
  CompilerCaches() = default;
};

} // namespace M::KGEN

#endif // KGEN_COMPILERCACHES_H
