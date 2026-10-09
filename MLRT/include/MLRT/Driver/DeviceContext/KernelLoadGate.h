//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Keeps first-time kernel loads from interleaving with a multi-device
// collective's per-device enqueues. The mechanism only: an instance is owned
// by the CUDA driver (see CUDADriver) rather than by one context, because the
// deadlock it prevents spans contexts. HIP loads do not wait for a busy
// device, so HIP has none.
//
//===----------------------------------------------------------------------===//

#ifndef MLRT_DRIVER_DEVICECONTEXT_KERNELLOADGATE_H
#define MLRT_DRIVER_DEVICECONTEXT_KERNELLOADGATE_H

#include "AsyncRT/ForwardDecls.h"
#include "AsyncRT/Runtime/AsyncValueRef.h"
#include "llvm/ADT/SmallVector.h"

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <utility>

namespace M::Driver {

/// Keeps first-time kernel loads from interleaving with a multi-device
/// collective's per-device enqueues.
///
/// `loadFunction`'s driver calls block until their device quiesces, and hold
/// that device's context lock while they do. A collective kernel spins at a
/// device-side barrier until every peer arrives, so a load that begins while a
/// collective is only partway through its enqueues waits on a device that may
/// never go idle. The driver's locks are per context, so the cycle closes
/// through a third party that ties a peer's enqueue to the loading context: a
/// thread that holds another context's lock while it waits for the loading
/// one's, such as a cross-device copy or a deferred free recording an event on
/// the other device, or the worker a peer's enqueue is pinned to, parked in the
/// load itself. Keeping loads out of the enqueues removes the interleaving
/// every such cycle needs. See MXSERV-441.
///
/// Four properties are load-bearing and were all learned the hard way.
///
/// No state is static. Every field below lives in this object, which is a
/// member of the driver, so the reference each `DeviceContext` hands out
/// reaches the same bytes from every caller. That matters because a served
/// process links this code statically into five shared objects and the two
/// sides of the gate land in different ones: the FFI entry points in
/// `libAsyncRTMojoBindings.so`, `CUDADeviceContext::loadFunction` in
/// `_core.so`. File scope here, and an internal-linkage `thread_local`
/// especially, is five variables no linker will coalesce.
///
/// Loads do not exclude each other. They are readers. An exclusive gate
/// serializes a cold load on one device behind a cold load on another, which
/// strands a peer that would otherwise have launched and let the first load's
/// device drain.
///
/// Waiting donates rather than parks. A worker blocked on a plain mutex cannot
/// run the collective sub-task pinned to it, so the gate would deadlock on the
/// very thread it is waiting for. Load waiters suspend on an
/// `AsyncValueRef<Chain>` through `AsyncRT::await`, which on a worker donates
/// the thread to the queue instead of blocking it.
///
/// No frame holds the gate across a wait. A donating wait runs other tasks as
/// nested calls on the waiter's stack, so a frame that held the gate while
/// waiting for its fan-out could end up beneath a task that needs the gate,
/// and only that buried frame could release it. So the collective side never
/// blocks: `tryAcquireCollective` either grants the gate or hands back a grant
/// to continue from, and whichever thread finishes the last per-device enqueue
/// calls `releaseCollective`. Loads may still wait, because a load waits before
/// it is counted and so holds nothing while it does. The failure this rules out
/// is reproduced by test_kernel_load_gate_nested.
///
/// Only `tryAcquireCollective` allocates, and only it needs a `CPUDevice`. A
/// load either finds the gate idle, which costs a counter under the mutex, or
/// waits on the chain the busy gate already allocated. So the load path never
/// reaches for a runtime, and a collective cannot claim the gate without a
/// caller that holds one.
class KernelLoadGate {
public:
  KernelLoadGate() = default;
  KernelLoadGate(const KernelLoadGate &) = delete;
  KernelLoadGate &operator=(const KernelLoadGate &) = delete;

  /// Claims the gate for one collective's enqueues. Never blocks.
  ///
  /// Returns true if the collective owns the gate now. Otherwise returns false
  /// and sets `grant` to a chain that becomes available once it does: after the
  /// collective ahead of it releases, or after in-flight loads drain. Either
  /// way the caller dispatches its enqueues only once it owns the gate, and
  /// must call `releaseCollective` exactly once after the last of them.
  bool tryAcquireCollective(AsyncRT::CPUDevice &cpuDevice,
                            AsyncRT::AsyncValueRef<AsyncRT::Chain> &grant);

  /// Ends the owning collective's claim. Hands the gate to the next queued
  /// collective, or reopens it to loads. Called from whichever thread finishes
  /// the last per-device enqueue, never from a frame waiting on the fan-out.
  void releaseCollective();

  /// Enters a region that must not overlap a collective's enqueues: the
  /// driver calls of a first-time kernel load, or a vendor library call that
  /// can block on a busy device. Waits while any collective holds or is
  /// queued for the gate; concurrent with other such regions.
  ///
  /// Does not wait on a thread registered through `beginEnqueue`, because that
  /// work belongs to the collective holding the gate.
  ///
  /// The paired form for a caller that cannot hold a C++ object across the
  /// region, which is the FFI. C++ callers should prefer `LoadScope`.
  void openLoad();
  void closeLoad();

  /// RAII form of `openLoad` / `closeLoad`.
  class LoadScope {
  public:
    explicit LoadScope(KernelLoadGate &gate) : gate_(gate) { gate_.openLoad(); }
    ~LoadScope() { gate_.closeLoad(); }
    LoadScope(const LoadScope &) = delete;
    LoadScope &operator=(const LoadScope &) = delete;

  private:
    KernelLoadGate &gate_;
  };

  /// Registers the calling thread as running one device's share of the enqueue
  /// work for the collective holding the gate, so its loads proceed instead of
  /// waiting on a gate their own collective owns.
  ///
  /// Keyed on thread identity because that is the only channel available: the
  /// sub-tasks run on the affinity workers rather than the thread that claimed
  /// the gate, and their loads reach `loadFunction` from arbitrary depth
  /// inside kernel code with no parameter to carry the fact down. Sound only
  /// because a sub-task body never suspends, so the registration cannot follow
  /// a migrating task onto another worker, and no other task runs on the
  /// thread while it is registered. Registering a thread twice would mean a
  /// body did wait, so it asserts.
  ///
  /// Paired calls rather than a scope object because the only caller is
  /// across the FFI, which cannot hold one.
  void beginEnqueue();
  void endEnqueue();

private:
  enum class State {
    /// No collective holds or is queued for the gate; loads proceed.
    Idle,
    /// A collective has claimed the gate and waits for in-flight loads to
    /// finish before it may enqueue.
    Draining,
    /// A collective owns the gate and is enqueueing.
    Active,
  };

  /// Whether the calling thread is registered through `beginEnqueue`.
  /// Callers hold `mu_`.
  bool isEnqueueThread() const;

  std::mutex mu_;
  State state_ = State::Idle;
  size_t inFlightLoads_ = 0;
  /// Threads running the open collective's per-device enqueues, keyed by
  /// `llvm::get_threadid()`, the identity AsyncRT itself checks its workers
  /// against. Holds one entry per participating device, so a scan is cheaper
  /// than a map, and the inline capacity covers a node's devices without
  /// allocating.
  llvm::SmallVector<uint64_t, 8> enqueueThreads_;
  /// Fulfilled when the gate returns to Idle; waited on by loads. Survives a
  /// hand-off between queued collectives, so loads wait out the whole run of
  /// them. Null while Idle.
  AsyncRT::AsyncValueRef<AsyncRT::Chain> idle_;
  /// The grant owed to the collective in Draining, fulfilled by the load whose
  /// close brings `inFlightLoads_` to zero. Null outside Draining.
  AsyncRT::AsyncValueRef<AsyncRT::Chain> drainGrant_;
  /// Grants for collectives queued behind the owner, in arrival order. Empty
  /// unless collectives overlap, and short when they do.
  llvm::SmallVector<AsyncRT::AsyncValueRef<AsyncRT::Chain>, 2>
      queuedCollectives_;
};

} // namespace M::Driver

#endif // MLRT_DRIVER_DEVICECONTEXT_KERNELLOADGATE_H
