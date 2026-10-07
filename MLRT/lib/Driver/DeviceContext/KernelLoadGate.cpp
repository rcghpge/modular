//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "MLRT/Driver/DeviceContext/KernelLoadGate.h"

#include "AsyncRT/Runtime/Algorithms.h"
#include "AsyncRT/Runtime/CPUDevice.h"

#include <cassert>

namespace M::Driver {

using AsyncRT::AsyncValueRef;
using AsyncRT::Chain;

bool KernelLoadGate::tryAcquireCollective(AsyncRT::CPUDevice &cpuDevice,
                                          AsyncValueRef<Chain> &grant) {
  // Hold lock while updating KernelLoadGate internal state
  std::lock_guard<std::mutex> lock(mu_);
  if (state_ == State::Idle) {
    // Idle -> Active: allocate idle chain for next loader, return true.
    idle_ = AsyncValueRef<Chain>::allocate(cpuDevice);
    if (activeLoads_ == 0) {
      state_ = State::Active;
      return true;
    }
    // Idle -> Draining: allocate drain grant for caller and return false.
    state_ = State::Draining;
    drainGrant_ = AsyncValueRef<Chain>::allocate(cpuDevice);
    grant = drainGrant_.copy();
    return false;
  }
  // {Active, Draining}: enqueue chain onto pending list and return false.
  pending_.push_back(AsyncValueRef<Chain>::allocate(cpuDevice));
  grant = pending_.back().copy();
  return false;
}

void KernelLoadGate::releaseCollective() {
  AsyncValueRef<Chain> wake;
  {
    // Hold lock while updating KernelLoadGate internal state
    std::lock_guard<std::mutex> lock(mu_);
    assert(state_ == State::Active &&
           "releasing a kernel-load gate no collective owns");
    if (pending_.empty()) {
      // Active -> Idle: mark idle chain ready to wake next loader.
      state_ = State::Idle;
      wake = std::move(idle_);
    } else {
      // Pop next chain from pending list.
      AsyncValueRef<Chain> next = std::move(pending_.front());
      pending_.pop_front();
      // The owner's own loads close before its enqueues finish, and every other
      // load has waited since the claim, so this is the normal case. Drain
      // rather than trust it.
      // TODO: assert (activeLoads_ == 0)?
      if (activeLoads_ == 0) {
        wake = std::move(next);
      } else {
        state_ = State::Draining;
        drainGrant_ = std::move(next);
      }
    }
  }
  // Wake up the next loader outside the critical section.
  if (wake)
    std::move(wake).emplace();
}

void KernelLoadGate::openLoad() {
  while (true) {
    AsyncValueRef<Chain> waitOn;
    {
      // Hold lock while updating KernelLoadGate internal state
      std::lock_guard<std::mutex> lock(mu_);
      // Idle: increment active load counter and return.
      // Special case: if this thread is enqueuing a collective, it already
      // holds the gate and doesn't need to wait for itself. But we still
      // track the pending load.
      if (state_ == State::Idle || isEnqueueThread()) {
        ++activeLoads_;
        return;
      }
      // {Active, Draining}: copy idle chain and wait on it.
      waitOn = idle_.copy();
    }
    // Wait for the collective to complete, then try again.
    AsyncRT::await(waitOn);
  }
}

void KernelLoadGate::closeLoad() {
  AsyncValueRef<Chain> grant;
  {
    // Hold lock while updating KernelLoadGate internal state
    std::lock_guard<std::mutex> lock(mu_);
    // Decrement active load counter.
    --activeLoads_;
    if (activeLoads_ == 0 && state_ == State::Draining) {
      // Draining -> Active: open the gate for the next collective.
      state_ = State::Active;
      grant = std::move(drainGrant_);
    }
  }
  if (grant)
    std::move(grant).emplace();
}

bool KernelLoadGate::isEnqueueThread() const {
  std::thread::id self = std::this_thread::get_id();
  for (const auto &entry : enqueueThreads_)
    if (entry.first == self)
      return true;
  return false;
}

void KernelLoadGate::beginEnqueue() {
  std::thread::id self = std::this_thread::get_id();
  std::lock_guard<std::mutex> lock(mu_);
  for (auto &entry : enqueueThreads_) {
    // Depth rather than a flag: a collective's per-device work can nest.
    if (entry.first == self) {
      ++entry.second;
      return;
    }
  }
  enqueueThreads_.emplace_back(self, 1);
}

void KernelLoadGate::endEnqueue() {
  std::thread::id self = std::this_thread::get_id();
  std::lock_guard<std::mutex> lock(mu_);
  for (auto &entry : enqueueThreads_) {
    if (entry.first != self)
      continue;
    if (--entry.second == 0) {
      // Order does not matter, so swap with the back rather than shift.
      entry = enqueueThreads_.back();
      enqueueThreads_.pop_back();
    }
    return;
  }
  assert(false && "unbalanced kernel-load gate enqueue scope");
}

} // namespace M::Driver
