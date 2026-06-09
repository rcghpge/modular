//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Unit tests for MGP_RT_CreateAsyncDeviceBufferRefByTakingHandle, the runtime
// entrypoint that ADOPTS an already-owning DeviceBuffer handle (net-zero, no
// addRef) instead of minting a fresh owner with createBufferOwning. It guards
// the refcount contract that pairs with the Mojo-side
// DeviceBuffer.take_handle() (deinit self, no release): takeRCRef must NOT
// addRef, and the produced TensorBufferRef must hold exactly the single
// reference the producer surrendered, releasing it on teardown. ASAN cannot
// observe a +1/-0 imbalance that nets >= 1, so we assert the exact
// ReferenceCounted count directly.
//
//===----------------------------------------------------------------------===//

#include "MLRT/AsyncRT/Runtime/AnyAsyncValueRef.h"
#include "MLRT/Core/Tensor/TensorBufferRef.h"
#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "MLRT/unittests/Driver/TestUtils.h"

#include "gtest/gtest.h"

#include <cstddef>

using namespace M;
using namespace M::MLRT;

namespace {

// Mirror of the opaque single-pointer wrapper MGPRT.cpp uses on the Mojo ABI
// boundary (it is file-private there, not exported in any header). Identical
// layout => identical by-value calling convention; the extern "C" symbol is
// resolved by name, so the parameter's C++ type only has to match the ABI.
template <typename T>
struct AsyncRTWrapper {
  void *ptr;
};

} // namespace

// The entrypoint under test, declared locally: it is a bare extern "C" export
// with no header, linked from the //GraphCompiler:MGPRT shared library.
extern "C" void MGP_RT_CreateAsyncDeviceBufferRefByTakingHandle(
    Driver::DeviceBuffer *handle, void *data, size_t size,
    AsyncRTWrapper<AnyAsyncValueRef> async);

namespace {

// Creating a "cpu" DeviceContext also installs the process-global CPUDevice
// (CpuDeviceContext ctor -> getOrCreateCPUDevice -> setGlobalCPUDevicePointer),
// which is what the entrypoint's CPUDevice::getCurrentCPUDeviceOrNull() reads.
// The returned ref must stay alive across the call, or ~CPUDevice clears the
// global pointer and the entrypoint dereferences null.
TEST(MGPRTTakeHandle, AdoptsHandleNetZeroAndReleasesOnDrop) {
  ASSIGN_NO_ERR(Driver::DeviceContextRef, ctx,
                Driver::DeviceContext::create(Driver::Device::cpuAPI));

  constexpr size_t kSize = 64;
  ASSIGN_NO_ERR(Driver::DeviceBufferRef, bufRef,
                ctx->enqueueCreateBuffer(kSize));
  ASSERT_TRUE(bufRef->isUnique()); // freshly minted owner, refcount 1

  // Keep-alive copy so the count is safe to read AFTER the async value drops
  // (reading getNumReferences() on a freed object would be a UAF).
  Driver::DeviceBufferRef keepAlive = bufRef.copy(); // refcount 2
  Driver::DeviceBuffer *observer = keepAlive.getPointer();
  void *data = observer->ptr();
  ASSERT_EQ(observer->getNumReferences(), 2u);

  // Surrender ONE reference net-zero, exactly like Mojo take_handle(): hand out
  // the raw pointer without dropping the ref (RCRef::release()).
  Driver::DeviceBuffer *handle = bufRef.release(); // refcount still 2
  ASSERT_EQ(handle, observer);
  ASSERT_EQ(observer->getNumReferences(), 2u);

  // Empty out-slot, as the runtime presents it to the pack site.
  AnyAsyncValueRef async;
  AsyncRTWrapper<AnyAsyncValueRef> wrapper{&async};
  MGP_RT_CreateAsyncDeviceBufferRefByTakingHandle(handle, data, kSize, wrapper);

  // takeRCRef must NOT addRef. copyRCRef here would make this 3 -- a leak ASAN
  // cannot see (it nets >= 1). This exact-count assertion is the guard.
  EXPECT_EQ(observer->getNumReferences(), 2u);

  // The slot now holds a real TensorBufferRef wrapping the adopted handle.
  ASSERT_TRUE(async.isReady());
  TensorBufferRef &tbr = async.get<TensorBufferRef>();
  EXPECT_EQ(tbr.getBuffer(), data);
  EXPECT_EQ(tbr.getSize(), kSize);

  // Dropping the slot releases exactly the one adopted reference.
  async.reset();
  EXPECT_EQ(observer->getNumReferences(), 1u); // only keepAlive remains

  // keepAlive drops at scope end -> refcount 0 -> allocation freed cleanly.
}

// Risk 6: byteSize == 0 takes the createBufferOwning(nullptr, 0, ...) path. The
// transfer + readback + free must tolerate null data / zero size.
TEST(MGPRTTakeHandle, ZeroSizeBuffer) {
  ASSIGN_NO_ERR(Driver::DeviceContextRef, ctx,
                Driver::DeviceContext::create(Driver::Device::cpuAPI));

  ASSIGN_NO_ERR(Driver::DeviceBufferRef, bufRef, ctx->enqueueCreateBuffer(0));
  ASSERT_TRUE(bufRef->isUnique());

  Driver::DeviceBufferRef keepAlive = bufRef.copy();
  Driver::DeviceBuffer *observer = keepAlive.getPointer();
  void *data = observer->ptr();
  ASSERT_EQ(observer->getNumReferences(), 2u);

  Driver::DeviceBuffer *handle = bufRef.release();

  AnyAsyncValueRef async;
  AsyncRTWrapper<AnyAsyncValueRef> wrapper{&async};
  MGP_RT_CreateAsyncDeviceBufferRefByTakingHandle(handle, data, 0, wrapper);

  EXPECT_EQ(observer->getNumReferences(), 2u);
  ASSERT_TRUE(async.isReady());
  TensorBufferRef &tbr = async.get<TensorBufferRef>();
  EXPECT_EQ(tbr.getBuffer(), data);
  EXPECT_EQ(tbr.getSize(), 0u);

  async.reset();
  EXPECT_EQ(observer->getNumReferences(), 1u);
}

} // namespace
