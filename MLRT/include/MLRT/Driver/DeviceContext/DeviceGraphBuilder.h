//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares the DeviceGraphBuilder interface for explicit device graph
// construction.
//
//===----------------------------------------------------------------------===//

#ifndef MLRT_DRIVER_DEVICECONTEXT_DEVICEGRAPHBUILDER_H
#define MLRT_DRIVER_DEVICECONTEXT_DEVICEGRAPHBUILDER_H

#include "MLRT/Driver/DeviceContext/DeviceContext.h"

#include "llvm/ADT/ArrayRef.h"

#include <optional>

// The graph builder drives legacy DeviceContext internals (DeviceFunction,
// device-graph capture) that the HAL DeviceContext does not expose; under
// --config=hal-device-context the whole builder compiles out.
#if !MODULAR_HAL_DEVICE_CONTEXT

namespace M::Driver {

class DeviceGraphBuilder;

/// Launch configuration for a single kernel-dispatch node.
struct DeviceFunctionLaunchParams {
  /// Grid dims
  uint32_t gridX = 0;
  uint32_t gridY = 0;
  uint32_t gridZ = 0;

  /// Block dims
  uint32_t blockX = 0;
  uint32_t blockY = 0;
  uint32_t blockZ = 0;

  uint32_t sharedMemBytes = 0;
  /// Backend-specific launch attributes (CUlaunchAttribute /
  /// hipLaunchAttribute).
  void *attrs = nullptr;
  uint32_t numAttrs = 0;
  /// Kernel arguments: array of pointers, one per argument.
  void **args = nullptr;
  uint32_t argCount = 0;
};

using DeviceGraphBuilderRef = RCRef<DeviceGraphBuilder>;

/// Opaque integer identifier for a node in an under-construction device graph.
using DeviceGraphNodeId = int32_t;

/// A lightweight handle identifying a node in an under-construction device
/// graph. Returned by DeviceGraphBuilder::addFunctionDirect(); can be passed
/// as a dependency when adding subsequent nodes to express the required
/// execution ordering. The dependency API will be added in a future revision.
struct DeviceGraphNode {
  DeviceGraphNodeId id;
};

/// Builder for explicit device graph construction.
///
/// Callers add individual typed nodes — kernels, memsets, memcopies,
/// allocations — and express dependency edges between them before
/// instantiating the graph.
///
/// Typical usage:
///
/// \code
///   auto builder = ctx->createGraphBuilder();
///   auto nodeA   = builder->addFunctionDirect(kernelA, paramsA);
///   auto nodeB   = builder->addFunctionDirect(kernelB, paramsB);
///   auto graph   = builder->instantiate();
///   graph->replay();
/// \endcode
class DeviceGraphBuilder : public ReferenceCounted<DeviceGraphBuilder> {
public:
  virtual ~DeviceGraphBuilder();

  /// Returns the DeviceContext that owns this graph builder.
  virtual DeviceContext &context() const = 0;

  /// Returns the handle of the most recently added node, or `std::nullopt`
  /// if no nodes have been added yet.
  virtual std::optional<DeviceGraphNode> lastNode() const = 0;

  // Each node-creation method below takes an explicit \p dependencies list
  // of DeviceGraphNode handles previously returned by this builder. An empty
  // ArrayRef makes the new node a graph root with no predecessors; a
  // non-empty ArrayRef uses those exact handles as predecessor edges. There
  // is no implicit "chain after the last node" behavior — callers must wire
  // dependencies explicitly.

  /// Add a kernel-dispatch node to the graph.
  ///
  /// \param func   The device function (kernel) to launch.
  /// \param params Grid/block dimensions, shared memory, and kernel arguments.
  /// \param dependencies Explicit dependency list; see above.
  ///
  /// \returns A node handle on success, or an error if the underlying driver
  ///          call fails.
  virtual ErrorOr<DeviceGraphNode>
  addFunctionDirect(const DeviceFunction &func,
                    const DeviceFunctionLaunchParams &params,
                    llvm::ArrayRef<DeviceGraphNode> dependencies) = 0;

  // Memcpy nodes — argument shapes mirror DeviceContext::enqueueCopy*.

  /// Add a host-to-device memcpy node.
  virtual ErrorOr<DeviceGraphNode>
  addCopyHostToDevice(void *dst, const void *src, size_t byteSize,
                      llvm::ArrayRef<DeviceGraphNode> dependencies) = 0;

  ErrorOr<DeviceGraphNode>
  addCopyHostToDevice(const DeviceBuffer &dst, const void *src,
                      llvm::ArrayRef<DeviceGraphNode> dependencies) {
    if (dst.context().device().api() != context().device().api())
      RETURN_ERROR("API mismatch between buffer and context");
    return addCopyHostToDevice(dst.ptr(), src, dst.byteSize(), dependencies);
  }

  /// Add a device-to-host memcpy node.
  virtual ErrorOr<DeviceGraphNode>
  addCopyDeviceToHost(void *dst, const void *src, size_t byteSize,
                      llvm::ArrayRef<DeviceGraphNode> dependencies) = 0;

  ErrorOr<DeviceGraphNode>
  addCopyDeviceToHost(void *dst, const DeviceBuffer &src,
                      llvm::ArrayRef<DeviceGraphNode> dependencies) {
    if (src.context().device().api() != context().device().api())
      RETURN_ERROR("API mismatch between buffer and context");
    return addCopyDeviceToHost(dst, src.ptr(), src.byteSize(), dependencies);
  }

  /// Add a device-to-device memcpy node. Both buffers must belong to a context
  /// using the same API as this builder; cross-context copies are not
  /// supported in graphs.
  ErrorOr<DeviceGraphNode>
  addCopyDeviceToDevice(DeviceBuffer &dst, DeviceBuffer &src,
                        llvm::ArrayRef<DeviceGraphNode> dependencies) {
    return addCopyDeviceToDevice(dst, src, 0, src.byteSize(), dependencies);
  }

  virtual ErrorOr<DeviceGraphNode>
  addCopyDeviceToDevice(DeviceBuffer &dst, DeviceBuffer &src,
                        size_t srcByteOffset, size_t srcByteSize,
                        llvm::ArrayRef<DeviceGraphNode> dependencies) = 0;

  // Memset nodes — argument shapes mirror DeviceContext::enqueueSetMemory.

  /// Add a memset node. \p valSize must be 1, 2, 4, or 8 bytes; for 8-byte
  /// patterns the high and low halves must match (the streaming form of
  /// `enqueueSetMemory` issues two ops in that case, which cannot be
  /// represented as a single graph node).
  virtual ErrorOr<DeviceGraphNode>
  addSetMemory(void *dst, size_t byteSize, uint64_t val, size_t valSize,
               llvm::ArrayRef<DeviceGraphNode> dependencies) = 0;

  ErrorOr<DeviceGraphNode>
  addSetMemory(const DeviceBuffer &dst, uint64_t val, size_t valSize,
               llvm::ArrayRef<DeviceGraphNode> dependencies) {
    if (dst.context().device().api() != context().device().api())
      RETURN_ERROR("API mismatch between buffer and context");
    return addSetMemory(dst.ptr(), dst.byteSize(), val, valSize, dependencies);
  }

  /// Add an empty (no-op) node to the graph. Empty nodes do no work at
  /// execution time and exist purely for transitive ordering: a single empty
  /// node fanned in from m predecessors and out to n successors expresses an
  /// m-to-n barrier with m+n edges instead of m*n edges, and can serve as a
  /// stable handle for "the completion of this phase" when the producer set
  /// is not visible to the consumer.
  ///
  /// \param dependencies Explicit dependency list; see above.
  virtual ErrorOr<DeviceGraphNode>
  addEmpty(llvm::ArrayRef<DeviceGraphNode> dependencies) = 0;

  /// Instantiate the currently constructed device graph into its executable
  /// format. Can be subsequently replayed multiple times.
  virtual ErrorOr<DeviceGraphRef> instantiate() = 0;

  /// Add an output to the device graph.
  void addOutput(AsyncRT::AnyAsyncValueRef output);

  /// Return the number of outputs registered on the device graph.
  size_t numOutputs() const { return outputs.size(); }

protected:
  /// Hands the recorded outputs to a subclass `instantiate()` so it can
  /// transfer their ownership to the instantiated `DeviceGraph`, whose stable
  /// backing memory must outlive the graph. Consumes the builder's outputs.
  llvm::SmallVector<AsyncRT::AnyAsyncValueRef> takeOutputs() {
    return std::move(outputs);
  }

private:
  /// A device graph can produce zero or more outputs whose backing memory needs
  /// to be tracked so that it can be referenced when executing the device
  /// graph.
  llvm::SmallVector<AsyncRT::AnyAsyncValueRef> outputs;
};

} // namespace M::Driver

#endif // MODULAR_HAL_DEVICE_CONTEXT

#endif // MLRT_DRIVER_DEVICECONTEXT_DEVICEGRAPHBUILDER_H
