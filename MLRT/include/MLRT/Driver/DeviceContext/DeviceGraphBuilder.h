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

  /// Add a kernel-dispatch node to the graph.
  ///
  /// \param func   The device function (kernel) to launch.
  /// \param params Grid/block dimensions, shared memory, and kernel arguments.
  ///
  /// \returns A node handle on success, or an error if the underlying driver
  ///          call fails.
  virtual ErrorOr<DeviceGraphNode>
  addFunctionDirect(const DeviceFunction &func,
                    const DeviceFunctionLaunchParams &params) = 0;

  /// Instantiate the currently constructed device graph into its executable
  /// format. Can be subsequently replayed multiple times.
  virtual ErrorOr<DeviceGraphRef> instantiate() = 0;
};

} // namespace M::Driver

#endif // MLRT_DRIVER_DEVICECONTEXT_DEVICEGRAPHBUILDER_H
