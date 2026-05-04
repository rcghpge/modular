//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares the CUDA device graph builder.
//
//===----------------------------------------------------------------------===//

#ifndef MLRT_LIB_DRIVER_DEVICECONTEXT_CUDA_CUDADEVICEGRAPHBUILDER_H
#define MLRT_LIB_DRIVER_DEVICECONTEXT_CUDA_CUDADEVICEGRAPHBUILDER_H

#include "CUDADeviceContext.h"
#include "MLRT/Driver/DeviceContext/DeviceGraphBuilder.h"
#include "llvm/ADT/SmallVector.h"

namespace M::Driver {

class CUDADeviceGraphBuilder;

using CUDADeviceGraphBuilderRef = RCRef<CUDADeviceGraphBuilder>;

/// CUDA implementation of DeviceGraphBuilder.
///
/// Wraps a CUgraph under construction. Use create() to allocate a new empty
/// graph; each addFunctionDirect() call appends one kernel node.
class CUDADeviceGraphBuilder : public DeviceGraphBuilder {
public:
  /// Construct a builder that takes ownership of an already-created \p graph.
  /// Prefer create() to allocate both together.
  CUDADeviceGraphBuilder(CUDADeviceContext &ctx, CUgraph graph);
  ~CUDADeviceGraphBuilder() override;

  /// Allocate a new empty CUgraph and return a builder for it.
  static ErrorOr<DeviceGraphBuilderRef> create(CUDADeviceContext &ctx);

  DeviceContext &context() const override;

  ErrorOr<DeviceGraphNode>
  addFunctionDirect(const DeviceFunction &func,
                    const DeviceFunctionLaunchParams &params) override;

  using DeviceGraphBuilder::addCopyHostToDevice;
  ErrorOr<DeviceGraphNode> addCopyHostToDevice(void *dst, const void *src,
                                               size_t byteSize) override;
  using DeviceGraphBuilder::addCopyDeviceToHost;
  ErrorOr<DeviceGraphNode> addCopyDeviceToHost(void *dst, const void *src,
                                               size_t byteSize) override;
  using DeviceGraphBuilder::addCopyDeviceToDevice;
  ErrorOr<DeviceGraphNode> addCopyDeviceToDevice(DeviceBuffer &dst,
                                                 DeviceBuffer &src,
                                                 size_t srcByteOffset,
                                                 size_t srcByteSize) override;
  ErrorOr<DeviceGraphNode> addSetMemory(void *dst, size_t byteSize,
                                        uint64_t val, size_t valSize) override;

  ErrorOr<DeviceGraphRef> instantiate() override;

  CUgraph cudaGraph() const { return graph_; }

  /// Returns the raw CUgraphNode for a previously added node.
  CUgraphNode cudaNode(DeviceGraphNodeId id) const { return nodes_[id]; }

private:
  CUDADeviceContextRef ctx_;
  CUgraph graph_;
  llvm::SmallVector<CUgraphNode> nodes_;
};

} // namespace M::Driver

#endif // MLRT_LIB_DRIVER_DEVICECONTEXT_CUDA_CUDADEVICEGRAPHBUILDER_H
