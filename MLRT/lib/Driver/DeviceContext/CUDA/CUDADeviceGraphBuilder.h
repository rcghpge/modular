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
  CUDADeviceGraphBuilder(CUDADeviceContext &ctx, CUgraph graph,
                         DeviceGraphMemoryPoolRef pool = {});
  ~CUDADeviceGraphBuilder() override;

  /// Allocate a new empty CUgraph and return a builder for it.
  static ErrorOr<DeviceGraphBuilderRef>
  create(CUDADeviceContext &ctx, DeviceGraphMemoryPoolRef pool = {});

  DeviceContext &context() const override;

  std::optional<DeviceGraphNode> lastNode() const override;

  ErrorOr<DeviceGraphNode>
  addFunctionDirect(const DeviceFunction &func,
                    const DeviceFunctionLaunchParams &params,
                    llvm::ArrayRef<DeviceGraphNode> dependencies) override;

  using DeviceGraphBuilder::addCopyHostToDevice;
  ErrorOr<DeviceGraphNode>
  addCopyHostToDevice(void *dst, const void *src, size_t byteSize,
                      llvm::ArrayRef<DeviceGraphNode> dependencies) override;
  using DeviceGraphBuilder::addCopyDeviceToHost;
  ErrorOr<DeviceGraphNode>
  addCopyDeviceToHost(void *dst, const void *src, size_t byteSize,
                      llvm::ArrayRef<DeviceGraphNode> dependencies) override;
  using DeviceGraphBuilder::addCopyDeviceToDevice;
  ErrorOr<DeviceGraphNode>
  addCopyDeviceToDevice(DeviceBuffer &dst, DeviceBuffer &src,
                        size_t srcByteOffset, size_t srcByteSize,
                        llvm::ArrayRef<DeviceGraphNode> dependencies) override;
  ErrorOr<DeviceGraphNode>
  addSetMemory(void *dst, size_t byteSize, uint64_t val, size_t valSize,
               llvm::ArrayRef<DeviceGraphNode> dependencies) override;
  ErrorOr<DeviceGraphNode>
  addEmpty(llvm::ArrayRef<DeviceGraphNode> dependencies) override;

  ErrorOr<DeviceGraphRef> instantiate() override;

  CUgraph cudaGraph() const { return graph_; }

  /// Returns the raw CUgraphNode for a previously added node.
  CUgraphNode cudaNode(DeviceGraphNodeId id) const { return nodes_[id]; }

private:
  /// Resolves a user-provided dependency list into raw CUgraphNode handles.
  /// An empty `ArrayRef` produces no dependencies (graph root).
  ErrorOr<llvm::SmallVector<CUgraphNode>>
  resolveDeps(llvm::ArrayRef<DeviceGraphNode> deps) const;

  CUDADeviceContextRef ctx_;
  CUgraph graph_;
  llvm::SmallVector<CUgraphNode> nodes_;
};

} // namespace M::Driver

#endif // MLRT_LIB_DRIVER_DEVICECONTEXT_CUDA_CUDADEVICEGRAPHBUILDER_H
