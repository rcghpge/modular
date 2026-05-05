//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares the HIP device graph builder.
//
//===----------------------------------------------------------------------===//

#ifndef MLRT_LIB_DRIVER_DEVICECONTEXT_HIP_HIPDEVICEGRAPHBUILDER_H
#define MLRT_LIB_DRIVER_DEVICECONTEXT_HIP_HIPDEVICEGRAPHBUILDER_H

#include "HIPDeviceContext.h"
#include "MLRT/Driver/DeviceContext/DeviceGraphBuilder.h"
#include "llvm/ADT/SmallVector.h"

namespace M::Driver {

class HIPDeviceGraphBuilder;

using HIPDeviceGraphBuilderRef = RCRef<HIPDeviceGraphBuilder>;

/// HIP implementation of DeviceGraphBuilder.
///
/// Wraps a hipGraph_t under construction. Use create() to allocate a new
/// empty graph; each addFunctionDirect() call appends one kernel node.
class HIPDeviceGraphBuilder : public DeviceGraphBuilder {
public:
  /// Construct a builder that takes ownership of an already-created \p graph.
  /// Prefer create() to allocate both together.
  HIPDeviceGraphBuilder(HIPDeviceContext &ctx, hipGraph_t graph);
  ~HIPDeviceGraphBuilder() override;

  /// Allocate a new empty hipGraph_t and return a builder for it.
  static ErrorOr<DeviceGraphBuilderRef> create(HIPDeviceContext &ctx);

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

  hipGraph_t hipGraph() const { return graph_; }

  /// Returns the raw hipGraphNode_t for a previously added node.
  hipGraphNode_t hipNode(DeviceGraphNodeId id) const { return nodes_[id]; }

private:
  /// Dependency list to chain a new node after the previously added one,
  /// matching the sequential ordering produced by stream capture. Returns an
  /// empty range when no nodes have been added yet (the new node becomes a
  /// graph root).
  llvm::ArrayRef<hipGraphNode_t> sequentialDeps() const {
    if (nodes_.empty())
      return {};
    return llvm::ArrayRef<hipGraphNode_t>(&nodes_.back(), 1);
  }

  HIPDeviceContextRef ctx_;
  hipGraph_t graph_;
  llvm::SmallVector<hipGraphNode_t> nodes_;
};

} // namespace M::Driver

#endif // MLRT_LIB_DRIVER_DEVICECONTEXT_HIP_HIPDEVICEGRAPHBUILDER_H
