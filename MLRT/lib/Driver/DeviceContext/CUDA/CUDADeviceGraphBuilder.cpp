//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements the CUDA device graph builder.
//
//===----------------------------------------------------------------------===//

#include "CUDADeviceGraphBuilder.h"
#include "CUDADeviceContext.h"
#include "CUDADriver.h"
#include "MLRT/Driver/DeviceContext/ErrorMacros.h"

#include "llvm/ADT/Twine.h"

using namespace M;
using namespace M::Driver;

// Local error-checking macro for CUDA driver results. Intentionally simpler
// than the full formatCUDAError() used in CUDADeviceContext.cpp; the numeric
// code is sufficient for graph-builder call sites.
#define CUDA_GRAPH_CHECK_ERR(cuda_call)                                        \
  do {                                                                         \
    CUDADriver::Result _result_ = (cuda_call);                                 \
    if (_result_ != CUDADriver::SUCCESS)                                       \
      RETURN_ERROR(llvm::Twine("CUDA graph builder call failed with error "    \
                               "code ") +                                      \
                   llvm::Twine(static_cast<unsigned>(_result_)));              \
  } while (0)

CUDADeviceGraphBuilder::CUDADeviceGraphBuilder(CUDADeviceContext &ctx,
                                               CUgraph graph,
                                               DeviceGraphMemoryPoolRef pool)
    : DeviceGraphBuilder(DeviceContextRef::copy(&ctx), std::move(pool)),
      ctx_(CUDADeviceContextRef::copy(&ctx)), graph_(graph) {}

CUDADeviceGraphBuilder::~CUDADeviceGraphBuilder() {
  if (!graph_)
    return;
  DeviceContextScope scope(*ctx_);
  CUDADeviceContext::cast(*ctx_).cudaDriver().cuGraphDestroy(graph_);
  graph_ = nullptr;
}

ErrorOr<DeviceGraphBuilderRef>
CUDADeviceGraphBuilder::create(CUDADeviceContext &ctx,
                               DeviceGraphMemoryPoolRef pool) {
  DeviceContextScope scope(ctx);
  RETURN_IF_ERROR(scope.err());

  CUgraph graph = nullptr;
  CUDA_GRAPH_CHECK_ERR(ctx.cudaDriver().cuGraphCreate(&graph, 0));

  return CUDADeviceGraphBuilderRef::create(ctx, graph, std::move(pool));
}

DeviceContext &CUDADeviceGraphBuilder::context() const { return *ctx_; }

std::optional<DeviceGraphNode> CUDADeviceGraphBuilder::lastNode() const {
  if (nodes_.empty())
    return std::nullopt;
  return DeviceGraphNode{static_cast<DeviceGraphNodeId>(nodes_.size() - 1)};
}

ErrorOr<llvm::SmallVector<CUgraphNode>> CUDADeviceGraphBuilder::resolveDeps(
    llvm::ArrayRef<DeviceGraphNode> deps) const {
  llvm::SmallVector<CUgraphNode> result;
  result.reserve(deps.size());
  for (const DeviceGraphNode &dep : deps) {
    if (dep.id < 0 || static_cast<size_t>(dep.id) >= nodes_.size())
      RETURN_ERROR(llvm::Twine("invalid DeviceGraphNode id ") +
                   llvm::Twine(dep.id));
    result.push_back(nodes_[dep.id]);
  }
  return result;
}

ErrorOr<DeviceGraphNode> CUDADeviceGraphBuilder::addFunctionDirect(
    const DeviceFunction &func, const DeviceFunctionLaunchParams &params,
    llvm::ArrayRef<DeviceGraphNode> dependencies) {
  DeviceContextScope scope(*ctx_);
  RETURN_IF_ERROR(scope.err());

  const auto &cudaFunc = CUDADeviceFunction::cast(func);

  CUDA_KERNEL_NODE_PARAMS_v2 nodeParams = {
      .func = cudaFunc.cudaFunction(),
      .gridDimX = params.gridX,
      .gridDimY = params.gridY,
      .gridDimZ = params.gridZ,
      .blockDimX = params.blockX,
      .blockDimY = params.blockY,
      .blockDimZ = params.blockZ,
      .sharedMemBytes = params.sharedMemBytes,
      .kernelParams = params.args,
      .extra = nullptr,
  };

  DEFINE_OR_RETURN_ERROR(llvm::SmallVector<CUgraphNode>, deps,
                         resolveDeps(dependencies));
  CUgraphNode node = nullptr;
  CUDA_GRAPH_CHECK_ERR(ctx_->cudaDriver().cuGraphAddKernelNode(
      &node, graph_, deps.data(), deps.size(), &nodeParams));

  // cuGraphAddKernelNode has no launch-attribute parameter, so attributes are
  // applied to the created node one by one. Only a subset of launch attributes
  // is valid on a kernel node; the rest describe stream-launch scheduling and
  // have no device-graph node equivalent:
  //  - The programmatic-dependent-launch family (PROGRAMMATIC_STREAM_
  //    SERIALIZATION, PROGRAMMATIC_EVENT, LAUNCH_COMPLETION_EVENT) expresses
  //    launch overlap. Graph dependency edges already order this node after
  //    its predecessors' full completion — strictly stronger than PDL's
  //    contract — so skipping them preserves correctness and only forgoes the
  //    overlap.
  //  - SYNCHRONIZATION_POLICY is a stream-scheduling hint with no node
  //    meaning.
  // An attribute outside the vendored enum is an error, not a silent drop.
  ArrayRef<CUlaunchAttribute_st> attrs(
      static_cast<const CUlaunchAttribute *>(params.attrs), params.numAttrs);

  for (const CUlaunchAttribute &attr : attrs) {
    switch (attr.id) {
    case CUlaunchAttributeID::CU_LAUNCH_ATTRIBUTE_IGNORE:
    case CUlaunchAttributeID::CU_LAUNCH_ATTRIBUTE_SYNCHRONIZATION_POLICY:
    case CUlaunchAttributeID::
        CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION:
    case CUlaunchAttributeID::CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_EVENT:
    case CUlaunchAttributeID::CU_LAUNCH_ATTRIBUTE_LAUNCH_COMPLETION_EVENT:
      continue;
    case CUlaunchAttributeID::CU_LAUNCH_ATTRIBUTE_ACCESS_POLICY_WINDOW:
    case CUlaunchAttributeID::CU_LAUNCH_ATTRIBUTE_COOPERATIVE:
    case CUlaunchAttributeID::CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION:
    case CUlaunchAttributeID::
        CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE:
    case CUlaunchAttributeID::CU_LAUNCH_ATTRIBUTE_PRIORITY:
    case CUlaunchAttributeID::CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN_MAP:
    case CUlaunchAttributeID::CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN:
    case CUlaunchAttributeID::CU_LAUNCH_ATTRIBUTE_DEVICE_UPDATABLE_KERNEL_NODE:
    case CUlaunchAttributeID::
        CU_LAUNCH_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT:
      CUDA_GRAPH_CHECK_ERR(ctx_->cudaDriver().cuGraphKernelNodeSetAttribute(
          node, attr.id, attr.value));
      continue;
    }
    RETURN_ERROR(llvm::Twine("unsupported launch attribute on a device-graph "
                             "kernel node: id ") +
                 llvm::Twine(static_cast<unsigned>(attr.id)));
  }

  DeviceGraphNodeId id = static_cast<DeviceGraphNodeId>(nodes_.size());
  nodes_.push_back(node);

  return DeviceGraphNode{id};
}

namespace {
// Build a 1D CUDA_MEMCPY3D_v2 covering \p byteSize bytes from \p src to \p dst
// with the given memory-type tags. Caller fills in src/dst pointer fields.
CUDA_MEMCPY3D_v2 make1DMemcpyParams(size_t byteSize) {
  CUDA_MEMCPY3D_v2 params = {};
  params.WidthInBytes = byteSize;
  params.Height = 1;
  params.Depth = 1;
  return params;
}
} // namespace

ErrorOr<DeviceGraphNode> CUDADeviceGraphBuilder::addCopyHostToDevice(
    void *dst, const void *src, size_t byteSize,
    llvm::ArrayRef<DeviceGraphNode> dependencies) {
  DeviceContextScope scope(*ctx_);
  RETURN_IF_ERROR(scope.err());

  CUDA_MEMCPY3D_v2 params = make1DMemcpyParams(byteSize);
  params.srcMemoryType = CU_MEMORYTYPE_HOST;
  params.srcHost = src;
  params.dstMemoryType = CU_MEMORYTYPE_DEVICE;
  params.dstDevice = dst;

  DEFINE_OR_RETURN_ERROR(llvm::SmallVector<CUgraphNode>, deps,
                         resolveDeps(dependencies));
  CUgraphNode node = nullptr;
  CUDA_GRAPH_CHECK_ERR(ctx_->cudaDriver().cuGraphAddMemcpyNode(
      &node, graph_, deps.data(), deps.size(), &params, ctx_->cudaContext()));

  DeviceGraphNodeId id = static_cast<DeviceGraphNodeId>(nodes_.size());
  nodes_.push_back(node);
  return DeviceGraphNode{id};
}

ErrorOr<DeviceGraphNode> CUDADeviceGraphBuilder::addCopyDeviceToHost(
    void *dst, const void *src, size_t byteSize,
    llvm::ArrayRef<DeviceGraphNode> dependencies) {
  DeviceContextScope scope(*ctx_);
  RETURN_IF_ERROR(scope.err());

  CUDA_MEMCPY3D_v2 params = make1DMemcpyParams(byteSize);
  params.srcMemoryType = CU_MEMORYTYPE_DEVICE;
  params.srcDevice = const_cast<void *>(src);
  params.dstMemoryType = CU_MEMORYTYPE_HOST;
  params.dstHost = dst;

  DEFINE_OR_RETURN_ERROR(llvm::SmallVector<CUgraphNode>, deps,
                         resolveDeps(dependencies));
  CUgraphNode node = nullptr;
  CUDA_GRAPH_CHECK_ERR(ctx_->cudaDriver().cuGraphAddMemcpyNode(
      &node, graph_, deps.data(), deps.size(), &params, ctx_->cudaContext()));

  DeviceGraphNodeId id = static_cast<DeviceGraphNodeId>(nodes_.size());
  nodes_.push_back(node);
  return DeviceGraphNode{id};
}

ErrorOr<DeviceGraphNode> CUDADeviceGraphBuilder::addCopyDeviceToDevice(
    DeviceBuffer &dst, DeviceBuffer &src, size_t srcByteOffset,
    size_t srcByteSize, llvm::ArrayRef<DeviceGraphNode> dependencies) {
  DeviceContextScope scope(*ctx_);
  RETURN_IF_ERROR(scope.err());

  // Graph DtoD nodes only support same-context copies; reject peer copies.
  const auto &srcCtx = CUDADeviceContext::cast(src.context());
  const auto &dstCtx = CUDADeviceContext::cast(dst.context());
  if (srcCtx.cudaContext() != ctx_->cudaContext() ||
      dstCtx.cudaContext() != ctx_->cudaContext()) {
    RETURN_ERROR("device-to-device graph copy requires src and dst buffers to "
                 "share the builder's CUDA context");
  }
  if (srcByteOffset + srcByteSize > src.byteSize()) {
    RETURN_ERROR("source range exceeds source buffer size");
  }
  if (srcByteSize != dst.byteSize()) {
    RETURN_ERROR("source and destination byte sizes must match");
  }

  CUDA_MEMCPY3D_v2 params = make1DMemcpyParams(srcByteSize);
  params.srcMemoryType = CU_MEMORYTYPE_DEVICE;
  params.srcDevice = static_cast<uint8_t *>(src.ptr()) + srcByteOffset;
  params.dstMemoryType = CU_MEMORYTYPE_DEVICE;
  params.dstDevice = dst.ptr();

  DEFINE_OR_RETURN_ERROR(llvm::SmallVector<CUgraphNode>, deps,
                         resolveDeps(dependencies));
  CUgraphNode node = nullptr;
  CUDA_GRAPH_CHECK_ERR(ctx_->cudaDriver().cuGraphAddMemcpyNode(
      &node, graph_, deps.data(), deps.size(), &params, ctx_->cudaContext()));

  DeviceGraphNodeId id = static_cast<DeviceGraphNodeId>(nodes_.size());
  nodes_.push_back(node);
  return DeviceGraphNode{id};
}

ErrorOr<DeviceGraphNode> CUDADeviceGraphBuilder::addSetMemory(
    void *dst, size_t byteSize, uint64_t val, size_t valSize,
    llvm::ArrayRef<DeviceGraphNode> dependencies) {
  DeviceContextScope scope(*ctx_);
  RETURN_IF_ERROR(scope.err());

  CUDA_MEMSET_NODE_PARAMS params = {};
  params.dst = dst;
  params.pitch = 0;
  params.height = 1;

  switch (valSize) {
  case 1:
    params.elementSize = 1;
    params.value = static_cast<uint8_t>(val);
    params.width = byteSize;
    break;
  case 2:
    if ((byteSize & 1) != 0)
      RETURN_ERROR("byteSize of buffer is misaligned");
    params.elementSize = 2;
    params.value = static_cast<uint16_t>(val);
    params.width = byteSize >> 1;
    break;
  case 4:
    if ((byteSize & 3) != 0)
      RETURN_ERROR("byteSize of buffer is misaligned");
    params.elementSize = 4;
    params.value = static_cast<uint32_t>(val);
    params.width = byteSize >> 2;
    break;
  case 8: {
    if ((byteSize & 7) != 0)
      RETURN_ERROR("byteSize of buffer is misaligned");
    uint32_t hiBits = static_cast<uint32_t>(val >> 32);
    uint32_t loBits = static_cast<uint32_t>(val & 0xfffffffful);
    if (hiBits != loBits) {
      RETURN_ERROR("8-byte memset with differing high/low halves cannot be "
                   "expressed as a single graph node");
    }
    params.elementSize = 4;
    params.value = loBits;
    params.width = byteSize >> 2;
    break;
  }
  default:
    RETURN_ERROR("unsupported valSize");
  }

  DEFINE_OR_RETURN_ERROR(llvm::SmallVector<CUgraphNode>, deps,
                         resolveDeps(dependencies));
  CUgraphNode node = nullptr;
  CUDA_GRAPH_CHECK_ERR(ctx_->cudaDriver().cuGraphAddMemsetNode(
      &node, graph_, deps.data(), deps.size(), &params, ctx_->cudaContext()));

  DeviceGraphNodeId id = static_cast<DeviceGraphNodeId>(nodes_.size());
  nodes_.push_back(node);
  return DeviceGraphNode{id};
}

ErrorOr<DeviceGraphNode>
CUDADeviceGraphBuilder::addEmpty(llvm::ArrayRef<DeviceGraphNode> dependencies) {
  DeviceContextScope scope(*ctx_);
  RETURN_IF_ERROR(scope.err());

  DEFINE_OR_RETURN_ERROR(llvm::SmallVector<CUgraphNode>, deps,
                         resolveDeps(dependencies));
  CUgraphNode node = nullptr;
  CUDA_GRAPH_CHECK_ERR(ctx_->cudaDriver().cuGraphAddEmptyNode(
      &node, graph_, deps.data(), deps.size()));

  DeviceGraphNodeId id = static_cast<DeviceGraphNodeId>(nodes_.size());
  nodes_.push_back(node);
  return DeviceGraphNode{id};
}

ErrorOr<DeviceGraphRef> CUDADeviceGraphBuilder::instantiate() {
  DeviceContextScope scope(*ctx_);
  RETURN_IF_ERROR(scope.err());

  CUgraph rawGraph = graph_;
  graph_ = nullptr;

  CUDADevice &cudaDevice = CUDADevice::cast(ctx_->device());
  DEFINE_OR_RETURN_ERROR(DeviceStreamRef, stream, ctx_->stream());

  auto graph =
      CUDADeviceGraphRef::create(cudaDevice, rawGraph, std::move(stream));
  RETURN_IF_ERROR(graph->instantiate());
  // Transfer the recorded inputs and outputs to the graph so their backing
  // memory outlives this builder.
  graph->setInputs(takeInputs());
  graph->setOutputs(takeOutputs());
  graph->setMemoryPool(takeMemoryPool());
  return DeviceGraphRef(std::move(graph));
}
