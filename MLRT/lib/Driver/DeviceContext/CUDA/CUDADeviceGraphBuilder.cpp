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
                                               CUgraph graph)
    : ctx_(CUDADeviceContextRef::copy(&ctx)), graph_(graph) {}

CUDADeviceGraphBuilder::~CUDADeviceGraphBuilder() {
  if (!graph_)
    return;
  DeviceContextScope scope(*ctx_);
  CUDADeviceContext::cast(*ctx_).cudaDriver().cuGraphDestroy(graph_);
  graph_ = nullptr;
}

ErrorOr<DeviceGraphBuilderRef>
CUDADeviceGraphBuilder::create(CUDADeviceContext &ctx) {
  DeviceContextScope scope(ctx);
  RETURN_IF_ERROR(scope.err());

  CUgraph graph = nullptr;
  CUDA_GRAPH_CHECK_ERR(ctx.cudaDriver().cuGraphCreate(&graph, 0));

  return CUDADeviceGraphBuilderRef::create(ctx, graph);
}

ErrorOr<DeviceGraphNode> CUDADeviceGraphBuilder::addFunctionDirect(
    const DeviceFunction &func, const DeviceFunctionLaunchParams &params) {
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

  CUgraphNode node = nullptr;
  CUDA_GRAPH_CHECK_ERR(ctx_->cudaDriver().cuGraphAddKernelNode(
      &node, graph_, nullptr, 0, &nodeParams));

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
  return DeviceGraphRef(std::move(graph));
}
