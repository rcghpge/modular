//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements the HIP device graph builder.
//
//===----------------------------------------------------------------------===//

#include "HIPDeviceGraphBuilder.h"
#include "HIPDeviceContext.h"
#include "HIPDriver.h"
#include "MLRT/Driver/DeviceContext/ErrorMacros.h"

#include "llvm/ADT/Twine.h"

using namespace M;
using namespace M::Driver;

#define HIP_GRAPH_CHECK_ERR(hip_call)                                          \
  do {                                                                         \
    HIPDriver::Result _result_ = (hip_call);                                   \
    if (_result_ != HIPDriver::SUCCESS)                                        \
      RETURN_ERROR(llvm::Twine("HIP graph builder call failed with error "     \
                               "code ") +                                      \
                   llvm::Twine(static_cast<unsigned>(_result_)));              \
  } while (0)

// ===== HIPDeviceGraphBuilder ===== //

HIPDeviceGraphBuilder::HIPDeviceGraphBuilder(HIPDeviceContext &ctx,
                                             hipGraph_t graph)
    : ctx_(HIPDeviceContextRef::copy(&ctx)), graph_(graph) {}

HIPDeviceGraphBuilder::~HIPDeviceGraphBuilder() {
  if (!graph_)
    return;
  DeviceContextScope scope(*ctx_);
  ctx_->hipDriver().hipGraphDestroy(graph_);
  graph_ = nullptr;
}

ErrorOr<DeviceGraphBuilderRef>
HIPDeviceGraphBuilder::create(HIPDeviceContext &ctx) {
  DeviceContextScope scope(ctx);
  RETURN_IF_ERROR(scope.err());

  hipGraph_t graph = nullptr;
  HIP_GRAPH_CHECK_ERR(ctx.hipDriver().hipGraphCreate(&graph, 0));

  return HIPDeviceGraphBuilderRef::create(ctx, graph);
}

ErrorOr<DeviceGraphNode> HIPDeviceGraphBuilder::addFunctionDirect(
    const DeviceFunction &func, const DeviceFunctionLaunchParams &params) {
  DeviceContextScope scope(*ctx_);
  RETURN_IF_ERROR(scope.err());

  const auto &hipFunc = HIPDeviceFunction::cast(func);

  hipKernelNodeParams nodeParams = {
      .blockDim = {params.blockX, params.blockY, params.blockZ},
      .extra = nullptr,
      .func = reinterpret_cast<void *>(hipFunc.hipFunction()),
      .gridDim = {params.gridX, params.gridY, params.gridZ},
      .kernelParams = params.args,
      .sharedMemBytes = params.sharedMemBytes,
  };

  hipGraphNode_t node = nullptr;
  HIP_GRAPH_CHECK_ERR(ctx_->hipDriver().hipGraphAddKernelNode(
      &node, graph_, nullptr, 0, &nodeParams));

  DeviceGraphNodeId id = static_cast<DeviceGraphNodeId>(nodes_.size());
  nodes_.push_back(node);
  return DeviceGraphNode{id};
}

ErrorOr<DeviceGraphRef> HIPDeviceGraphBuilder::instantiate() {
  DeviceContextScope scope(*ctx_);
  RETURN_IF_ERROR(scope.err());

  hipGraph_t rawGraph = graph_;
  graph_ = nullptr;

  HIPDevice &hipDevice = HIPDevice::cast(ctx_->device());
  DEFINE_OR_RETURN_ERROR(DeviceStreamRef, stream, ctx_->stream());

  auto graph =
      HIPDeviceGraphRef::create(hipDevice, rawGraph, std::move(stream));
  RETURN_IF_ERROR(graph->instantiate());
  return DeviceGraphRef(std::move(graph));
}
