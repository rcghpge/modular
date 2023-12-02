//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_GRAPHCOMPILERCLOPTIONS_H
#define GENERICML_GRAPHCOMPILER_GRAPHCOMPILERCLOPTIONS_H

#include "GenericML/GraphCompiler/Passes.h"
#include "Support/CommandLine.h"

namespace M::MO {

/// Collects mt command line options which influence
/// M::MO::buildMoToMgpPipeline.
///
/// See also MoToMgpOptions for the corresponding MLIR xxx-opt command line
/// options.
class GraphCompilerCLOptions {
private:
  llvm::cl::OptionCategory GraphCompilerCLOptionsCategory{
      "Graph compiler options"};

public:
  cl::opt<std::string> cacheDir{
      "cache-dir",
      cl::desc("Path to CAS instance. If non-empty, large constants will be "
               "outlined into a CAS instance at the given path during "
               "compilation, and the compiled model will retrieve the "
               "constants from the same instance at model 'init' time. "
               "Default empty (ie disabled)."),
      cl::init(""), llvm::cl::cat(GraphCompilerCLOptionsCategory)};

  cl::opt<bool> enableExtraDevices{
      "enable-extra-devices",
      cl::desc("If true, account for all host devices during compilation, "
               "including any host GPUs in addition to the CPU. The resulting "
               "model will only be runnable on a similarly provisioned host. "
               "Default is false, which will include just the host CPU."),
      cl::init(false), llvm::cl::cat(GraphCompilerCLOptionsCategory)};

  cl::opt<bool> enableFakeCudaDevice{
      "enable-fake-cuda-device",
      cl::desc("If true, compile assuming a single CUDA GPU is available. "
               "For testing only."),
      cl::init(false), llvm::cl::cat(GraphCompilerCLOptionsCategory)};

  cl::opt<std::string> defaultDeviceLabel{
      "default-device-label",
      cl::desc("If non-empty, assign as many ops as possible to the device "
               "with the given label (and device id zero)."),
      cl::init(""), llvm::cl::cat(GraphCompilerCLOptionsCategory)};

  cl::opt<bool> enableMemoryPlanning{
      "enable-memory-planning",
      cl::desc("If true, statically plan as many buffer allocations as "
               "possible."),
      cl::init(false), llvm::cl::cat(GraphCompilerCLOptionsCategory)};

  cl::opt<bool> genVizSpec{
      "gen-viz-spec",
      cl::desc(
          "If true, generates a visualization spec compatible with Netron."),
      cl::init(false), llvm::cl::cat(GraphCompilerCLOptionsCategory)};

  cl::opt<std::string> vizOutputDir{
      "viz-output-dir",
      cl::desc("If genVizSpec, can be used to specify the output directory for "
               "the visualization spec."),
      cl::init(""), llvm::cl::cat(GraphCompilerCLOptionsCategory)};

  cl::opt<bool> debugPrintAllTensors{
      "debug-print-all-tensors",
      cl::desc("If enabled, print all intermediate tensors. The usual "
               "'result-output-style', 'print-precision' and "
               "'binary-output-dir' flags may be used to control the "
               "printed format and saved location."),
      cl::init(false), llvm::cl::cat(GraphCompilerCLOptionsCategory)};

  cl::opt<size_t> softFusionNonHostKernelLimit{
      "soft-fusion-non-host-kernel-limit",
      cl::desc(
          "Maximum number of non-host kernels to place in a single soft fused "
          "kernel."),
      cl::init(3), llvm::cl::cat(GraphCompilerCLOptionsCategory)};

  cl::opt<bool> cacheBuffersAcrossExecutes{
      "cache-buffers-across-executes",
      cl::desc("If true, allow all buffer allocations to be cached and reused "
               "across calls to the model's 'execute'. If a buffer's size "
               "changes between calls due to dynamic shapes then the buffer "
               "will be reallocated."),
      cl::init(false), llvm::cl::cat(GraphCompilerCLOptionsCategory)};

  cl::opt<size_t> minPinnedHostBufferSize{
      "min-pinned-host-buffer-size",
      llvm::cl::desc(
          "If non-zero, replace host-side allocations with pinned allocations "
          "when the buffer is at least this size in bytes, and is the source "
          "or target of a host/device memory transfer."),
      llvm::cl::init(0)};

  cl::list<std::string> kernelLibraryPaths{
      "kernel-library-paths",
      cl::desc("Paths to mojo packages containing mojo kernel definitions used "
               "to replace TF and MO ops."),
      llvm::cl::cat(GraphCompilerCLOptionsCategory)};

  void overrideLoweringOptions(MoToMgpOptions &moToMgpOptions) const {
    moToMgpOptions.cacheDir = cacheDir.getValue();
    moToMgpOptions.enableExtraDevices = enableExtraDevices.getValue();
    moToMgpOptions.enableFakeCudaDevice = enableFakeCudaDevice.getValue();
    moToMgpOptions.defaultDeviceLabel = defaultDeviceLabel.getValue();
    moToMgpOptions.enableMemoryPlanning = enableMemoryPlanning.getValue();
    moToMgpOptions.genVizSpec = genVizSpec.getValue();
    moToMgpOptions.vizOutputDir = vizOutputDir.getValue();
    moToMgpOptions.debugPrintAllTensors = debugPrintAllTensors.getValue();
    moToMgpOptions.softFusionNonHostKernelLimit =
        softFusionNonHostKernelLimit.getValue();
    moToMgpOptions.cacheBuffersAcrossExecutes =
        cacheBuffersAcrossExecutes.getValue();
    moToMgpOptions.minPinnedHostBufferSize = minPinnedHostBufferSize.getValue();
    moToMgpOptions.kernelLibraryPaths = kernelLibraryPaths;
  }
};

} // namespace M::MO

#endif // GENERICML_GRAPHCOMPILER_GRAPHCOMPILERCLOPTIONS_H
