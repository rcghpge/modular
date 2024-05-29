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
class GraphCompilerOptions {

public:
  std::string cacheDir{""};

  bool enableExtraDevices{false};

  bool enableFakeCudaDevice{false};

  std::string defaultDeviceLabel{""};

  bool enableMemoryPlanning{false};

  bool lowerToCompiledRegions{true};

  bool enableCompiledRegionsFusion{true};

  bool genVizSpec{false};

  std::string vizOutputPath{""};

  bool debugPrintAllTensors{false};

  size_t softFusionNonHostKernelLimit{3};

  bool cacheBuffersAcrossExecutes{false};

  size_t minPinnedHostBufferSize{0};

  llvm::SmallVector<std::string> kernelLibraryPaths{};

  void overrideLoweringOptions(MoToMgpOptions &moToMgpOptions) const {

    moToMgpOptions.cacheDir = cacheDir;
    moToMgpOptions.enableExtraDevices = enableExtraDevices;
    moToMgpOptions.enableFakeCudaDevice = enableFakeCudaDevice;
    moToMgpOptions.defaultDeviceLabel = defaultDeviceLabel;
    moToMgpOptions.enableMemoryPlanning = enableMemoryPlanning;
    moToMgpOptions.lowerToCompiledRegions = lowerToCompiledRegions;
    moToMgpOptions.enableCompiledRegionsFusion = enableCompiledRegionsFusion;
    moToMgpOptions.genVizSpec = genVizSpec;
    moToMgpOptions.vizOutputPath = vizOutputPath;
    moToMgpOptions.debugPrintAllTensors = debugPrintAllTensors;
    moToMgpOptions.softFusionNonHostKernelLimit = softFusionNonHostKernelLimit;
    moToMgpOptions.minPinnedHostBufferSize = minPinnedHostBufferSize;
    moToMgpOptions.kernelLibraryPaths = kernelLibraryPaths;
  }
};

class GraphCompilerCLOptions {
public:
  GraphCompilerOptions &options;
  GraphCompilerCLOptions(GraphCompilerOptions &opts) : options(opts) {}

private:
  llvm::cl::OptionCategory GraphCompilerCLOptionsCategory{
      "Graph compiler options"};

  M::cl::MOpt<std::string, true> cacheDir{
      "cache-dir",
      cl::desc("Path to CAS instance. If non-empty, large constants will be "
               "outlined into a CAS instance at the given path during "
               "compilation, and the compiled model will retrieve the "
               "constants from the same instance at model 'init' time. "
               "Default empty (ie disabled)."),
      llvm::cl::location(options.cacheDir),
      llvm::cl::cat(GraphCompilerCLOptionsCategory)};

  M::cl::MOpt<bool, true> enableExtraDevices{
      "enable-extra-devices",
      cl::desc("If true, account for all host devices during compilation, "
               "including any host GPUs in addition to the CPU. The resulting "
               "model will only be runnable on a similarly provisioned host. "
               "Default is false, which will include just the host CPU."),
      llvm::cl::location(options.enableExtraDevices),
      llvm::cl::cat(GraphCompilerCLOptionsCategory)};

  M::cl::MOpt<bool, true> enableFakeCudaDevice{
      "enable-fake-cuda-device",
      cl::desc("If true, compile assuming a single CUDA GPU is available. "
               "For testing only."),
      llvm::cl::location(options.enableFakeCudaDevice),
      llvm::cl::cat(GraphCompilerCLOptionsCategory)};

  M::cl::MOpt<std::string, true> defaultDeviceLabel{
      "default-device-label",
      cl::desc("If non-empty, assign as many ops as possible to the device "
               "with the given label (and device id zero)."),
      llvm::cl::location(options.defaultDeviceLabel),
      llvm::cl::cat(GraphCompilerCLOptionsCategory)};

  M::cl::MOpt<bool, true> enableMemoryPlanning{
      "enable-memory-planning",
      cl::desc("If true, statically plan as many buffer allocations as "
               "possible."),
      llvm::cl::location(options.enableMemoryPlanning),
      llvm::cl::cat(GraphCompilerCLOptionsCategory)};

  M::cl::MOpt<bool, true> lowerToCompiledRegions{
      "lower-to-compiled-regions",
      cl::desc("If true, will lower MGP Ops down to compiled region."),
      llvm::cl::location(options.lowerToCompiledRegions),
      llvm::cl::cat(GraphCompilerCLOptionsCategory)};

  M::cl::MOpt<bool, true> enableCompiledRegionsFusion{
      "enable-compiled-regions-fusion",
      cl::desc("If false, will disable compiled region fusion"),
      llvm::cl::location(options.enableCompiledRegionsFusion),
      llvm::cl::cat(GraphCompilerCLOptionsCategory)};

  M::cl::MOpt<bool, true> genVizSpec{
      "gen-viz-spec",
      cl::desc(
          "If true, generates a visualization spec compatible with Netron."),
      llvm::cl::location(options.genVizSpec),
      llvm::cl::cat(GraphCompilerCLOptionsCategory)};

  M::cl::MOpt<std::string, true> vizOutputPath{
      "viz-output-path",
      cl::desc("If genVizSpec, can be used to specify the output path for "
               "the visualization spec."),
      llvm::cl::location(options.vizOutputPath),
      llvm::cl::cat(GraphCompilerCLOptionsCategory)};

  M::cl::MOpt<bool, true> debugPrintAllTensors{
      "debug-print-all-tensors",
      cl::desc("If enabled, print all intermediate tensors. The usual "
               "'result-output-style', 'print-precision' and "
               "'binary-output-dir' flags may be used to control the "
               "printed format and saved location."),
      llvm::cl::location(options.debugPrintAllTensors),
      llvm::cl::cat(GraphCompilerCLOptionsCategory)};

  M::cl::MOpt<size_t, true> softFusionNonHostKernelLimit{
      "soft-fusion-non-host-kernel-limit",
      cl::desc(
          "Maximum number of non-host kernels to place in a single soft fused "
          "kernel."),
      llvm::cl::location(options.softFusionNonHostKernelLimit),
      llvm::cl::cat(GraphCompilerCLOptionsCategory)};

  M::cl::MOpt<size_t, true> minPinnedHostBufferSize{
      "min-pinned-host-buffer-size",
      llvm::cl::desc(
          "If non-zero, replace host-side allocations with pinned allocations "
          "when the buffer is at least this size in bytes, and is the source "
          "or target of a host/device memory transfer."),
      llvm::cl::location(options.minPinnedHostBufferSize)};

  M::cl::MListOpt<std::string, llvm::SmallVector<std::string>>
      kernelLibraryPaths{
          "kernel-library-paths",
          cl::desc(
              "Paths to mojo packages containing mojo kernel definitions used "
              "to replace TF and MO ops."),
          llvm::cl::location(options.kernelLibraryPaths),
          llvm::cl::cat(GraphCompilerCLOptionsCategory)};
};

} // namespace M::MO

#endif // GENERICML_GRAPHCOMPILER_GRAPHCOMPILERCLOPTIONS_H
