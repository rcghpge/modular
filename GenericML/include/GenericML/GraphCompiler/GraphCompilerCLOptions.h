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

  std::string defaultDeviceLabel{"cpu"};

  size_t defaultDeviceId{0};

  bool enableAssignDevices{true};

  bool enableVerifyDeviceAssignment{false};

  bool enableCompiledRegionsFusion{true};

  bool debugPrintAllTensors{false};

  size_t softFusionNonHostKernelLimit{3};

  bool cacheBuffersAcrossExecutes{false};

  llvm::SmallVector<std::string> kernelLibraryPaths{};

  bool debugKGENStubs{false};

  void overrideLoweringOptions(MoToMgpOptions &moToMgpOptions) const {

    moToMgpOptions.cacheDir = cacheDir;
    moToMgpOptions.defaultDeviceLabel = defaultDeviceLabel;
    moToMgpOptions.defaultDeviceId = defaultDeviceId;
    moToMgpOptions.enableCompiledRegionsFusion = enableCompiledRegionsFusion;
    moToMgpOptions.debugPrintAllTensors = debugPrintAllTensors;
    moToMgpOptions.softFusionNonHostKernelLimit = softFusionNonHostKernelLimit;
    moToMgpOptions.kernelLibraryPaths = kernelLibraryPaths;
    moToMgpOptions.debugKGENStubs = debugKGENStubs;
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

  M::cl::MOpt<std::string, true> defaultDeviceLabel{
      "default-device-label",
      cl::desc(
          "If non-empty, assign as many ops as possible to the device "
          "with the given label (and device id based off default-device-id)."),
      llvm::cl::location(options.defaultDeviceLabel),
      llvm::cl::cat(GraphCompilerCLOptionsCategory)};

  M::cl::MOpt<size_t, true> defaultDeviceId{
      "default-device-id",
      cl::desc("If default-device-label is specified, this field specifies the"
               "default device id to target. Defaults to zero."),
      llvm::cl::location(options.defaultDeviceId),
      llvm::cl::cat(GraphCompilerCLOptionsCategory)};

  M::cl::MOpt<bool, true> enableAssignDevices{
      "enable-assign-devices",
      cl::desc("If true, enables the AssignDevicesPass which assigns ops to "
               "devices."),
      llvm::cl::location(options.enableAssignDevices),
      llvm::cl::cat(GraphCompilerCLOptionsCategory)};

  M::cl::MOpt<bool, true> enableVerifyDeviceAssignment{
      "enable-verify-device-assignment",
      cl::desc("If true, enables the VerifyDeviceAssignmentPass which verifies "
               "that all ops have been assigned to the correct device."),
      llvm::cl::location(options.enableVerifyDeviceAssignment),
      llvm::cl::cat(GraphCompilerCLOptionsCategory)};

  M::cl::MOpt<bool, true> enableCompiledRegionsFusion{
      "enable-compiled-regions-fusion",
      cl::desc("If false, will disable compiled region fusion"),
      llvm::cl::location(options.enableCompiledRegionsFusion),
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

  M::cl::MListOpt<std::string, llvm::SmallVector<std::string>>
      kernelLibraryPaths{
          "kernel-library-paths",
          cl::desc(
              "Paths to mojo packages containing mojo kernel definitions used "
              "to replace TF and MO ops."),
          llvm::cl::location(options.kernelLibraryPaths),
          llvm::cl::cat(GraphCompilerCLOptionsCategory)};

  M::cl::MOpt<bool, true> debugKGENStubs{
      "debug-kgen-stubs",
      cl::desc(
          "Build stubs with debuginfo so that they show up in the debugger."),
      llvm::cl::location(options.debugKGENStubs),
      llvm::cl::cat(GraphCompilerCLOptionsCategory)};
};

} // namespace M::MO

#endif // GENERICML_GRAPHCOMPILER_GRAPHCOMPILERCLOPTIONS_H
