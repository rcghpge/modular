//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_GRAPHCOMPILERCLOPTIONS_H
#define GENERICML_GRAPHCOMPILER_GRAPHCOMPILERCLOPTIONS_H

#include "Support/CommandLine.h"

namespace M::MO {

/// Collects command line options which influence M::MO::buildMoToMgpPipeline.
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
};

} // namespace M::MO

#endif // GENERICML_GRAPHCOMPILER_GRAPHCOMPILERCLOPTIONS_H
