//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This is a command line utility that reads a safetensor file and prints the
// metadata and tensor info to the console.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/Support/SafeTensor/SafeTensor.h"
#include "GraphCompiler/Driver/Device/Driver.h"
#include "Support/CommandLine.h"
#include "Support/CommonCLOptions.h"
#include "Support/RCRef.h"
#include "llvm/Support/raw_ostream.h"

#include <filesystem>

using namespace M;
using namespace M::GraphCompiler::Driver;

namespace {

/// Contains the core CLI options for *-executor, so that other executables
/// can use these without duplicating them.
class SafeTensorCLOptions : public CommonCLOptions {
public:
  /// Contains the core CLI options for safetensor.
  class Options : public CommonOptions {};

  Options &options;
  using CommonCLOptions::CommonCLOptions;

  SafeTensorCLOptions(int argc, char **argv, Options &opts,
                      bool skipInitLLVM = true)
      : CommonCLOptions(argc, argv, opts, skipInitLLVM), options(opts),
        SafeTensorOptionsCategory("SafeTensor command line options") {}

  llvm::cl::OptionCategory SafeTensorOptionsCategory;

  llvm::cl::opt<bool> showMetadata{
      "metadata", cl::desc("Display the metadata in the safetensor file"),
      llvm::cl::cat(SafeTensorOptionsCategory)};

  llvm::cl::opt<std::string> tensorInfo{
      "tensor-info", cl::desc("Display the tensor-info in the safetensor file"),
      llvm::cl::cat(SafeTensorOptionsCategory)};
};
} // namespace

static int reportError(const Twine &errorMessage) {
  llvm::errs() << "safetensor: " << errorMessage << "\n";
  return EXIT_FAILURE;
}

int main(int argc, char **argv) {
  SafeTensorCLOptions::Options opts;
  SafeTensorCLOptions cli(argc, argv, opts);
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "safetensor command line options\n");

  if (opts.inputFilename == "-")
    return reportError("stdin not supported, please specify an input file");

  auto device = M::GraphCompiler::Driver::makeCPUDevice();
  if (device.isError())
    return reportError(device.getError());
  ErrorOr<SafeTensor> safeTensorOr =
      SafeTensor::get(opts.inputFilename, RCRef<Device>::copy(&(**device)));
  if (safeTensorOr.isError())
    return reportError(safeTensorOr.getError());

  const auto &safeTensor = safeTensorOr.get();

  if (cli.showMetadata) {
    llvm::outs() << safeTensor.getMetadata() << "\n";
  } else if (!cli.tensorInfo.empty()) {
    ErrorOr<SafeTensor::TensorInfo> info =
        safeTensor.getTensorInfo(cli.tensorInfo);
    if (info.isError())
      return reportError(info.getError());

    llvm::outs() << info->toJSON() << "\n";
  } else {
    for (const auto &[key, value] : safeTensor.getTensorInfo()) {
      llvm::outs() << "tensor name: " << key << '\n';
      llvm::outs() << "value: " << value.toJSON() << '\n';
    }
  }

  return EXIT_SUCCESS;
}
