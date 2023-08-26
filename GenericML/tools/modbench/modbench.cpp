//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
#include <filesystem>

using namespace llvm;

struct ModBenchCLOptions {
  llvm::cl::opt<std::string> outputFilename{
      "o", llvm::cl::desc("Output filename"), llvm::cl::value_desc("filename"),
      llvm::cl::init("-")};
};

void writeReport(llvm::raw_ostream &os) {
  os << "This will one day be the modbench report!";
  os.flush();
}

static int reportError(Twine errorMessage) {
  llvm::errs() << "modbench: " << errorMessage << "\n";
  return EXIT_FAILURE;
}

int main(int argc, char **argv) {
  ModBenchCLOptions cli;

  llvm::cl::ParseCommandLineOptions(argc, argv, "ModBench Tool");
  auto outFilePathStr = cli.outputFilename.getValue();

  std::error_code ec;
  if (outFilePathStr != "-" && !std::filesystem::exists(outFilePathStr, ec) &&
      !ec) {
    auto outFilePath = std::filesystem::path(outFilePathStr);
    if (outFilePath.has_parent_path())
      std::filesystem::create_directories(outFilePath.parent_path(), ec);
  }

  // If anything failed, report the failure.
  if (ec) {
    exit(reportError("std::filesystem: " + ec.message() + ": " +
                     outFilePathStr));
  }

  std::error_code error;
  auto outputFile = std::make_unique<llvm::ToolOutputFile>(
      outFilePathStr, error, llvm::sys::fs::OF_None);
  if (error)
    exit(reportError("Cannot open output file: '" + outFilePathStr +
                     "': " + error.message()));

  auto &os = outputFile ? outputFile->os() : llvm::outs();
  writeReport(os);
  outputFile->keep();
  return EXIT_SUCCESS;
}
