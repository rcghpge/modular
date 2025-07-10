//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
#ifndef GENERICML_API_APIVALIDATORCLOPTIONS_H
#define GENERICML_API_APIVALIDATORCLOPTIONS_H

#include "GenericML/Driver/DriverCLOptions.h"
#include "Support/CommonCLOptions.h"
#include "Support/ML/TensorShape.h"

namespace M {

class APIOptions : public OptionsBase {
public:
  SmallVector<std::string> paths{"paths"};
  std::string cacheDir{""};
  std::string crashDir{""};
  size_t numThreads{0};
  SmallVector<std::string> inputNames{};
  SmallVector<TensorShape> inputShapes{};
};
class APICLOptions : public CLOptionsBase {
public:
  APIOptions &options;
  APICLOptions(int argc, char **argv, APIOptions &o, bool skipInitLLVM = false)
      : CLOptionsBase(argc, argv, o, skipInitLLVM), options(o) {}

private:
  M::cl::MListOpt<std::string, SmallVector<std::string>> paths{
      "paths", cl::desc("Paths of models"), llvm::cl::location(options.paths)};
  M::cl::MOpt<std::string, true> cacheDir{
      "cache-dir", cl::desc("Filesystem path of CAS cache"), llvm::cl::Optional,
      llvm::cl::location(options.cacheDir)};
  M::cl::MOpt<std::string, true> crashDir{
      "crash-dir", cl::desc("Filesystem path of crash dump dir"),
      llvm::cl::Optional, llvm::cl::location(options.crashDir)};
  M::cl::MOpt<size_t, true> numThreads{"num-threads",
                                       cl::desc("Number of threads"),
                                       llvm::cl::location(options.numThreads)};

  M::cl::MListOpt<std::string, SmallVector<std::string>> inputNames{
      "input-names",
      cl::desc("Specify input names. This is always required "
               "and it must match the model's input names."),
      llvm::cl::location(options.inputNames)};

  M::cl::MListOpt<TensorShape, SmallVector<TensorShape>, TensorShapeParser>
      inputShapes{
          "input-shapes",
          cl::desc(
              "Specify input shapes. This is always required "
              "and it must match the model's input shapes. Provided shapes "
              "must correspond to the tensors denoted by input-names in that "
              "order."),
          llvm::cl::location(options.inputShapes)};
};

} // namespace M

#endif // GENERICML_API_APIVALIDATORCLOPTIONS_H
