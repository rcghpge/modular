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

class APICLOptions : public CLOptionsBase {
public:
  using CLOptionsBase::CLOptionsBase;

  cl::list<std::string> paths{"paths", cl::desc("Paths of models")};
  cl::opt<std::string> cacheDir{"cache-dir",
                                cl::desc("Filesystem path of CAS cache"),
                                llvm::cl::Optional};
  cl::opt<size_t> num_threads{"num-threads", cl::desc("Number of threads"),
                              cl::init(0)};
  cl::list<TensorShape, bool, TensorShapeParser> inputShapes{
      "input-shapes", cl::desc("Specify input shapes. This is always required "
                               "and it must match the model's input shapes.")};
};

} // namespace M

#endif // GENERICML_API_APIVALIDATORCLOPTIONS_H
