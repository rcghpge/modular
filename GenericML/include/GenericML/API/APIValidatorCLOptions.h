//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
#ifndef GENERICML_API_APIVALIDATORCLOPTIONS_H
#define GENERICML_API_APIVALIDATORCLOPTIONS_H

#include "Support/CommonCLOptions.h"

namespace M {

class APICLOptions : public CLOptionsBase {
public:
  using CLOptionsBase::CLOptionsBase;

  cl::list<std::string> paths{"paths", cl::desc("Paths of models")};
  cl::opt<size_t> num_threads{"num-threads", cl::desc("Number of threads"),
                              cl::init(0)};
};

} // namespace M

#endif // GENERICML_API_APIVALIDATORCLOPTIONS_H
