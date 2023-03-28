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
public:
  cl::opt<std::string> cacheDir{
      "cache-dir",
      cl::desc("Path to CAS instance. If non-empty, large constants will be "
               "outlined into a CAS instance at the given path during "
               "compilation, and the compiled model will retrieve the "
               "constants from the same instance at model 'init' time. "
               "Default empty (ie disabled)."),
      cl::init("")};
};

} // namespace M::MO

#endif // GENERICML_GRAPHCOMPILER_GRAPHCOMPILERCLOPTIONS_H
