//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MGPDialect/RT/BinaryID.h"
#include "Support/BinaryID.h"

MODULAR_CXX_EXPORT std::string M::MGPRT::getMGPRTBinaryID() {
  // The ID of the shared library containing this code, so MEF caches are
  // invalidated when it changes.
  return M::getBinaryID();
}
