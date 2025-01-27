//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/MGPRT/BinaryID.h"
#include "Support/BinaryID.h"

MODULAR_CXX_EXPORT std::string M::MGPRT::getMGPRTBinaryID() {
  return M::getBinaryID();
}
