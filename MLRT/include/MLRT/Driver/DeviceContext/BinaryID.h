//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef MLRT_DRIVER_DEVICECONTEXT_BINARYID_H
#define MLRT_DRIVER_DEVICECONTEXT_BINARYID_H

#include "Support/SymbolExport.h"
#include <string>

namespace M::AsyncRT {

MODULAR_CXX_EXPORT std::string getMojoBindingsBinaryID();

}

#endif // MLRT_DRIVER_DEVICECONTEXT_BINARYID_H
