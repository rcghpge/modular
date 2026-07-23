//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/Driver/Device/Utils.h"
#include "Support/SymbolExport.h"

using namespace M;
using namespace M::GraphCompiler::Driver;

MODULAR_DRIVER_EXPORT Status *M_newStatus() { return new Status(); }

MODULAR_DRIVER_EXPORT bool M_isError(Status *status) {
  return status->errorOr.isError();
}

MODULAR_DRIVER_EXPORT const char *M_getError(Status *status) {
  if (!M_isError(status))
    return nullptr;
  return status->errorOr.getError();
}

MODULAR_DRIVER_EXPORT void M_deleteStatus(Status *status) { delete status; }
