//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/DeviceDriver/Utils.h"

#include "GenericML/API/max/c/symbol_export.h"

using namespace M;
using namespace Driver;

MODULAR_API_EXPORT Status *M_newStatus() { return new Status{success()}; }

MODULAR_API_EXPORT bool M_isError(Status *status) {
  return status->errorOr.isError();
}

MODULAR_API_EXPORT const char *M_getError(Status *status) {
  if (!M_isError(status))
    return nullptr;
  return status->errorOr.getError();
}

MODULAR_API_EXPORT void M_deleteError(Status *status) { delete status; }
