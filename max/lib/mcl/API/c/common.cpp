//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "max/c/common.h"
#include "Config/Version.h"
#include "max/c/symbol_export.h"
#include "max/c/types.h"
#include "max/mcl/API/c/detail/types.h"

using namespace M;

MODULAR_API_EXPORT const char *M_version() { return getModularVersionString(); }

MODULAR_API_EXPORT M_Status *M_newStatus() {
  auto *status = new M_Status();
  status->status = success();
  return status;
}

MODULAR_API_EXPORT const char *M_getError(const M_Status *status) {
  auto &statusImpl = status->status;
  return statusImpl.getError();
}

MODULAR_API_EXPORT int M_isError(const M_Status *status) {
  auto &statusImpl = status->status;
  return statusImpl.isError();
}

MODULAR_API_EXPORT void M_freeStatus(M_Status *status) { delete status; }
