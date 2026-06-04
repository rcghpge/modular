//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef TARGET_LIB_HIP_DRIVER_H
#define TARGET_LIB_HIP_DRIVER_H

#include "Error.h"
#include "HIPApi.h"
#include "Target/lib/Support/Utils.h"

using M::Target::CustomStatus;

struct M_driver_driver {
private:
  HIPApi rawHipApi;

public:
  CustomStatus status;

  M_driver_status_t initStatus = M_DRIVER_STATUS_SUCCESS;

  [[nodiscard]] static DriverResult<M_driver_driver_t> create();

  static M_driver_status_t destroy(M_driver_driver_t driver) {
    delete driver;
    return M_DRIVER_STATUS_SUCCESS;
  }

  M_driver_driver(HIPApi &&rawHipApi)
      : rawHipApi(std::move(rawHipApi)),
        status(M_DRIVER_NUM_CUSTOM_STATUS_CODES) {
    status.messages[M_DRIVER_STATUS_UNIMPLEMENTED] = "Unimplemented";
  }

  ~M_driver_driver() { rawHipApi.unload(); }

  HIPApi &hip() { return rawHipApi; }

  DriverResult<int64_t> deviceCount();
};

#endif // TARGET_LIB_HIP_DRIVER_H
