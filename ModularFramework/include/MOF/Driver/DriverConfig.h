//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef MOF_DRIVER_DRIVERCONFIG_H
#define MOF_DRIVER_DRIVERCONFIG_H

#include "GenericML/Driver/DriverConfig.h"

namespace M::MOF {

class MOFDriverConfig : public DriverConfig {

public:
  MOFDriverConfig() : DriverConfig(OptionKind::kMOF, "default") {}

  static bool classof(const DriverConfig *config) {
    return config->getKind() == OptionKind::kMOF;
  }
};

} // namespace M::MOF

#endif // MOF_DRIVER_DRIVERCONFIG_H
