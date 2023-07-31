//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_DRIVER_DRIVERCONFIG_H
#define GENERICML_DRIVER_DRIVERCONFIG_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"

namespace M {
class DriverConfig {
public:
  enum OptionKind { kTF, kTorch, kMOF };

  DriverConfig(OptionKind kind, llvm::StringLiteral name)
      : Kind(kind), configName(name) {}

  OptionKind getKind() const { return Kind; }

  llvm::StringLiteral getConfigName() const { return configName; }

private:
  const OptionKind Kind;
  const llvm::StringLiteral configName;
};
} // namespace M

#endif // GENERICML_DRIVER_DRIVERCONFIG_H
