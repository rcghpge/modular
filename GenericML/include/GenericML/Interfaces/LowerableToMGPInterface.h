//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_INTERFACES_LOWERABLETOMGPINTERFACE_H
#define GENERICML_INTERFACES_LOWERABLETOMGPINTERFACE_H

#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/DialectInterface.h"
#include "mlir/IR/OpDefinition.h"

namespace M::MGP {

class MOGraphLoweringState;

} // namespace M::MGP

#include "GenericML/Interfaces/LowerableToMGPInterface.h.inc"

#endif // GENERICML_INTERFACES_LOWERABLETOMGPINTERFACE_H
