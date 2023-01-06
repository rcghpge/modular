//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_INTERFACES_MOLOWERINGINTERFACE_H
#define GENERICML_INTERFACES_MOLOWERINGINTERFACE_H

#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/DialectInterface.h"
#include "mlir/IR/OpDefinition.h"

namespace M::MGP {

class MOGraphLoweringState;

} // namespace M::MGP

#include "GenericML/Interfaces/MOLoweringInterface.h.inc"

#endif // GENERICML_INTERFACES_MOLOWERINGINTERFACE_H
