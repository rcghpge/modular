//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/MOGGDialect/Support/EmitMojoState.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGInterfaces.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGOps.h"

using namespace M;

namespace M::MOGG {

LogicalResult EmitMojoState::lowerOpToMojo(Operation *op) {
  if (auto emitMojoInterface = dyn_cast<MOGG::EmitsMojoInterface>(op)) {
    if (emitMojoInterface.emitMojo(*this).failed()) {
      return emitMojoInterface->emitError(
          "Could not lower this operation to Mojo");
    }
    return success();
  } else if (isa<MOGG::OutputOp>(op)) {
    // These ops don't need to be converted.
    return success();
  } else {
    return op->emitError("This operation doesn't implement EmitsMojoInterface");
  }
}

} // namespace M::MOGG
