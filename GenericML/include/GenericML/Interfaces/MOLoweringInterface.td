//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_INTERFACES_MOLOWERINGINTERFACE_TD
#define GENERICML_INTERFACES_MOLOWERINGINTERFACE_TD

include "mlir/IR/OpBase.td"

def MOLoweringInterface :
    OpInterface<"MOLoweringInterface"> {
  let description = [{
    This interface provides a way for dialects to implement methods needed to
    lower themselves during the MOToPrimitives pass.
  }];

  let cppNamespace = "::M::MGP";

  let methods = [
    InterfaceMethod<
      [{
        Lower this op into primitives compatible with MGP. Implementers make
        use of the shared lowering state object, and mutate it appropriately. In
        particular, they need to:
         * call lowering.execute.builder and lowering.init.builder to insert
           the lowered ops into their appropriate location
         * update lowering.execute.loweredTensors for any new outputs they
           generate, and/or lowering.execute.chain / lowering.init.chain, if
           needed
      }],
      "mlir::LogicalResult",
      "lower",
      (ins "M::MGP::MOGraphLoweringState &":$lowering)>,
  ];
}

#endif // GENERICML_INTERFACES_MOLOWERINGINTERFACE_TD
