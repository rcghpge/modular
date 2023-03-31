//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_INTERFACES_LOWERABLETOMGPINTERFACE_TD
#define GENERICML_GRAPHCOMPILER_INTERFACES_LOWERABLETOMGPINTERFACE_TD

include "mlir/IR/OpBase.td"

def LowerableToMGPInterface :
    OpInterface<"LowerableToMGPInterface"> {
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
         * use the builders returned by getExecuteOpBuilder() and
           getInitOpBuilder() to insert the lowered ops into their appropriate
           location
         * update the lowering state using emplaceExecuteLoweredValue(...) for
           any new outputs they generate, and use get{Execute,Init}Chain() as
           needed
      }],
      "mlir::LogicalResult",
      "lower",
      (ins "M::MGP::MOGraphLoweringState &":$lowering)>,
  ];
}

#endif // GENERICML_GRAPHCOMPILER_INTERFACES_LOWERABLETOMGPINTERFACE_TD
