//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_INTERFACES_LOWERABLETOMOGGINTERFACE_TD
#define GENERICML_GRAPHCOMPILER_INTERFACES_LOWERABLETOMOGGINTERFACE_TD

include "mlir/IR/OpBase.td"

def LowerableToMOGGInterface :
    OpInterface<"LowerableToMOGGInterface"> {
  let description = [{
    This interface provides a way for `map` dialect ops to implement their own
    lowering into `mogg` codegen ops during the MAPToMOGG pass.
  }];

  let cppNamespace = "::M::MAP";

  let methods = [
    InterfaceMethod<
      [{
        Lower this op into mogg ops, using `state`'s builder and reading/
        recording value substitutions through `state.mapping`. A region-owning
        op recurses into its own body via `state.lowerRegionBody`.
      }],
      "mlir::LogicalResult",
      "lowerToMOGG",
      (ins "M::MAP::MAPToMOGGState &":$state)>,
  ];
}

#endif // GENERICML_GRAPHCOMPILER_INTERFACES_LOWERABLETOMOGGINTERFACE_TD
