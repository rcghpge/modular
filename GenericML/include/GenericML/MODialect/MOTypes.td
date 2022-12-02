//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This defines types in the MO dialect.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_MODIALECT_MOTYPES_TD
#define GENERICML_MODIALECT_MOTYPES_TD

include "GenericML/MODialect/MODialect.td"
include "Support/Compiler/TensorUtils.td"
include "mlir/IR/BuiltinTypeInterfaces.td"

// Base class for types in this dialect.
class MO_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<MODialect, name, traits> {
  let mnemonic = typeMnemonic;
}

//===----------------------------------------------------------------------===//
// MO_Tensor
//===----------------------------------------------------------------------===//

def MO_Tensor
    : MO_Type<"Tensor", "tensor",
              [DeclareTypeInterfaceMethods<ShapedTypeInterface>]> {
  // TODO: Relax the constraint on dynamic rank once we reach
  // runtime agreement with MGP team.
  let summary = "Type of tensor specs of statically known rank.";
  let description = [{
    This type represents the shape and element type of a tensor. Each
    dimension must be a strictly positive integer value, or the distinguished
    -1 dimension denoting unknown. The element type must be a builtin dtype,
    or built in none denoting unknown.

    This type does not yet support:
     - Dynamic ranks.

    Examples:
    ```mlir
    !mo.tensor<4x16xf32>
    !mo.tensor<?x?xnone>
    ```
  }];

  let parameters = (ins DimensionListParameter:$shape, "Type":$elementType);

  let genVerifyDecl = 1;
  let assemblyFormat = [{ `<` $shape `` $elementType `>` }];

  let builders = [
    TypeBuilderWithInferredContext<
      (ins "ArrayRef<int64_t>":$shape, "Type":$elementType),
      [{
        return $_get(elementType.getContext(), shape, elementType);
      }]>
  ];
  let extraClassDeclaration = [{
    size_t getRank() const;
  }];
}

#endif // GENERICML_MODIALECT_MOTYPES_TD
