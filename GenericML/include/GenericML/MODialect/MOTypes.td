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
// MO_ShapeRefParameter
//===----------------------------------------------------------------------===//
// For standard ShapeRefs, which require allocation.
def ShapeParameter : ArrayRefParameter<"int64_t", "shape"> {
  let printer = [{printShapeParameter($_printer, $_self)}];
  let parser = [{parseShapeParameter($_parser)}];
}

//===----------------------------------------------------------------------===//
// MO_Tensor
//===----------------------------------------------------------------------===//

def MO_Tensor : MO_Type<"Tensor", "tensor", [ShapedTypeInterface]> {
  // TODO: Relax the constraint on dynamic rank once we reach
  // runtime agreement with MGP team.
  let summary = "Type of tensor specs of statically known rank.";
  let description = [{
    This type represents the shape and element type of a tensor. Each
    dimension must be a strictly positive integer value, the distinguished
    kDynamic dimension denoting unknown, or the distinguish kDynamicRank denoting 
    unkown rank. The element type must be a builtin dtype, or built in none
    denoting unknown.

    Examples:
    ```mlir
    !mo.tensor<4x16xf32>
    !mo.tensor<?x?xi32>
    !mo.tensor<*none>
    ```
  }];

  let parameters = (ins ShapeParameter : $shape, "Type" : $elementType);

  let genVerifyDecl = 1;
  let assemblyFormat = [{ `<` $shape `` $elementType `>` }];

  let builders = [TypeBuilderWithInferredContext<
      (ins "ArrayRef<int64_t>" : $shape, "Type" : $elementType),
      [{ return $_get(elementType.getContext(), shape, elementType); }]>];
  let extraClassDeclaration = [{
    size_t getRank() const;
    bool hasRank() const;
    ShapedType cloneWith(Optional<ArrayRef<int64_t>> shape, Type elementType)
        const;
  }];
}

#endif // GENERICML_MODIALECT_MOTYPES_TD
