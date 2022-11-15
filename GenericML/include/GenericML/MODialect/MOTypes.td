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
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"

// Base class for types in this dialect.
class MO_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<MODialect, name, traits> {
  let mnemonic = typeMnemonic;
}

//===----------------------------------------------------------------------===//
// MO_TensorSpec
//===----------------------------------------------------------------------===//

// TODO: this is a duplicate of the one in MTFLTypes.td
def DimensionListParameter : ArrayRefParameter<"int64_t", "shape"> {
  let printer = [{
    for (int64_t dim : $_self) {
      if (ShapedType::isDynamic(dim))
        $_printer << '?';
      else
        $_printer << dim;
      $_printer << 'x';
    }
  }];
  let parser = [{
    [&]() -> FailureOr<llvm::SmallVector<int64_t>> {
        llvm::SmallVector<int64_t> dims;
        if ($_parser.parseDimensionList(dims, /*allowDynamic=*/true))
          return failure();

        return dims;
      }
    ()
  }];
}

def MO_TensorSpec
    : MO_Type<"TensorSpec", "spec",
              [DeclareTypeInterfaceMethods<ShapedTypeInterface>]> {
  // TODO: Relax the constraint on dynamic rank once we reach
  // runtime agreement with MOP team.
  let summary = "Type of tensor specs of statically known rank.";
  let description = [{
    This type represents the shape and element type of a tensor. Each
    dimension must be a strictly positive integer value, or the distinguished
    -1 dimension denoting unknown. The element type must be a builtin dtype,
    or built in none denoting unknown.

    This type does not support:
     - Dynamic ranks.

    Tensor specs are lowered to MOP::TensorSpec in the Modular runtime.
    The runtime value will agree with the rank, known shape dimensions,
    and known element type of the tensor spec type.

    Examples:
    ```mlir
    !mo.spec<4x16xf32>
    !mo.spec<?x?xnone>
    ```
  }];

  let parameters = (ins DimensionListParameter:$shape, "Type":$elementType);

  let genVerifyDecl = 1;
  let assemblyFormat = "`` $shape `` $elementType";

  let builders = [
    TypeBuilderWithInferredContext<(ins "ArrayRef<int64_t>":$shape,
                                        "Type":$elementType)>,
  ];
  let extraClassDeclaration = [{
    size_t getRank() const;
  }];
}

//===----------------------------------------------------------------------===//
// MO_Tensor
//===----------------------------------------------------------------------===//

def MO_Tensor : MO_Type<"Tensor", "tensor",
                         [DeclareTypeInterfaceMethods<ShapedTypeInterface>]> {
  let summary = "Tensors who's type is defined according to tensor spec.";
  let description = [{
    Unlike the MLIR builtin tensor type this type:
     - Only supports shapes of known rank.
     - Does not support affine maps for describing layout.
     - Allows the element type to be unknown at compile time.

    For convenience we 'inline' the tensor shape into the tensor type syntax:
    ```mlir
    !mo.tensor<4x16xf32>
    !mo.tensor<?x?xui8>
    ```
  }];

  let parameters = (ins MO_TensorSpec:$tensorSpec);
  let assemblyFormat = "`<` $tensorSpec `>`";
}

#endif // GENERICML_MODIALECT_MOTYPES_TD
