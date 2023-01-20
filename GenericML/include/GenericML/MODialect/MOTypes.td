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

/// The type of parametric shape attributes.
def MO_ShapeType : MO_Type<"Shape", "shape">;

def MO_Tensor : MO_Type<"Tensor", "tensor", [ShapedTypeInterface]> {
  // TODO: Relax the constraint on dynamic rank once we reach
  // runtime agreement with MGP team.
  let summary = "Type of tensor specs of statically known rank.";
  let description = [{
    This type represents the shape and element type of a tensor. Each
    dimension must be a strictly positive integer value, the distinguished
    kDynamic dimension denoting unknown, or the distinguished kDynamicRank
    denoting an unkown number of dynamic axes. The element type is an M::DType,
    with `invalid` denoting an unknown type.

    Examples:
    ```mlir
    !mo.tensor<[4, 16], f32>
    !mo.tensor<[?, ?], i32>
    !mo.tensor<?, invalid>
    ```
  }];

  let parameters = (ins
    ShapeParameter : $shape,
    "DType" : $dtype,
    DefaultValuedParameter<"std::optional<Attribute>", "std::nullopt"> : $metadata);

  let genVerifyDecl = 1;
  let assemblyFormat = "`<` $shape `,` $dtype ( `,` $metadata^)? `>`";

  let builders = [TypeBuilderWithInferredContext<(ins
      "ArrayRef<int64_t>" : $shape,
      "Type" : $elementType,
      CArg<"std::optional<Attribute>", "std::nullopt"> : $metadata),
    [{
      ErrorOr<DType> errOrDType = fromMLIRType(elementType);
      assert(!errOrDType.isError() &&
              "element type is not representable as a DType");
      return $_get(elementType.getContext(), shape, *errOrDType, metadata);
    }]>];

  let extraClassDeclaration = [{
    size_t getRank() const;
    bool hasRank() const;
    Type getElementType() const;
    ShapedType cloneWith(std::optional<ArrayRef<int64_t>> shape, Type dtype,
                         std::optional<Attribute> metadata = std::nullopt) const;
  }];
}

//===----------------------------------------------------------------------===//
// ODS constraints to inject verifier logic via type annotation.
//===----------------------------------------------------------------------===//

/// Checks if a type has rank N if statically ranked.
class HasRankIfRankedPred<int N> : Or<[Neg<HasRankPred>, HasAnyRankOfPred<[N]>]>;

/// Checks if a MO_TensorType has int or index elements
def IntOrIndexElementTypePred : CPred<"$_self.cast<::M::MO::TensorType>().getElementType().isIntOrIndex()">;

/// Checks if a MO_TensorType has SI64 elements.
def HasSI64ElementTypePred : SubstLeaves<
  "$_self", "cast<M::MO::TensorType>($_self).getElementType()", SI64.predicate>;

/// Confines the MO_TensorType so that it's ranked and a scalar
def MO_RankedScalarTensor : ConfinedType<MO_Tensor,
    [HasAnyRankOfPred<[0]>],
    "ranked scalar tensor"> {
  let cppClassName = "::M::MO::TensorType";
}

/// Confines the MO_TensorType so that it's either unranked or a non-scalar
def MO_NonScalarIfRanked: ConfinedType<MO_Tensor,
    [Neg<HasAnyRankOfPred<[0]>>],
    "unranked or non-scalar tensor"> {
  let cppClassName = "::M::MO::TensorType";
}

/// Confines the MO_TensorType so that it has int or index elements
def MO_IndicesTensor : ConfinedType<MO_Tensor,
    [IntOrIndexElementTypePred],
    "tensor with integer or index elements"> {
  let cppClassName = "::M::MO::TensorType";
}

/// Confines the MO_TensorType so that it has SI64 elements and rank 1
def MO_ShapeLike : ConfinedType<MO_Tensor,
    [HasAnyRankOfPred<[1]>, HasSI64ElementTypePred],
    "rank 1 and signed 64 integer elements"> {
  let cppClassName = "::M::MO::TensorType";
}

/// Confines the MO_TensorType so that it has SI64 elements, and rank 1 if the
/// rank is static.
def MO_ShapeLikeIfRanked : ConfinedType<MO_Tensor,
    [HasRankIfRankedPred<1>, HasSI64ElementTypePred],
    "rank 1 or unranked with signed 64 integer elements"> {
  let cppClassName = "::M::MO::TensorType";
}

/// Confines the MO_TensorType so that it has rank 2 if rank is staticl
def MO_MatrixIfRanked : ConfinedType<MO_Tensor,
    [HasRankIfRankedPred<2>],
    "rank 2 or unranked"> {
  let cppClassName = "::M::MO::TensorType";
}

#endif // GENERICML_MODIALECT_MOTYPES_TD
