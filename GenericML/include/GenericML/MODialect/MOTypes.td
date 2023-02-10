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

include "GenericML/MODialect/MOAttrs.td"
include "KGEN/KGENDialect/KGENTypeInterfaces.td"
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

/// The type of parametric shape attributes.
def MO_ShapeType : MO_Type<"Shape", "shape", [
  DeclareTypeInterfaceMethods<ParameterTypeInterface>]>;

def MO_Tensor : MO_Type<"Tensor", "tensor", [ShapedTypeInterface]> {
  let summary = "Type of tensor specs of statically known rank.";
  let description = [{
    This type represents the shape and element type of a tensor. The shape is
    stored as a MO::ShapeAttr, while the element type is an M::DType, with
    `invalid` denoting an unknown type.

    Examples:
    ```mlir
    !mo.tensor<[4, 16], f32>    // static shape
    !mo.tensor<[N, N, 6], i32>  // parameterized shape
    !mo.tensor<[?, ?], i32>     // unknown shape of known rank
    !mo.tensor<[1, ?, N], i32>  // partially known and parameterized shape
    !mo.tensor<?, invalid>      // unknown shape of unknown rank
    !mo.tensor<Sh, invalid>     // shape parameter reference
    ```
  }];

  let parameters = (ins
    "TypedAttr" : $shapeAttr,
    "DType" : $dtype,
    ArrayRefParameter<"int64_t">:$shape,
    DefaultValuedParameter<"std::optional<Attribute>", "std::nullopt"> : $metadata);

  let genVerifyDecl = 1;
  let assemblyFormat = [{
    `<` custom<ShapeParam>($shapeAttr, $shape) `,` $dtype ( `,` $metadata^)? `>`
  }];

  let builders = [
    TypeBuilderWithInferredContext<(ins
      "TypedAttr" : $shapeAttr,
      "DType" : $dtype,
      CArg<"std::optional<Attribute>", "std::nullopt"> : $metadata)>,
    TypeBuilderWithInferredContext<(ins
      "TypedAttr" : $shapeAttr,
      "Type" : $elementType,
      CArg<"std::optional<Attribute>", "std::nullopt"> : $metadata)>,
    TypeBuilderWithInferredContext<(ins
      "ArrayRef<int64_t>" : $shape,
      "Type" : $elementType,
      CArg<"std::optional<Attribute>", "std::nullopt"> : $metadata)>];

  let extraClassDeclaration = [{
    size_t getRank() const;
    bool hasRank() const;
    Type getElementType() const;
    ShapedType cloneWith(std::optional<ArrayRef<int64_t>> shape, Type dtype,
                         std::optional<Attribute> metadata = std::nullopt) const;
    TensorType cloneWith(TypedAttr shapeAttr) const;
    bool hasStaticShape() const;

    /// Helper to get the shape attribute with MO::ShapeAttr type, fails if the
    /// shape is unknown.
    MO::ShapeAttr getTypedShapeAttr() const;

    /// Returns the referenced parameter names.
    SmallPtrSet<StringAttr, 4> getUsedParamNames() const;
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

/// Checks if a MO_TensorType has SI32 elements.
def HasSI32ElementTypePred : SubstLeaves<
  "$_self", "cast<M::MO::TensorType>($_self).getElementType()", SI32.predicate>;

/// Checks if a MO_TensorType has shape [N, 2]
def HasPaddingShape : And<[
  HasAnyRankOfPred<[2]>,
  CPred<"$_self.cast<::M::MO::TensorType>().getDimSize(1) == 2">]>;

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

/// Confines the MO_TensorType so that it has SI32 or SI64 elements and rank 1
def MO_ShapeLike : ConfinedType<MO_Tensor,
    [HasAnyRankOfPred<[1]>, Or<[HasSI64ElementTypePred,
                                HasSI32ElementTypePred]>],
    "rank 1 and signed 64 or signed 32 integer elements"> {
  let cppClassName = "::M::MO::TensorType";
}

/// Confines the MO_TensorType so that it has SI32 or SI64 elements and shape
/// [N, 2]
def MO_PaddingLike : ConfinedType<MO_Tensor,
    [HasPaddingShape, Or<[HasSI64ElementTypePred,
                          HasSI32ElementTypePred]>],
    "shape [N, 2] and signed 64 or signed 32 integer elements"> {
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
