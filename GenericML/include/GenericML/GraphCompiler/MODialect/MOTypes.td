//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This defines types in the MO dialect.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MODIALECT_MOTYPES_TD
#define GENERICML_GRAPHCOMPILER_MODIALECT_MOTYPES_TD

include "GenericML/GraphCompiler/MODialect/MODialect.td"
include "Support/Compiler/TensorUtils.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/EnumAttr.td"

// Base class for types in this dialect.
class MO_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<MODialect, name, traits> {
  let mnemonic = typeMnemonic;
}

//===----------------------------------------------------------------------===//
// MO_Tensor
//===----------------------------------------------------------------------===//

def MO_Tensor : MO_Type<"Tensor", "tensor"> {
  let summary = "Type of tensor specs of statically known rank.";
  let description = [{
    This type represents the shape and element type of a tensor, plus an
    optional dictionary of metadata (e.g., layout, device, etc.).

    The `shapeAttr` is one of:
    1. `KGEN::UnknownAttr` for unparameterized shape of unknown rank, e.g., `?`.
    2. `KGEN::ParamDeclRefAttr` for a a shape parameter, e.g., `Sh0`.
    3. `MOSH::ShapeAttr` for a shape of known rank, e.g., `[D0, 42, ?]`.

    The element type is an M::DType, with `invalid` denoting an unknown type.
    The type implements a subset of the methods in ShapedTypeInterface.

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
    "DictionaryAttr" : $metadata);

  let genVerifyDecl = 1;
  let hasCustomAssemblyFormat = 1;

  let builders = [
    TypeBuilderWithInferredContext<(ins
      "TypedAttr" : $shapeAttr,
      "DType" : $dtype,
      CArg<"DictionaryAttr", "{}"> : $metadata)>,
    TypeBuilderWithInferredContext<(ins
      "TypedAttr" : $shapeAttr,
      "Type" : $elementType,
      CArg<"DictionaryAttr", "{}"> : $metadata)>,
    TypeBuilderWithInferredContext<(ins
      "ArrayRef<int64_t>" : $shape,
      "Type" : $elementType,
      CArg<"DictionaryAttr", "{}"> : $metadata)>,
    TypeBuilderWithInferredContext<(ins
      "ArrayRef<TypedAttr>" : $shape,
      "Type" : $elementType,
      CArg<"DictionaryAttr", "{}"> : $metadata)>,
    TypeBuilder<(ins
      "TypedAttr" : $shapeAttr,
      "DType" : $dtype,
      CArg<"DictionaryAttr", "{}"> : $metadata)>,
    TypeBuilder<(ins
      "ArrayRef<int64_t>" : $shape,
      "DType" : $dtype,
      CArg<"DictionaryAttr", "{}"> : $metadata)>,
    TypeBuilder<(ins
      "ArrayRef<TypedAttr>" : $shape,
      "DType" : $dtype,
      CArg<"DictionaryAttr", "{}"> : $metadata)>];

  let extraClassDeclaration = [{
    /// Returns a clone with a new integer shape and element dtype.
    TensorType cloneWith(
      std::optional<ArrayRef<int64_t>> shape, Type dtype,
      DictionaryAttr metadata = {}) const;

    /// Returns a clone with a new dimensions for shape.
    TensorType cloneWith(
      ArrayRef<TypedAttr> shapeDims) const;

    /// Returns clone with a new shape attribute.
    TensorType cloneWith(TypedAttr shapeAttr) const;

    /// Returns clone with a new dtype attribute.
    TensorType cloneWith(DType dtype) const;

    /// Returns a clone with a new layout attribute.
    TensorType cloneWith(LayoutAttr layoutAttr) const;

    /// Returns clone with a new metadata attribute.
    TensorType cloneWith(DictionaryAttr metadataAttr) const;

    /// Returns clone with a shape of unknown rank.
    TensorType cloneWithUnknownRank() const;

    /// Returns clone with a shape of known rank, but unknown dims
    TensorType cloneWithUnknownDims(int64_t rank) const;

    /// Returns a clone of type with given #M.device_ref annotation. The
    /// type must not already have such an annotation.
    TensorType cloneWith(DeviceRefAttr device) const;

    /// Checks if the tensor is statically ranked.
    bool isDynamicallyRanked() const;
    bool hasRank() const;

    /// Get the rank of the tensor. Fails if not statically ranked.
    size_t getRank() const;

    /// Checks if this type has a static shape, i.e. if the type is ranked and
    /// all dimensions have known size.
    bool hasStaticShape() const;

    /// Get the mlir element type of the tensor.
    Type getElementType() const;

    /// Helper to get the shape attribute with MOSH::ShapeAttr type. Fails if
    /// the shape is unknown.
    MOSH::ShapeAttr getTypedShapeAttr() const;

    /// Returns the referenced parameter names (shape or dimension).
    SmallPtrSet<StringAttr, 4> getUsedParamNames() const;

    /// Return the symbolic dimensions. Fails if the shape is unknown.
    ArrayRef<TypedAttr> getDims() const;

    /// Return the nth symbolic dimensions. Fails if the shape is unknown.
    TypedAttr getDim(size_t n) const;

    /// Return whether the nth dimension is `?`. Parameters are known.
    /// Fails if dynamically ranked.
    bool isUnknownDim(size_t n) const;

    /// Return whether the nth dimension is a parameter. Does not include `?`.
    /// Fails if dynamically ranked. Accepts expressions with parameters e.g.
    /// (A + 2).
    bool isParameterizedDim(size_t n) const;

    /// Return whether the nth dimension is a known integer.
    /// Fails if dynamically ranked.
    bool isIntegerDim(size_t n) const;

    /// Returns what type of value is represented by the given dimension attr.
    static bool isUnknownDimParam(TypedAttr attr);
    static bool isParameterizedDimParam(TypedAttr attr);
    static bool isIntegerDimParam(TypedAttr attr);

    /// Return the dimensions as integers, using mlir::ShapedType::kDynamic for
    /// unknown dimensions. Fails if the shape is unknown.
    SmallVector<int64_t> getIntDims() const;

    /// Return the nth dimension as an integer, using mlir::ShapedType::kDynamic
    /// for unknown dimensions. Fails if the shape is unknown.
    int64_t getIntDim(size_t n) const;

    /// Checks if the shape or any dimensions are unknown. Parameters are
    /// considered known.
    bool hasUnknownShape() const;

    /// Returns the number of elements as a symbolic (i.e. parametric)
    /// expression. Fails if not statically ranked.
    TypedAttr getSymbolicNumElements() const;

    // Returns the number of elements in the tensor if defined. This is when
    // the rank is known and all dimensions are statically known integers >= 0.
    // Otherwise returns no value.
    std::optional<int64_t> getNumElements() const;

    /// Computes the common broadcast shape of the shapes of given tensors.
    /// Supports parametric and unknown shapes, but not dynamic rank.
    static ErrorOr<TensorType> broadcastShapes(TensorType x, TensorType y);

    /// Short-hand to create an SI32 scalar tensor type.
    static TensorType getScalarSI32Type(MLIRContext *ctx,
      DictionaryAttr metadata = {});

    /// Short-hand to create an SI64 scalar tensor type.
    static TensorType getScalarSI64Type(MLIRContext *ctx,
      DictionaryAttr metadata = {});

    /// Short-hand to create an F32 scalar tensor type.
    static TensorType getScalarF32Type(MLIRContext *ctx,
      DictionaryAttr metadata = {});

    /// Short-hand to create an F64 scalar tensor type.
    static TensorType getScalarF64Type(MLIRContext *ctx,
      DictionaryAttr metadata = {});

    /// Short-hand to create a SI64 vector tensor type.
    static TensorType getVectorSI64Type(MLIRContext *ctx, int64_t size,
      DictionaryAttr metadata = {});

    /// Return the layout attribute carried by this tensor type. If no layout
    /// attribute is specified, return a null attribute `LayoutAttr{}`.
    MO::LayoutAttr getLayoutAttr();

    /// Returns the #M.device_ref annotation for this type, or null if no
    /// such annotation.
    DeviceRefAttr getDeviceRef() const;

    /// Duplicates a tensor type but with a dimension set to a new value.
    TensorType cloneWithNewDim(uint64_t dim, uint64_t size);
  }];
}

//===----------------------------------------------------------------------===//
// MO_List
//===----------------------------------------------------------------------===//

def MO_List : MO_Type<"List", "list"> {
  let summary = "Type of immutable list container.";
  let description = [{
    This type represents an immutable list of elements (currently restricted to
    `!mo.tensor`).
  }];

  let parameters = (ins "Type":$elementType);
  let assemblyFormat = "`<` $elementType `>`";
  let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// MO_Scalar
//===----------------------------------------------------------------------===//

def MO_Scalar : MO_Type<"Scalar", "scalar"> {
  let summary = "Type of scalars.";
  let description = [{
    This type represents scalars.

  }];

  let parameters = (ins
    "DType" : $dtype);

  let assemblyFormat = "`<` $dtype `>`";

  let extraClassDeclaration = [{
    /// Get the mlir element type of the tensor.
    Type getDType() const;
  }];
}

//===----------------------------------------------------------------------===//
// MO_Chain
//===----------------------------------------------------------------------===//

def MO_Chain : MO_Type<"Chain", "chain"> { 
  let summary = "Controls sequencing of side-effecting operations.";
  let description = [{
    This type is used to sequence side-efffecting operations. Any operation in
    the MO dialect that has side-effects should both consume and produce a
    chain type.
  }];
}

//===----------------------------------------------------------------------===//
// ODS constraints to inject verifier logic via type annotation.
//===----------------------------------------------------------------------===//

/// Checks if a MO_TensorType has static rank.
def MO_HasRankPred : CPred<"cast<MO::TensorType>($_self).hasRank()">;

// Checks if a MO_TensorType has one of the specified ranks.
class MO_HasAnyRankOfPred<list<int> ranks> : And<[
  MO_HasRankPred,
  Or<!foreach(rank, ranks,
    CPred<[{cast<MO::TensorType>($_self).getRank() == }] # rank>)>]>;

/// Checks if a type has rank N if statically ranked.
class HasRankIfRankedPred<int N> :
  Or<[Neg<MO_HasRankPred>, MO_HasAnyRankOfPred<[N]>]>;

/// Checks if a MO_TensorType has float elements.
def FloatElementTypePred : CPred<
  "llvm::isa<FloatType>(cast<::M::MO::TensorType>($_self).getElementType())">;

/// Checks if a MO_TensorType has int or index elements.
def IntOrIndexElementTypePred : CPred<
  "cast<::M::MO::TensorType>($_self).getElementType().isIntOrIndex()">;

// Checks if a MO_TensorType has signed elements.
def SignedElementTypePred : CPred<
  "!cast<::M::MO::TensorType>($_self).getElementType().isUnsignedInteger() && "
  "!cast<::M::MO::TensorType>($_self).getDtype().isBool()">;

/// Checks if a MO_TensorType has SI64 elements.
def HasSI64ElementTypePred : SubstLeaves<
  "$_self", "cast<M::MO::TensorType>($_self).getElementType()", SI64.predicate>;

/// Checks if a MO_TensorType has SI32 elements.
def HasSI32ElementTypePred : SubstLeaves<
  "$_self", "cast<M::MO::TensorType>($_self).getElementType()", SI32.predicate>;

/// Checks if a MO_TensorType has I1 elements.
def HasBoolElementTypePred : SubstLeaves<
  "$_self", "cast<M::MO::TensorType>($_self).getElementType()", I1.predicate>;

/// Base class for defining constraints on MO_TensorType.
class MO_ConfinedTensor<list<Pred> predicates, string summary = ""> :
  ConfinedType<MO_Tensor, predicates, summary, "::M::MO::TensorType">;

class MO_ConfinedScalar<list<Pred> predicates, string summary = ""> :
  ConfinedType<MO_Scalar, predicates, summary, "::M::MO::ScalarType">;

def BoolScalarPred : CPred<
  "cast<::M::MO::ScalarType>($_self).getDtype().isBool()">;

def MO_BoolScalar : MO_ConfinedScalar<[BoolScalarPred], "boolean scalar">;

/// Confines the MO_TensorType so that it's ranked and a scalar.
def MO_RankedScalarTensor : MO_ConfinedTensor<
  [MO_HasAnyRankOfPred<[0]>], "ranked scalar tensor">;

/// Confines the MO_TensorType so that it's a scalar or unranked.
def MO_ScalarTensorIfRanked : MO_ConfinedTensor<
  [HasRankIfRankedPred<0>], "scalar or unranked tensor">;

/// Confines the MO_TensorType so that it's ranked and a scalar int or index.
def MO_RankedScalarIndexTensor : MO_ConfinedTensor<
  [MO_HasAnyRankOfPred<[0]>, IntOrIndexElementTypePred],
  "ranked scalar index tensor">;

/// Confines the MO_TensorType so that it's a scalar float.
def MO_RankedScalarFloatTensor : MO_ConfinedTensor<
  [MO_HasAnyRankOfPred<[0]>, FloatElementTypePred],
  "ranked scalar floating point tensor">;

/// Confines the MO_TensorType so that it has float elements, and rank 0 if
/// rank is static.
def MO_FloatScalarIfRanked : MO_ConfinedTensor<
  [FloatElementTypePred, HasRankIfRankedPred<0>],
  "rank 0 or unranked with float elements">;

/// Confines the MO_TensorType so that it has floats.
def MO_FloatTensor : MO_ConfinedTensor<[FloatElementTypePred], "floating point tensor">;

/// Confines the MO_TensorType so that it's either unranked or a non-scalar.
def MO_NonScalarIfRanked: MO_ConfinedTensor<
  [Neg<MO_HasAnyRankOfPred<[0]>>], "unranked or non-scalar tensor">;

/// Confines the MO_TensorType so it is a rank 1 floating point tensor.
def MO_Rank1FloatTensor: MO_ConfinedTensor<
  [MO_HasAnyRankOfPred<[1]>, FloatElementTypePred], "rank-1 tensor with floats">;

/// Confines the MO_TensorType so it is a rank 1 int or index tensor.
def MO_Rank1IndexTensor: MO_ConfinedTensor<
  [MO_HasAnyRankOfPred<[1]>, IntOrIndexElementTypePred], "rank-1 tensor with indices">;

/// Confines the MO_TensorType so it is rank 2 and has index dtype.
def MO_Rank2IndexTensor: MO_ConfinedTensor<
  [MO_HasAnyRankOfPred<[2]>, IntOrIndexElementTypePred], "rank-2 index tensor">;

/// Confines the MO_TensorType so it is rank 2.
def MO_Rank2Tensor: MO_ConfinedTensor<
  [MO_HasAnyRankOfPred<[2]>], "rank-2 tensor">;

/// Confines the MO_TensorType so it is rank 3.
def MO_Rank3Tensor: MO_ConfinedTensor<
  [MO_HasAnyRankOfPred<[3]>], "rank-3 tensor">;

/// Confines the MO_TensorType so it is rank 4.
def MO_Rank4Tensor: MO_ConfinedTensor<
  [MO_HasAnyRankOfPred<[4]>], "rank-4 tensor">;

/// Confines the MO_TensorType so that it has int or index elements.
def MO_IndicesTensor : MO_ConfinedTensor<
  [IntOrIndexElementTypePred], "tensor with integer or index elements">;

/// Confines the MO_TensorType so that it has signed elements.
def MO_SignedTensor : MO_ConfinedTensor<
  [SignedElementTypePred], "tensor with signed elements">;

/// Confines the MO_TensorType so that it has SI32 or SI64 elements and rank 1.
def MO_ShapeLike : MO_ConfinedTensor<
  [MO_HasAnyRankOfPred<[1]>, Or<[HasSI64ElementTypePred,
                                 HasSI32ElementTypePred]>],
  "rank 1 and signed 64 or signed 32 integer elements">;

/// Confines the MO_TensorType so that it has SI32 or SI64 elements and shape
/// [N].
def MO_PaddingLike : MO_ConfinedTensor<
  [MO_HasAnyRankOfPred<[1]>, Or<[HasSI64ElementTypePred, HasSI32ElementTypePred]>],
  "shape [N] and signed 64 or signed 32 integer elements">;

/// Confines the MO_TensorType so that it has SI32 or SI64 elements and shape
/// [N] or unranked.
def MO_PaddingLikeIfRanked : MO_ConfinedTensor<
  [HasRankIfRankedPred<1>, Or<[HasSI64ElementTypePred, HasSI32ElementTypePred]>],
  "unranked or shape [N] and signed 64 or signed 32 integer elements">;

/// Confines the MO_TensorType so that it has SI64 elements, and rank 1 if the
/// rank is static.
def MO_ShapeLikeIfRanked : MO_ConfinedTensor<
  [HasRankIfRankedPred<1>, Or<[HasSI64ElementTypePred,
                               HasSI32ElementTypePred]>],
  "rank 1 or unranked with signed 64 or signed 32 integer elements">;

/// Confines the MO_TensorType so that it has rank 1 if rank is static.
def MO_VectorIfRanked : MO_ConfinedTensor<
  [HasRankIfRankedPred<1>], "rank 1 or unranked">;

/// Confines the MO_TensorType so that it has int/index elements, and rank 1 if
/// rank is static.
def MO_IndicesVectorIfRanked : MO_ConfinedTensor<
  [IntOrIndexElementTypePred, HasRankIfRankedPred<1>],
  "rank 1 or unranked with integer or index elements">;

/// Confines the MO_TensorType so that it has int/index elements, and rank 0 if
/// rank is static.
def MO_IndexScalarIfRanked : MO_ConfinedTensor<
  [IntOrIndexElementTypePred, HasRankIfRankedPred<0>],
  "rank 0 or unranked with integer or index elements">;

/// Confines the MO_TensorType so that it has rank 2 if rank is static.
def MO_MatrixIfRanked : MO_ConfinedTensor<
  [HasRankIfRankedPred<2>], "rank 2 or unranked">;

/// Confines the MO_TensorType so that it cannot have unknown shape or dimension
/// parameters.
def MO_KnownTensor : MO_ConfinedTensor<
  [CPred<"!cast<MO::TensorType>($_self).hasUnknownShape()">],
  "parametrically known shape">;

/// Confines the MO_TensorType so that it has Boolean elements
def MO_BooleanMask : MO_ConfinedTensor<
    [HasBoolElementTypePred], "boolean element type">;

/// Confines the MO_TensorType so that it has Boolean elements and rank 0
def MO_CondLike : ConfinedType<MO_Tensor,
    [MO_HasAnyRankOfPred<[0]>, HasBoolElementTypePred],
    "rank 0 and boolean element type"> {
  let cppClassName = "::M::MO::TensorType";
}

/// Confines the MO_TensorType so that it has Boolean elements and rank 0 if
/// rank is static.
def MO_CondLikeIfRanked : ConfinedType<MO_Tensor,
    [HasRankIfRankedPred<0>, HasBoolElementTypePred],
    "rank 0 or unranked with boolean element type"> {
  let cppClassName = "::M::MO::TensorType";
}

/// Confines the MO_TensorType so that it has signed 64-bit integer elements
def MO_I64Tensor : ConfinedType<MO_Tensor,
    [HasSI64ElementTypePred],
    "signed 64 integer element type"> {
  let cppClassName = "::M::MO::TensorType";
}

class DTypeStrFunc<string name> : StrFunc<"::M::MO::getDTypeOrInvalid($" # name # ".getType())">;

/// Check all operands `names...` have the same dtype.
/// eg: `AllDTypesMatch<["input", "result"]>`.
/// There is an existing `AllElementTypesMatch` def,
/// but it doesn't work with MO::TensorType.
class AllDTypesMatch<list<string> names> :
    AllMatchSameOperatorTrait<names, DTypeStrFunc<"_self">.result,
                              "dtype">;

#endif // GENERICML_GRAPHCOMPILER_MODIALECT_MOTYPES_TD
