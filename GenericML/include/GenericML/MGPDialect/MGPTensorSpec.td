//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This defines interfaces on tensors in the MGP dialect.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MGPDIALECT_MGPTENSORSPEC_TD
#define GENERICML_GRAPHCOMPILER_MGPDIALECT_MGPTENSORSPEC_TD

include "GenericML/MGPDialect/MGPDialect.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/OpBase.td"

// Base class for types in this dialect. Same thing as MGP_Type but we need a
// separate definition for this file.
class MGP_BaseType<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<MGPDialect, name, traits> {
  let mnemonic = typeMnemonic;
}

//===----------------------------------------------------------------------===//
// TensorSpecType
//===----------------------------------------------------------------------===//

def MGP_TensorSpecType : MGP_BaseType<"TensorSpec", "tensor_spec", [
    DeclareTypeInterfaceMethods<ShapedTypeInterface>,
]> {
  let summary = "Type of tensor specs.";
  let description = [{
    `!mgp.tensor_spec` represents the shape and dtype of a tensor.

    Both static and dynamic (aka unknown) aspects are supported:
     - The rank may be static (implied by the number of dimensions, ie the
       number of `x` delimiters) or dynamic (denoted by `*`). Only ranks up to
       kMaxRank are supported (currently 8). A rank of zero denotes 'scalars',
       though in practice this has the same runtime representation as rank 1,
       dimension 1.
     - For static ranks, each dimension may be static (an unsigned integer
       literal) or dynamic (denoted by `?`). Symbolic dimensions are not
       expressible in MGP.
     - The dtype may be static (using the DType::getFromString syntax) or
       dynamic (denoted by`unk`).

    DTypes can represent most variations of MLIR IntegerType and FloatTypes.
    However:
     - Signless and non power-of-two MLIR IntegerTypes have no DType
       representation.
     - The complex and a few float DTypes have no MLIR Type representation.
     - The 'invalid' (zero) DType is never permitted.
    If a tensor spec is given a DType we also require it to be representable as
    an MLIR type.

    This type implements ShapedTypeInterface. We'll use 'dims'/'dtype' when
    referring to our internal representation, and 'shape'/'element type' when
    referring to the ShapedTypeInterface representation.

    Here are all the different representations in flight:
     - For dimensions:
        - 'optional' form: std::optional<uint64_t>, where nullopt denotes
          a dynamic dimension. God's own representation.
        - 'signed' form: int64_t, where ShapedType::kDynamicSize denotes
          a dynamic dimension. As used by ShapedTypeInterface.
        - 'raw' form: uint64_t, where ~0 denotes a dynamic dimension. Used
          when dimensions need to be encoded into MEF ops via attributes.
     - For shapes:
        - 'optional' form: std::optional<ArrayRef<std::optional<uint64_t>>>,
          where nullopt denotes a dynamic rank.
        - 'shaped type' form: ArrayRef<int64_t>, as used by ShapedTypeInterface.
        - 'raw' form: IntArrayElementsAttr of UI64Ints.
     - For dtypes:
        - 'optional' form: std::optional<DType>, where nullopt denotes a
          dynamic dtype.
        - 'shaped type' form: An MLIR type, where NoneType denotes a dynamic
          dtype.
        - 'raw' form: uint8_t, where 0 (ie invalid) denotes a dynamic dtype.

    `!mgp.tensor_specs` are represented by `AsyncRT::AsyncValueRef<M::TensorSpec>`
    in the host C++ runtime. All dimensions in the runtime value should 'refine'
    the corresponding compile time spec, ie runtime dims must equal statically
    known compile-time dims.

    Examples:
    ```mlir
    // Fully statically known spec.
    !mgp.tensor_spec<4x16xf32>
    // Spec with one dynamic dimension and dynamic dtype.
    !mgp.tensor_spec<?x3xunk>
    // Dynamically ranked spec.
    !mgp.tensor_spec<*xi32>
    ```
  }];

  // NOTE: To support the ShapedTypeInterface::getShape() method we maintain
  // 'optional' (dimsOrEmpty) and 'shaped type' (shapeOrEmpty) forms.
  // See the builders for the smart constructors.
  let parameters = (ins "bool":$hasRank,
                        ArrayRefParameter<"std::optional<uint64_t>">:$dimsOrEmpty,
                        "std::optional<DType>":$optDType,
                        ArrayRefParameter<"int64_t">:$shapeOrEmpty);
  let assemblyFormat = "`<` custom<DimsAndDType>($hasRank, $dimsOrEmpty, $optDType, $shapeOrEmpty) `>`";
  let genVerifyDecl = 1;

  let extraClassDeclaration = [{
    /// For the ShapedTypeInterface we implement:
    ///   bool hasRank();
    ///   ShapedType cloneWith(std::optional<ArrayRef<int64_t>> shape,
    ///                        Type elementType);
    ///   Type getElementType();
    ///   ArrayRef<int64_t> getShape(); // requires hasRank().
    /// CAUTION: Most of the following methods require hasRank() and will
    ///          assert fail otherwise. Take care.

    /// Returns the dimensions of this tensor spec, or nullopt if dynamic
    /// ranked.
    /// Dynamic ranked safe.
    std::optional<llvm::ArrayRef<std::optional<uint64_t>>> getOptDims() const;

    /// Returns the dimensions of this tensor spec.
    /// Requires hasRank().
    llvm::ArrayRef<std::optional<uint64_t>> getDims() const;

    /// Returns the rank of this tensor spec.
    /// Requires hasRank().
    size_t getRank() const;

    /// Returns the number of unknown, aka 'dynamic' dimensions in
    /// this tensor spec.
    /// Requires hasRank().
    size_t getNumDynamicDims() const;

    /// Return the dimension at index i in this spec, or nullopt if the
    /// dimension is dynamic.
    /// Requires hasRank().
    std::optional<uint64_t> operator[](size_t i) const;

    /// Returns the number of elements in a tensor of this spec, or
    /// nullopt if the tensor shape is not fully static (ie dynamic rank,
    /// or at least one dynamic dimension).
    /// Dynamic ranked safe.
    std::optional<uint64_t> getNumElements() const;

    /// Returns the number of elements for the known dimensions of this spec.
    /// The actual number of elements will be this value multiplied by the
    /// runtime dimensions of all dynamic dimensions.
    /// Requires hasRank().
    uint64_t getPreScaledNumElements() const;

    /// Returns the size of the buffer needed to represent a tensor
    /// of this spec, or nullopt if the tensor spec is not fully static
    /// (ie dynamic rank, at least one dynamic dimension, or dynamic dtype).
    /// Dynamic ranked safe.
    std::optional<uint64_t> getSizeInBytes() const;

    /// Returns the byte size needed to represent getNumPreScaledElements(),
    /// or nullopt if the tensor spec does not have a statically known dtype.
    /// Requires hasRank().
    std::optional<uint64_t> getPreScaledSizeInBytes() const;

    /// Returns the dimensions of this tensor spec in 'raw form'.
    /// Each element of the result will be an UI64Int, with ~0 denoting
    /// an unknown dimension.
    /// Requires hasRank().
    IntArrayElementsAttr getRawDims() const;

    /// Returns the dtype of this tensor spec in 'raw form'.
    /// The result will be a UI8Int using the DType::value encoding,
    /// with 0 denoting the unknown dtype.
    mlir::IntegerAttr getRawDType() const;
  }];

  let builders = [
    /// Constructs a tensor spec from its components in 'optional' form.
    TypeBuilder<(ins "std::optional<llvm::ArrayRef<std::optional<uint64_t>>>":$optDims,
                     "std::optional<DType>":$optDType)>,

    /// Constructs a tensor spec from its components in 'shaped type' form.
    /// Only supports statically known ranks.
    TypeBuilderWithInferredContext<(ins "llvm::ArrayRef<int64_t>":$shape,
                                        "mlir::Type":$elementType)>,

    /// Constructs a dynamically ranked tensor spec with the given element type.
    TypeBuilderWithInferredContext<(ins "mlir::Type":$elementType)>,

    /// Construct a tensor spec from its components in 'raw form',
    /// using underlying integer values. Only supports statically known ranks.
    TypeBuilder<(ins "llvm::ArrayRef<uint64_t>":$rawDims, "uint8_t":$rawDType)>,

    /// Construct a tensor spec from its components in 'raw form', using
    /// actual attributes. As a convenience, the null dtype attribute is taken
    /// to denote the unknown element type. Only supports statically known
    /// ranks.
    TypeBuilderWithInferredContext<(ins "IntArrayElementsAttr":$rawDims,
                                        "mlir::IntegerAttr":$rawDTypeOrNull)>,
  ];
}

#endif
