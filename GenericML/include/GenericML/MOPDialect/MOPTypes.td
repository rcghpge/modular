//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This defines types in the MOP dialect.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_MOPDIALECT_MOPTYPES_TD
#define GENERICML_MOPDIALECT_MOPTYPES_TD

include "GenericML/MOPDialect/MOPDialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/SubElementInterfaces.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/AttrTypeBase.td"

// Base class for types in this dialect.
class MOPType<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<MOPDialect, name, traits> {
  let mnemonic = typeMnemonic;
}

//===----------------------------------------------------------------------===//
// BufferType
//===----------------------------------------------------------------------===//

def BufferType : MOPType<"Buffer", "buffer",
                         [DeclareTypeInterfaceMethods<ShapedTypeInterface>]> {
  let summary = "Type of uninterpreted byte arrays, optionally with statically "
                "known size and alignment";
  let description = [{
    Buffers are byte arrays which may have a statically known size and
    alignment 'constraints'. Nothing is implied about the internal structure
    of the bytes, such as rank, dimensions, underlying element type, padding,
    or element endianness. The compiler is responsible for ensuring the
    producers and consumers of buffers agree on their interpretation of the
    buffer for the intended target.

    Generally buffers are the lowest-common denominator across kernel
    implementations. Buffers may be wrapped by various tensor types, or
    used directly.

    This type implements `ShapedTypeInference` as a rank-1 tensor over
    `uint8_t` with optimally known first dimension. This in turn supports the
    `ElementsAttrInterface` on buffer attributes.

    Buffers are represented by `M::LLCL::RCBufferRefs` in the Modular runtime.

    Note that align constraints in buffer types are generally only needed
    when constructing a buffer from a dense_resource. Otherwise the align
    constraint is not propagated within types to simplify type checking.

    Examples:
    ```mlir
    !mop.buffer<size = 16>
    !mop.buffer<size = 16, align = 64>
    !mop.buffer
    ```
  }];

  let parameters = (ins OptionalParameter<"Optional<int64_t>">:$size,
                        OptionalParameter<"Optional<int64_t>">:$align);
  let assemblyFormat = "`<` struct(params) `>`";
  let genVerifyDecl = 1;

  let extraClassDeclaration = [{
     /// getShape may need to return an ArrayRef representing a
     /// dynamic rank-1 shape if the size is unconstrained.
     static const int64_t unknownSize;

     /// Returns true if rhs is a BufferType and lhs and rhs have equal size
     /// constraints.
     static bool agreeOnSize(BufferType lhs, Type rhs);

     /// Returns this buffer type without any align constraint.
     BufferType forgetAlign();
  }];
}

//===----------------------------------------------------------------------===//
// ContextType
//===----------------------------------------------------------------------===//

def ContextType : MOPType<"Context", "context"> {
  let summary = "Type of execution contexts holding a number of primitive "
                "operation contexts";
  let description = [{
    Contexts are objects created by the 'init' phase of a model, then reused
    for each call to the model's 'execute'. Contexts may contain any number
    of primitive operation contexts. A primitive operation context holds
    attributes, scratch space, pre-computed data and, if required, cached
    output buffers needed by a primitive. Each primitive may declare its
    own context structure, and each call to that primitive in the graph may
    have its own instance of that structure.

    Contexts are represented by `M::MOP::Context` in the Modular runtime.

    Example:
    ```mlir
    !mop.context
    ```
  }];
}

//===----------------------------------------------------------------------===//
// DType
//===----------------------------------------------------------------------===//

// TODO: See also kgen.dtype.
def DTypeType : MOPType<"DType", "dtype"> {
  let summary = "Type of tensor element types, aka 'dtype'";
  let description = [{
    This type encodes the element type of a tensor. It can represent
    integer and floating point types of a variety of widths and signedness.

    DTypes are represented by M::DType in the Modular runtime. However
    currently the dtypes of tensor types are always known statically and so
    there are no operators on this type. It exists only to mediate between the
    MLIR IntegerType and FloatType types and the Modular DType encoding of
    those types.

    For convenience this type can be constructed using the built-in integer and
    floating point types:
    ```mlir
    !mop.dtype<i8>
    !mop.dtype<fp32>
    ```
    However not all such types are supported.
  }];

  let parameters = (ins "Type":$mlirType);
  let assemblyFormat = "`<` $mlirType `>`";
  let genVerifyDecl = 1;

  let extraClassDeclaration = [{
    /// Returns the M::DType representing the given MLIR IntegerType or FloatType.
    /// Returns the invalid DType if type cannot be represented.
    static DType mlirTypeToRawDType(Type mlirType);

    /// Returns the MLIR IntegerType or FloatType representing the given DType.
    /// Returns the null type if DType cannot be represented.
    static Type rawDTypeToMLIRType(MLIRContext* ctxt, DType rawDType);

    /// Returns the raw DType for this type.
    DType getRawDType() const;
  }];
}

//===----------------------------------------------------------------------===//
// TensorShapeType
//===----------------------------------------------------------------------===//

def TensorShapeType : MOPType<"TensorShape", "shape"> {
  let summary = "Type of tensor shapes of statically known rank";
  let description = [{
    This type represents the shape of a tensor. It must be between one and
    five dimensions. Each dimension can be a strictly positive integer
    value or the distinguished '?' unknown dimensions.

    Dynamic ranks are not supported. Symbolic dimensions are not supported.

    TensorShapes are represented by M::TensorShape in the Modular runtime.
    The runtime shape will have the same rank and matching dimensions for
    the known dimensions in this type.

    For convenience this type can be constructed using the same body syntax
    as the built-in tensor type, though without the trailing element type:
    ```mlir
    !mop.shape<4x16>
    !mop.shape<?x?>
    ```
  }];

  let parameters = (ins ArrayRefParameter<"int64_t", "dimensions">:$dims);

  let extraClassDeclaration = [{
    /// Returns the rank of this tensor shape.
    size_t getRank() const { return getDims().size(); }

    /// Returns the maximum supported rank for all tensor shapes.
    static constexpr size_t getMaximumRank() { return 5; }

    /// Returns the number of dynamic dimensions in this shape.
    int64_t getNumDynamicDims() const;

    /// Return the dimension at index i in this shape, or empty
    /// if the dimension is dynamic.
    Optional<int64_t> operator[](size_t i) const;

    /// Returns the number of elements in a tensor of this shape, or empty
    /// the shape is not fully static.
    Optional<int64_t> getNumElements() const;
  }];

  let assemblyFormat = "`<` custom<ShapeDims>($dims) `>`";
  let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// TensorType
//===----------------------------------------------------------------------===//

// TODO: See also zap.ndbuffer.
def TensorType : MOPType<"Tensor", "tensor",
                         [DeclareTypeInterfaceMethods<SubElementTypeInterface>,
                          DeclareTypeInterfaceMethods<ShapedTypeInterface>]> {
  let summary = "Type of tensors over elements of given dtype in "
                "dense row-major order according to given shape";
  let description = [{
    This type is for dense CPU-friendly tensors of statically known rank and
    partially or fully statically known shape. It is parameterized over both the
    tensor shape and the element dtype. Values of this type consist of a runtime
    shape and an underlying buffer. The elements in the buffer are in dense
    row-major order according to the runtime shape, with endianness according to
    the target architecture. Dimensions in the runtime shape will match the known
    dimensions in the type's shape.

    Tensors are represented by M::Tensor in the modular runtime.

    Unlike the MLIR builtin tensor type this type:
     - Only supports a small set of dtypes.
     - Only supports shapes of known rank.
     - Does not support quantization attributes on the dtype.
     - Does not support affine maps for describing layout.
     - Is explicitly backed by the M::Tensor and M::RCBufferRef types.

    For convenience this type can be constructed using the same body syntax as the
    built-in tensor type:
    ```mlir
    !mop.tensor<4x16xfp32>
    !mop.tensor<?x?xi8>
    ```

    Note that it is not possible to write the tensor type using individual
    shape and dtype types in the MLIR syntax.
  }];

  // Note: The ShapedTypeInterface::getShape method prevents us from calling
  // the shape parameter 'shape'.
  let parameters = (ins TensorShapeType:$tensorShape, DTypeType:$dType);
  let assemblyFormat = "`<` custom<CompactShapeAndDType>($tensorShape, $dType) `>`";

  let extraClassDeclaration = [{
    /// Returns the statically known size of the tensor in bytes, or empty if the
    /// shape is not fully static.
    Optional<int64_t> getSizeInBytes() const;

    /// Returns a BufferType which captures the required size for this tensor
    /// (with no align constraint). The result will only have a size constraint
     // if the tensor's shape is fully static.
    BufferType getExpectedBufferType() const;
  }];
}

#endif  // GENERICML_MOPDIALECT_MOPTYPES_TD
