//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This defines attributes in the MOSH dialect.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MOSHDIALECT_MOSHATTRS_TD
#define GENERICML_GRAPHCOMPILER_MOSHDIALECT_MOSHATTRS_TD

include "GenericML/GraphCompiler/MOSHDialect/MOSHDialect.td"
include "KGEN/KGENDialect/KGENAttrs.td"

def MOSH_ShapeAttr : KGEN_ExprAttr<"Shape", "ape"> {
  let summary = "Shape expression attribute";
  let description = [{
    The `#mosh.ape` attribute contains a variable number of `TypedAttr`s, each
    of which have index type. The type of this attribute is always `!mosh.ape`.

    Each dimension `TypedAttr` inside `values` is one of:
    1. `KGEN::UnknownAttr` for unparameterized dimension, e.g., `?`.
    2. `KGEN::ParamDeclRefAttr` for a dimension parameter, e.g., `D0`.
    3. `KGEN::ParamOperatorAttr` for a dimension parameter expression, e.g.,
    `add(D1, 2)`.
    4. `IntegerAttr` for a concrete integer dimension, e.g., `42`

    Note that -1 can be used as a special dimension value that denotes a
    dimension to be inferred from other dimensions and total number of elements
    in the tensor. At most 1 dimension can be -1.

    Example:

    ```mlir
    kgen.param.declare N = <3>
    #mosh<ape[1, ?, N]> : !mosh.ape
    #mosh<ape[3, -1, 42]> : !mosh.ape
    ```
  }];
  let dialect = MOSHDialect;

  let parameters = (ins
    OptionalArrayRefParameter<"TypedAttr">:$values,
    AttributeSelfTypeParameter<"", "MOSH::ShapeType">:$type
  );

  let assemblyFormat = [{
    `[` (`]`) : (custom<ShapeValue>($values, ref($type))^ `]`)?
  }];

  let builders = [
    AttrBuilderWithInferredContext<(ins "ArrayRef<TypedAttr>":$values,
                                        "MOSH::ShapeType":$type), [{
      return $_get(type.getContext(), values, type);
    }]>,
    AttrBuilderWithInferredContext<(ins "ArrayRef<int64_t>":$values,
                                        "MOSH::ShapeType":$type)>,
    AttrBuilderWithInferredContext<(ins "IntArrayElementsAttr":$intDims,
                                        "MOSH::ShapeType":$type)>
  ];

  let extraClassDeclaration = [{
    /// Returns the rank (i.e. size) of the shape.
    size_t getRank() const { return getSize(); }

    /// Returns the number of elements inside the shape. This mainly serves to
    /// calls less confusing when `ShapeAttr` is used to represent a list of
    /// ints, rather than shape (e.g., bounds for a slice op).
    size_t getSize() const { return getValues().size(); }

    /// Returns the dimension attribute at given axis.
    TypedAttr operator[](int axis) const { return getValues()[axis]; }

    /// Whether the shape is static, i.e. all dims are known constants.
    bool isStatic() const { return isConstant(); }

    /// Returns the shape as integer values, using mlir::ShapedType::kDynamic
    /// for unknown dimensions.
    SmallVector<int64_t> getIntDims() const;

    /// Returns a dimension as an integer, using mlir::ShapedType::kDynamic for
    /// unknown dimensions.
    int64_t getIntDim(size_t n) const;

    /// Check whether any of the dimensions are KGEN::UnknownAttr.
    bool hasUnknownDims() const;

    /// Returns the number of elements as a symbolic expression.
    TypedAttr getSymbolicNumElements() const;

    /// Computes the common broadcast shape of the given shapes. Supports
    /// parametric and unknown shapes, as long as they can be broadcasted
    /// without constraints.
    static ErrorOr<ShapeAttr> broadcastShapes(ShapeAttr xSh, ShapeAttr ySh);

    /// Calculate the minimal common shape that all the input shape attributes
    /// can be broadcasted to. If not possible, returns a null attribute.
    static TypedAttr getCommonBroadcastShape(ArrayRef<TypedAttr> shapeParams);

    /// Calculate which dimension indices in the final bcastedShape which are
    /// the result of broadcasting. Returns indices in sorted order.
    /// E.g. [1, 3, 4, 5].
    ///
    /// A dimension is the result of broadcasting if it is a newly introduced
    /// dim or is the result of a 1 dim becoming a non-one-dim.
    ///
    /// Note, in the case we have two parameters A, B which broadcast to each
    /// other, we must return an error, as whether the dimension is broadcasted
    /// depends on runtime values.
    static ErrorOr<SmallVector<int64_t>> getBroadcastedAxes(ShapeAttr beforeShape, ShapeAttr bcastedShape);

    /// Construct a `ShapeAttr` from given integers and shape type, without
    /// considering `mlir::ShapedType::kDynamic` (i.e., all dimensions will be
    /// known integer attributes).
    /// TODO(#15938) remove once we have a better representation of symbolic ints
    static ShapeAttr fromInts(ArrayRef<int64_t> values, ShapeType type);
  }];

  let genVerifyDecl = 1;
}

#endif // GENERICML_GRAPHCOMPILER_MOSHDIALECT_MOSHATTRS_TD
