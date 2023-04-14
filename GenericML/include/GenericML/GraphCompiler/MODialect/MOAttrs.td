//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This defines attributes in the MO dialect.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MODIALECT_MOATTRS_TD
#define GENERICML_GRAPHCOMPILER_MODIALECT_MOATTRS_TD

include "GenericML/GraphCompiler/MODialect/MODialect.td"
include "KGEN/KGENDialect/KGENAttrs.td"

def MO_ChainAttr : KGEN_ExprAttr<"Chain", "chain"> {
  let summary = "Chain expression attribute";

  let description = [{
    Represents non-error chain values. The type of this attribute is always
    `!mo.chain`.

    Example:

    ```mlir
    #mo<chain> : !mo.chain
    ```
  }];

  let dialect = MODialect;
  let parameters = (ins AttributeSelfTypeParameter<"", "MO::ChainType">:$type);

  let builders = [TypeBuilderWithInferredContext<(ins "MO::ChainType":$type), [{
    return get(type.getContext(), type);
  }]>];

  let assemblyFormat = "";
}

def MO_ShapeAttr : KGEN_ExprAttr<"Shape", "shape"> {
  let summary = "Shape expression attribute";
  let description = [{
    The `#mo.shape` attribute contains a variable number of `TypedAttr`s, each
    of which have index type. The type of this attribute is always `!mo.shape`.

    Note that -1 can be used as a special dimension value that denotes a
    dimension to be inferred from other dimensions and total number of elements
    in the tensor. At most 1 dimension can be -1.

    Example:

    ```mlir
    kgen.param.declare N = <3>
    #mo<shape[1, ?, N]> : !mo.shape
    #mo<shape[3, -1, 42]> : !mo.shape
    ```
  }];
  let dialect = MODialect;

  let parameters = (ins OptionalArrayRefParameter<"TypedAttr">:$values,
                        AttributeSelfTypeParameter<"", "MO::ShapeType">:$type);

  let assemblyFormat = [{
    `[` (`]`) : (custom<ShapeValue>($values, ref($type))^ `]`)?
  }];

  let builders = [
    AttrBuilderWithInferredContext<(ins "ArrayRef<TypedAttr>":$values,
                                        "MO::ShapeType":$type), [{
      return $_get(type.getContext(), values, type);
    }]>,
    AttrBuilderWithInferredContext<(ins "ArrayRef<int64_t>":$values,
                                        "MO::ShapeType":$type)>,
    AttrBuilderWithInferredContext<(ins "IntArrayElementsAttr":$intDims,
                                        "MO::ShapeType":$type)>
  ];

  let extraClassDeclaration = [{
    /// Returns the rank (i.e. size) of the shape.
    size_t getRank() const { return getValues().size(); };

    /// Whether the shape is static, i.e. all dims are known constants.
    bool isStatic() const { return isConstant(); };

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
  }];

  let genVerifyDecl = 1;
}

/// Helps declare optional parameter declaration arguments.
def MO_ParamDecls : DefaultValuedAttr<KGEN_ParamDeclArrayAttr,
                                      "ArrayRef<ParamDeclAttr>({})">;

#endif // GENERICML_GRAPHCOMPILER_MODIALECT_MOATTRS_TD
