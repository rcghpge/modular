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

def MO_ShapeAttr : KGEN_ExprAttr<"Shape", "shape"> {
  let summary = "shape expression attribute";
  let description = [{
    The `#mo.shape` attribute contains a variable number of `TypedAttr`s, each
    of which have index type. The type of this attribute is always `!mo.shape`.

    Example:

    ```mlir
    kgen.param.declare N = <3>
    #mo<shape[1, ?, N]> : !mo.shape
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
                                        "MO::ShapeType":$type)>
  ];

  let extraClassDeclaration = [{
    /// Returns the rank (i.e. size) of the shape.
    size_t getRank() const { return getValues().size(); };

    /// Whether the shape is static, i.e. all dims are known constants.
    bool isStatic() const { return isConstant(); };

    /// Convenience function returning the type as integer values, using
    /// mlir::ShapedType::kDynamic for unknown dimensions.
    SmallVector<int64_t> getIntDims() const;
  }];

  let genVerifyDecl = 1;
}

#endif // GENERICML_GRAPHCOMPILER_MODIALECT_MOATTRS_TD
