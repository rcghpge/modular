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

/// Helps declare optional parameter declaration arguments.
def MO_ParamDecls : DefaultValuedAttr<KGEN_ParamDeclArrayAttr,
                                      "ArrayRef<ParamDeclAttr>({})">;

class MOAttr<string attrName, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<MODialect, attrName, traits> {
  let mnemonic = attrMnemonic;
}

//===----------------------------------------------------------------------===//
// MO_LayoutAttr
//===----------------------------------------------------------------------===//

def MO_LayoutAttr : MOAttr<"Layout", "layout"> {
  let summary = "An attribute that represents tensor memory layout.";
  let description = [{
    This attribute holds the memory layout information for some tensor value.
  }];

  let parameters = (ins "mlir::StringAttr":$format);

  let genVerifyDecl = 1;
  let hasCustomAssemblyFormat = 1;

  let builders = [
    AttrBuilder<(ins "mlir::StringRef" : $formatStr)>,
  ];
}

def MO_DTypeAttr : MOAttr<"DType", "dtype"> {
  let summary = "An attribute that represents tensor data type.";
  let description = [{
    This attribute holds the data type of a tensor.
  }];

  let parameters = (ins "DType":$dtype);
  let assemblyFormat = [{
    `<` $dtype `>`
  }];

  let builders = [
    AttrBuilder<(ins "DType" : $dtype)>,
  ];
}

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
  let parameters = (ins
    AttributeSelfTypeParameter<"", "MO::ChainType">:$type);

  let builders = [
    TypeBuilderWithInferredContext<(ins "MO::ChainType":$type),
      [{ return get(type.getContext(), type); }]>
  ];

  let assemblyFormat = "";
}

def MO_DeviceIndexAttr : MOAttr<"DeviceIndex", "device_index"> {
  let summary = "A pair that associates an index with a device ref attr.";
  let description = [{
    Associates an index with a device. This is useful for things like mapping
    operands of operations to devices.

    Example:
    (4 : index, <"cpu", 0>)
  }];

  let parameters = (ins
    "mlir::IntegerAttr":$index,
    "M::DeviceRefAttr":$device
  );

  let assemblyFormat = [{
    `(` $index `,` $device `)`
  }];

  let builders = [
    TypeBuilderWithInferredContext<(ins "mlir::IntegerAttr":$index, "M::DeviceRefAttr":$device),
      [{ return get(index.getContext(), index, device); }]>,
    TypeBuilderWithInferredContext<(ins "size_t":$index, "M::DeviceRefAttr":$device),
      [{
        auto ctx = device.getContext();
        auto indexTy = IndexType::get(ctx);
        auto indexAttr = IntegerAttr::get(indexTy, index);
        return get(indexAttr, device);
      }]>
  ];
}

def MO_DeviceIndexArrayAttr: ArrayOfAttr<MODialect, "DeviceIndexArrayAttr",
    "DeviceIndex", "MO::DeviceIndexAttr"> {
  let summary = "An array of DeviceIndex.";
  let description = [{
    Associates several indices with a device. This is useful for things like
    mapping operands of operations to devices.

    Example:
    #mo<DeviceIndex[(4 : index, <"cpu", 0>), (4 : index, <"cpu", 0>)]>
  }];
}

#endif // GENERICML_GRAPHCOMPILER_MODIALECT_MOATTRS_TD
