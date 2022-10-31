//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This defines attributes in the MOP dialect.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_MOPDIALECT_MOPATTRS_TD
#define GENERICML_MOPDIALECT_MOPATTRS_TD

include "GenericML/MOPDialect/MOPDialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"

// Base class for attributes in this dialect.
class MOP_Attr<string attrName, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<MOPDialect, attrName, traits> {
  let mnemonic = attrMnemonic;
}

//===----------------------------------------------------------------------===//
// MOP_BufferAttr
//
// TODO: Needs to be moved to M dialect so can be encoded in MLIRToBEF without
//       taking a dependency on GenericML
//===----------------------------------------------------------------------===//

// Make sure the attribute's data argument is interned with the requested
// alignment.
def MOP_BufferData : ArrayRefParameter<"uint8_t",
    "aligned storage for buffer data"> {
  let allocator = [{
    $_dst = copyIntoBuffer($_allocator, $_self, align);
  }];
}

def MOP_BufferAttr : MOP_Attr<"Buffer", "buffer",
                         [DeclareAttrInterfaceMethods<ElementsAttrInterface>] > {
  let summary = "A literal byte array with optional alignment";
  let description = [{
    This attribute encodes literal byte arrays which may have a specific
    alignment constraint. Nothing is implied about the internal structure of
    the bytes (rank, dimensions, underlying element type, padding, or element
    endianness).

    This attribute implements the ElementsAttrInterface as a rank-1 tensor over
    uint8_t elements.

    Generally this attribute should only be used for 'small' literals. Larger literals
    should be placed into the resources table and be referenced by the dense_resources
    attribute. Note that resources encode their alignment in their first four bytes
    using the little-endian convention.
  }];

  let parameters = (ins OptionalParameter<"Optional<int64_t>">:$align, MOP_BufferData:$data);

  let assemblyFormat = "`<` ($align^ `:`)? custom<BufferData>($data) `>`";

  let genVerifyDecl = 1;

  let extraClassDeclaration = [{
    /// ElementsAttr interface implementation.
    using ContiguousIterableTypesT = std::tuple<uint8_t>;
    FailureOr<const uint8_t*> try_value_begin_impl(OverloadToken<uint8_t>) const;
  }];
}

#endif  // GENERICML_MOPDIALECT_MOPATTRS_TD
