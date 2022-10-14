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
    alignment. Nothing is implied about the internal structure of the bytes,
    such as rank, dimensions, underlying element type, padding, or element
    endianness. The compiler is responsible for ensuring the producers and
    consumers of buffers agree on their interpretation of the buffer for
    the intended target.

    Generally buffers are the lowest-common denominator across kernel
    implementations. Buffers may be wrapped by various tensor types, or
    used directly.

    This type implements ShapedTypeInference as a rank-1 tensor over uint8_t
    with optimally known first dimension. This in turn supports the
    ElementsAttrInterface on buffer attributes.

    Buffers are represented by M::LLCL::RCBufferRefs in the Modular runtime.
  }];

  let parameters = (ins OptionalParameter<"Optional<int64_t>">:$size,
                        OptionalParameter<"Optional<int64_t>">:$align);
  let assemblyFormat = "`<` struct(params) `>`";
  let genVerifyDecl = 1;

  let extraClassDeclaration = [{
     /// getShape may need to return an ArrayRef representing a
     /// dynamic rank-1 shape if the size is unconstrained.
     static const int64_t unknownSize;
  }];
}

//===----------------------------------------------------------------------===//
// ContextType
//===----------------------------------------------------------------------===//

def ContextType : MOPType<"Context", "context"> {
  let summary = "Type of execution contexts holding a number of primitive "
                "contexts";
  let description = [{
    Contexts are objects created by the 'init' phase of a model, then reused
    for each call to the model's 'execute'. Contexts may contain any number
    of primitive operation contexts. A primitive operation context holds
    attributes, scratch space, pre-computed data and, if required, cached
    output buffers needed by a primitive. Each primitive may declare its
    own context structure, and each call to that primitive in the graph may
    have its own instance of that structure.

    Contexts are represented by M::MOP::Context in the Modular runtime.
  }];
}

#endif  // GENERICML_MOPDIALECT_MOPTYPES_TD
