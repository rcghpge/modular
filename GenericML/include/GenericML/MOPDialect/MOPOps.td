//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This defines operations in the MOP dialect.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_MOPDIALECT_MOPOPS_TD
#define GENERICML_MOPDIALECT_MOPOPS_TD

include "GenericML/MOPDialect/MOPDialect.td"
include "GenericML/MOPDialect/MOPAttrs.td"
include "GenericML/MOPDialect/MOPTypes.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"

// Base class for ops in this dialect.
class MOPOp<string mnemonic, list<Trait> traits = []> :
    Op<MOPDialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// BufferConstantOp
//===----------------------------------------------------------------------===//

def BufferConstantOp : MOPOp<"buffer.constant"> {
  let summary = "Constructs a buffer whose data is aliased with the "
                "uint8_t array in the given attribute";
  let description = [{
    Constructs a buffer whose byte array is aliased with the uint8_t array
    in the given attribute. The attribute must support contiguous iteration
    over uint8_t. The attribute must agree with the buffer type's size and/or
    alignment if given.

    Example valid attributes include BufferAttr, DenseUI8ResourceElementsAttr
    and DenseI8Array. However note that DenseI8Array does not convey any
    alignment constraint.
  }];

  let arguments = (ins ElementsAttr:$value);
  let results = (outs BufferType:$result);
  let assemblyFormat = "attr-dict `:` qualified(type($result))";
  let hasVerifier = 1;
}

#endif  // GENERICML_MOPDIALECT_MOPOPS_TD
