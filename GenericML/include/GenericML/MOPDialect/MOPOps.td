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

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "GenericML/MOPDialect/MOPDialect.td"
include "GenericML/MOPDialect/MOPAttrs.td"
include "GenericML/MOPDialect/MOPTypes.td"

// Base class for ops in this dialect.
class MOPOp<string mnemonic, list<Trait> traits = []> :
    Op<MOPDialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// BufferConstantOp
//===----------------------------------------------------------------------===//

def BufferConstantOp : MOPOp<"buffer.constant",
                             [DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Constructs a buffer whose data is aliased with the "
                "`uint8_t` array in the given attribute";
  let description = [{
    Constructs a buffer whose byte array is aliased with the `uint8_t` array
    in the given attribute. The attribute must support contiguous iteration
    over `uint8_t`. The attribute must agree with the buffer type's size and/or
    alignment if given.

    Example valid attributes include `BufferAttr`, `DenseUI8ResourceElementsAttr`
    and `DenseI8Array`. However note that `DenseI8Array` does not convey any
    alignment constraint.

    Note that any align constraint of the argument is not propagated into the
    result type.

    Example:
    ```mlir
    mop.buffer.constant #mop.buffer<8: "0x00010203">
    mop.buffer.constant dense_resource<label>:!mop.buffer<size=4, align=8>
    mop.buffer.constant array<ui8: 0, 1, 2, 3>
    ```
  }];

  let arguments = (ins ElementsAttr:$value);
  let results = (outs BufferType:$result);
  let assemblyFormat = "$value attr-dict";
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    /// Returns the align constraint for the buffer, if any.
    Optional<int64_t> getAlign();
  }];
}

//===----------------------------------------------------------------------===//
// ContextNewOp
//===----------------------------------------------------------------------===//

def ContextNewOp : MOPOp<"context.new"> {
  let summary = "Constructs a context with room for primitive operation "
                "contexts";
  let description = [{
    Constructs a context with the given number of slots for primitive operation
    contexts. Generally a computational operator, such as 'lib.foo', will be
    backed by two primitives, such as 'libp.foo.setup' and 'libp.foo'. The
    former can be called by the model's 'init' to allocate and initialize
    a primitive operation context for each call to 'lib.foo' in the model.
    The latter can retrieve that primitive operation context during the model's
    'execute'.

    Example:
    ```mlir
    mop.context.new 4
    ```
  }];

  let arguments = (ins UI64Attr:$numSlots);
  let results = (outs ContextType:$result);
  let assemblyFormat = "$numSlots attr-dict";
}

//===----------------------------------------------------------------------===//
// TensorPackOp
//===----------------------------------------------------------------------===//

def TensorPackOp : MOPOp<"tensor.pack", [
           TypesMatchWith<"shape operand must match result tensor type shape",
                          "result", "tensorShape",
                          "$_self.cast<TensorType>().getTensorShape()">,
           TypesMatchWith<"buffer operand must be consistent with result tensor type",
                          "result", "buffer",
                          "$_self.cast<TensorType>().getExpectedBufferType()",
                          "BufferType::agreeOnSize">]> {
  let summary = "Constructs a MOP tensor from a shape and buffer";
  let description = [{
    Constructs a MOP tensor from a given shape and buffer.

    Example:
    ```mlir
    mop.tensor.pack %shape %buf : !mop.tensor<4x16xf32>
    ```
  }];

  let arguments = (ins TensorShapeType: $tensorShape, BufferType: $buffer);
  let results = (outs TensorType: $result);
  let assemblyFormat = "$tensorShape $buffer attr-dict `:` qualified(type($result))";
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    /// Returns the result tensor's dtype.
    DTypeType getDType();
  }];
}

//===----------------------------------------------------------------------===//
// ModelOp
//===----------------------------------------------------------------------===//

def ModelOp : MOPOp<"model", [HasParent<"ModuleOp">, Symbol,
                              IsolatedFromAbove]> {
  let summary = "Represents an executable model with 'init' and 'execute' "
                "entry points.";
  let description = [{
    This op constructs an executable model from single-block regions
    representing it's 'init' and 'execute' entry points.

    The 'init' entry point has the implicit signature:
      () -> (!bef.chain, !mop.context)

    The 'execute' entry point has the implicit signature:
      (!bef.chain, !mop.context, in1, ..., inn) ->
      (!bef.chain, res1, ..., resm)
    where:
      (in1, ..., inn) -> (res1, ..., resm)
    is given as the op's explicit signature.

    The runtime implicitly calls 'init' when the model executable is
    first loaded, and re-uses the resulting !mop.context for each
    subsequent call to 'execute'.

    Example:
    ```mlir
    mop.model @example_model
      init {
        ...
      }
      execute (%ch: !bef.chain, %ctxt: !mop.context, %arg: !mop.tensor<4x?xf32>)
              -> (!bef.chain, !mop.tensor<4x?xf32>) {
        ...
      }
    ```
  }];

  let arguments = (ins SymbolNameAttr:$sym_name, DictionaryAttr:$extraAttrs,
                       TypeAttrOf<FunctionType>:$inputResultTypes);
  let regions = (region SizedRegion<1>:$init, SizedRegion<1>:$execute);

  let extraClassDeclaration = [{
    Block *getInitBlock() { return &getInit().front(); }
    Block *getExecuteBlock() { return &getExecute().front(); }
  }];

  let builders = [
    /// Creates a new model with empty 'init' and 'execute' bodies and
    /// with `inputTypes` and `resultTypes` for the op's explicit signature.
    /// `inputArgLocs` specifies the location information for the
    /// explicit arguments to the 'execute' entry point, and thus should
    /// have the same length as `inputTypes`.
    OpBuilder<(ins "StringAttr":$sym_name,
                   "ArrayRef<Type>":$inputTypes,
                   "ArrayRef<Type>":$resultTypes,
                   "ArrayRef<Location>":$inputArgLocs,
                   CArg<"ArrayRef<NamedAttribute>", "{}">:$extraAttrs)>,
  ];

  // We can almost get away with an assemblyFormat of:
  //  "$sym_name attr-dict `init` $init `execute` $inputResultTypes $execute";
  // but we want to bind the execute region/block arguments right
  // after the 'execute' keyword and not within its region/block.
  let hasCustomAssemblyFormat = 1;
  let hasRegionVerifier = 1;
  let hasVerifier = 1;
}

#endif  // GENERICML_MOPDIALECT_MOPOPS_TD
