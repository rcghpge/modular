//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This defines operations in the MGP dialect.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_MGPDIALECT_MGPOPS_TD
#define GENERICML_MGPDIALECT_MGPOPS_TD

include "GenericML/MGPDialect/MGPOpInterfaces.td"
include "GenericML/MGPDialect/MGPTensorSpec.td"
include "GenericML/MGPDialect/MGPDialect.td"
include "GenericML/MGPDialect/MGPTypes.td"
include "GenericML/MGPDialect/MGPAttrs.td"
include "GenericML/GraphCompiler/Interfaces/EmitsMojoInterface.td"
include "GenericML/GraphCompiler/MOGGDialect/MOGGInterfaces.td"
include "GenericML/GraphCompiler/MODialect/MOAttrs.td"
include "GenericML/GraphCompiler/MODialect/MOInterfaces.td"
include "GraphRT/GRTDialect/GRTExtendsLifetimeInterface.td"
include "GraphRT/GRTDialect/GRTMayAliasInterface.td"
include "GraphRT/MEFDialect/MEFTypes.td"
include "Support/MDialect/MAttrs.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Base class for ops in this dialect.
class MGP_Op<string mnemonic, list<Trait> traits = []> :
    Op<MGPDialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// BufferConstantOp
//===----------------------------------------------------------------------===//

def BufferConstantOp : MGP_Op<"buffer.constant", [
    InferTypeOpAdaptor,
    DeclareOpInterfaceMethods<MGP_SelectiveLowerAttrOpInterface, [
      "getLiftableAttributes",
      "getIgnorableAttributes"
    ]>,
    DeclareOpInterfaceMethods<MGP_LiftableBufferCreateOpInterface, ["isTrivial"]>,
]> {
  let summary = "Allocates and initializes a buffer from a host attribute.";
  let description = [{
    Returns an `!mgp.buffer` initialized with the byte array in the given attribute
    (which will be loaded from the MEFFile holding the compiled model).

    Example:
    ```mlir
      %buf = mgp.buffer.constant
               dense_resource<label>:!M.aligned_bytes<256, align 16>
    ```
  }];

  // CAUTION: The attribute arguments must be in alphabetical order by their
  // names.
  let arguments = (ins ElementsAttr:$aValue,                             // 0
                       UI64Attr:$bRawAlign                               // 1
                  );
  let results = (outs MGP_DeviceBufferType:$outBuffer);
  let assemblyFormat = [{
    custom<ElementsAttrWithAlign>($aValue, $bRawAlign) attr-dict
  }];
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    /// Returns the buffer's alignment.
    std::optional<uint64_t> getOptAlign();
  }];

  let builders = [
    OpBuilder<(ins "mlir::ElementsAttr":$aValue)>
  ];

  let extraClassDefinition = [{
    bool BufferConstantOp::isTrivial() { return true; }

    // Lift the value
    SmallVector<Attribute> BufferConstantOp::getLiftableAttributes() {
      return {getAValueAttr()};
    }

    // Ignore the alignment.
    SmallVector<Attribute> BufferConstantOp::getIgnorableAttributes() {
      return {getBRawAlignAttr()};
    }

  }];

}

//===----------------------------------------------------------------------===//
// ContextCreateOp
//===----------------------------------------------------------------------===//

def ContextCreateOp : MGP_Op<"context.create"> {
  let summary = "Creates a context with given number of state slots.";
  let description = [{
    Creates a context with the given number of 'state slots'.

    Example:
    ```mlir
    %ctx = mgp.context.create 4
    %ctx : !mgp.context
    ```
  }];

  let arguments = (ins UI64Attr:$numSlots);
  let results = (outs MGP_ContextType:$result);
  let assemblyFormat = "$numSlots attr-dict";
}


//===----------------------------------------------------------------------===//
// TensorCreateOp
//===----------------------------------------------------------------------===//

def TensorCreateOp : MGP_Op<"tensor.create", [
    Pure,
    InferTypeOpAdaptor,
    DeclareOpInterfaceMethods<GRTMayAliasInterface>,
    DeclareOpInterfaceMethods<MGP_CreateTensorOpInterface>,
    DeclareOpInterfaceMethods<GRTExtendsLifetimeInterface>,
]> {
  let summary = "Creates an `!mgp.tensor` from an `!mgp.tensor_spec` and "
                "`!mgp.buffer`.";
  let description = [{
    Creates an `!mgp.tensor` from an `!mgp.buffer` and a `!mgp.tensor_spec`.

    If this op holds the only reference to the buffer then it is moved into
    the tensor. Otherwise a deep copy of the buffer is taken.

    Example:
    ```mlir
    %ctx : !mef.context
    %ch : !mgp.chain<<"gpu", 3>>
    %buf : !mgp.buffer<<"gpu", 3>>
    %spec : !mgp.tensor_spec<4x16xf32>
    %tensor = mgp.tensor.create %ch, %ctx[5], %buf, %spec : !mgp.tensor<4x16xf32, <"gpu", 3>>
    %tensor : !mgp.tensor<4x16xf32, <"gpu", 3>>
    ```
  }];

  let arguments = (ins ChainType:$inChain,
                       MGP_DeviceBufferType:$buffer,
                       MGP_TensorSpecType:$tensorSpec);
  let results = (outs MGP_DeviceTensorType: $outTensor);
  let assemblyFormat = [{
    $inChain `,` $buffer `,` $tensorSpec attr-dict
    `:` custom<TensorTypeToDeviceBufferSpec>(type($outTensor),
          type($buffer), type($tensorSpec), type($inChain))
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// TensorExtractBufferOp
//===----------------------------------------------------------------------===//

def TensorExtractBufferOp : MGP_Op<"tensor.extract.buffer", [
    Pure,
    InferTypeOpAdaptor,
    DeclareOpInterfaceMethods<GRTMayAliasInterface>,
    DeclareOpInterfaceMethods<GRTExtendsLifetimeInterface>,
]> {
  let summary = "Extracts the `!mgp.buffer` from an `!mgp.tensor`.";
  let description = [{
    Returns the `!mgp.buffer` component of tensor.

    If this op holds the only reference to the tensor then the buffer will be
    moved out of the tensor. Otherwise a deep copy of the buffer is returned.

    Example:
    ```mlir
    %tensor : !mgp.tensor<8x64xf32, <"gpu", 3>>
    %buf = mgp.tensor.extract.buffer %tensor : <8x64xf32, <"gpu", 3>>
    %buf : !mgp.buffer<<"gpu", 3>>
    ```
  }];

  let arguments = (ins MGP_DeviceTensorType:$tensor);
  let results = (outs MGP_DeviceBufferType:$outBuffer);
  let assemblyFormat = [{
    $tensor attr-dict `:` type($tensor)
  }];
  let hasFolder = 1;
  let hasVerifier = 1;
}


//===----------------------------------------------------------------------===//
// TensorExtractTensorSpecOp
//===----------------------------------------------------------------------===//

def TensorExtractTensorSpecOp : MGP_Op<"tensor.extract.tensor_spec", [
    Pure,
    InferTypeOpAdaptor,
]> {
  let summary = "Extracts the `!mgp.tensor_spec` from a `!mgp.tensor`.";
  let description = [{
    Returns the `!mgp.tensor_spec` of a tensor. If requested, the dtype
    can be replaced with `unk` to represent the tensor's 'shape'.

    Example:
    ```mlir
    %tensor : !mgp.tensor<8x64xf32, <"gpu", 3>>
    %shape0 = mgp.tensor.extract.tensor_spec %tensor : <8x64xf32, <"gpu", 3>>
    %shape0 : !mgp.tensor_spec<8x64xk32>
    %shape1 = mgp.tensor.extract.tensor_spec %tensor : <8x64xf32, <"gpu", 3>>
    %shape1 : !mgp.tensor_spec<8x64xunk>
    ```
  }];

  let arguments = (ins MGP_DeviceTensorType: $tensor);
  let results = (outs MGP_TensorSpecType:$outSpec);
  let assemblyFormat = [{
    $tensor attr-dict `:` type($tensor)
  }];
  let hasVerifier = 1;
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// TensorSpecCreateOp
//===----------------------------------------------------------------------===//

def TensorSpecCreateOp : MGP_Op<"tensor_spec.create", [
  Pure, InferTypeOpAdaptor
]> {
  let summary = [{
    Creates a CPU tensor spec from a mix of runtime and static dimensions and
    a statically known dtype.
  ]};
  let description = [{
    Creates a CPU tensor spec of statically known rank with dimensions taken
    from either runtime operands or statically known attributes. The element
    type must be known statically. Operands must be given for each unknown
    dimension in order. It is valid for all dimensions to be known statically,
    in which case no operands are required.

    Tensor specs are represented as M::TensorSpecs in the Modular runtime.

    Example:
    ```mlir
    %dim0 : index
    %dim1 : index

    %spec = mgp.tensor_spec.create %dim0, %dim1 : !mgp.tensor_spec<?x?xf32>
    %spec : !mgp.tensor_spec<?x?xf32>

    %spec = mgp.tensor_spec.create : !mgp.tensor_spec<32x4xf32>
    %spec : !mgp.tensor_spec<32x4xf32>
    ```
  }];

  // CAUTION: The attribute arguments must be in alphabetical order by their
  // names.
  // NOTE: The two attributes effectively encode the result's tensor spec type
  // in a form understandable by MEF.
  let arguments = (ins Variadic<Index>:$runtimeDims,
                   UI64ArrayElementsAttr:$aRawDims,  // 0
                   UI8Attr:$bRawDType);              // 1
  let results = (outs MGP_TensorSpecType:$result);
  let assemblyFormat = [{
    $runtimeDims attr-dict
    `:` custom<RawDimsAndDType>($aRawDims, $bRawDType)
  }];
  let hasVerifier = 1;

  let builders = [
    OpBuilder<(ins "TensorSpecType":$spec,
                   CArg<"ValueRange", "{}">:$runtimeDims)>
  ];
}

//===----------------------------------------------------------------------===//
// TensorSpecGetDimOp
//===----------------------------------------------------------------------===//

def TensorSpecGetDimOp : MGP_Op<"tensor_spec.get_dim", [
    Pure,
    DeclareOpInterfaceMethods<InferTypeOpInterface>,
]> {
  let summary = [{
    Return the dimension of the tensor_spec at the specified axis.
  ]};
  let description = [{
    Returns the actual runtime value of each dimension in a CPU tensor spec.
    If the dimension is known statically, it is returned as a constant index.
  }];

  let arguments = (ins MGP_TensorSpecType: $spec,
                   UI64Attr:$axis);
  let results = (outs Index:$dim);
  let assemblyFormat = [{
    `[` $axis `]` $spec attr-dict `:` type($spec)
  }];
  let hasVerifier = 1;
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// AssertOp
//===----------------------------------------------------------------------===//

def AssertOp : MGP_Op<"assert", [
    DeclareOpInterfaceMethods<MGP_SelectiveLowerAttrOpInterface, ["getLiftableAttributes"]>,
    DeclareOpInterfaceMethods<MGP_LiftableChainCreateOpInterface>,
    DeclareOpInterfaceMethods<EmitsMojoInterface>,
  ]> {
  let summary = "Assert that a boolean condition is true.";
  let description = [{
    Return an error with the given message if the condition is false.

    Example:
    ```mlir
    %ch = mgp.assert %cond, "condition is not true"
    ```
  }];

  // CAUTION: The attribute arguments must be in alphabetical order by their
  // names.
  let arguments = (ins I1:$cond, StrAttr:$message);
  let results = (outs ChainType:$outChain);
  let assemblyFormat = "$cond `,` $message attr-dict";
  let hasCanonicalizeMethod = 1;
}

//===----------------------------------------------------------------------===//
// ExecuteGenericOp
//===----------------------------------------------------------------------===//

def ExecuteGenericOp : MGP_Op<"generic.execute"> {
  let summary = "Executes a function loaded into the CPU Runtime";
  let description = [{
    Executes a function already loaded into the compiled kernel context at the
    given slot/module index within the `!mgp.context`. This will execute the function using
    the generic API. The primitive will pass in raw async values for the func
    to unpack internally.

    This execution will always return an output chain, in addition to the results
    of the kernel.  The chain can be used for error handling or side-effect dependencies.
  }];

  let arguments = (ins MGP_ContextType:$theContext,
                       Variadic<AnyType>:$args,
                       UI64Attr:$aRuntimeSlot,                         // 0
                       UI64Attr:$bModuleIndex,                         // 1
                       UI64Attr:$cFunctionIndex,                       // 2
                       StrAttr:$dSummary,                              // 3
                       DefaultValuedAttr<BoolAttr, "false">:$eIsAsync, // 4
                       ArrayAttr:$fConstantResource                    // 5
                       );
  let results = (outs Variadic<AnyType>:$outResults);
}

//===----------------------------------------------------------------------===//
// ListAppend
//===----------------------------------------------------------------------===//

def MGP_ListAppendOp : MGP_Op<"list.append", [
  // The input list elements and appended element are all shallow copied into
  // the output list.
  DeclareOpInterfaceMethods<GRTMayAliasInterface>,
]> {
  let summary = "List append Operation.";
  let description = [{
    Creates a new list that's the result of appending an element to the end of a
    copy of the input list (i.e., the op is value semantics and doesn't affect
    the input list).

    Example:

    ```mlir
    %lst : !mgp.list<!mgp.tensor<3x3xf32>>,
    %tns : !mgp.tensor<3x4xf32>,
    %t = mgp.list.append(%lst, %tns) : (
      !mgp.list<!mgp.tensor<3x3xf32>>, !mgp.tensor<3x4xf32>
    ) -> !mgp.list<!mgp.tensor<3x?xf32>>
    ```
  }];

  let arguments = (ins MGP_List:$input,
                       AnyType:$element);
  let results = (outs MGP_List:$result);

  let assemblyFormat = [{
    ` ` operands ` ` attr-dict `:` functional-type(operands, results)
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ListConcat
//===----------------------------------------------------------------------===//

def MGP_ListConcatOp : MGP_Op<"list.concat", [
  // The input elements from the two lists are shallow copied into the result list
  DeclareOpInterfaceMethods<GRTMayAliasInterface>,
]> {
  let summary = "List concat Operation.";
  let description = [{
    Creates a new list that's the result of concatenating the two input lists
    together.

    Example:

    ```mlir
    %lst : !mgp.list<!mgp.tensor<3x3xf32>>,
    %tns : !mgp.list<!mgp.tensor<3x4xf32>>,
    %t = mgp.list.concat(%lst, %tns) : (
      !mgp.list<!mgp.tensor<3x3xf32>>, !mgp.list<!mgp.tensor<3x4xf32>>
    ) -> !mgp.list<!mgp.tensor<3x?xf32>>
    ```
  }];

  let arguments = (ins MGP_List:$lhs,
                       MGP_List:$rhs);
  let results = (outs MGP_List:$result);

  let assemblyFormat = [{
    ` ` operands ` ` attr-dict `:` functional-type(operands, results)
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ListCreate
//===----------------------------------------------------------------------===//

def ListCreateOp : MGP_Op<"list.create", [
  // The inputs are shallow copied into the output list
  DeclareOpInterfaceMethods<GRTMayAliasInterface>,
]> {
  let summary = "Create a new list with given elements.";
  let description = [{
    Creates a new `!mgp.list` with the inputs as elements in the same order.

    Example:
    ```mlir
    %t1 : !mgp.tensor<3x5xf32>,
    %t2 : !mgp.tensor<3x10xf32>
    %l = mgp.list.create(%t1, %t2) : (
      !mgp.tensor<3x5xf32>, !mgp.tensor<3x10xf32>
    ) -> !mgp.list<!mgp.tensor<3x?xf32>>
    ```
  }];

  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs MGP_List:$result);
  let assemblyFormat = [{
    ` ` $inputs ` ` attr-dict `:` functional-type($inputs, results)
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ListGet
//===----------------------------------------------------------------------===//

def MGP_ListGetOp : MGP_Op<"list.get", [
  // The retrieved element is shallow copied
  DeclareOpInterfaceMethods<GRTMayAliasInterface>,
  InferTypeOpAdaptor,
]> {
  let summary = "List get Operation.";
  let description = [{
    Retrieve the element at given (base zero) index of the list. The index can be
    negative, with numpy indexing semantics, i.e., `-1` means the last element,
    `-2` means the second to last element, etc.

    Example:

    ```mlir
    %lst : !mgp.list<!mgp.tensor<3x?xf32>>,
    %idx : !mgp.tensor<si32>
    %t = mgp.list.get(%lst, %idx) :
      !mgp.list<!mgp.tensor<3x?xf32>>, !mgp.tensor<si32>
    ```
  }];

  let arguments = (ins MGP_List:$input,
                       // TODO: Should be IndexType
                       MGP_DeviceTensorType:$index);
  let results = (outs AnyType:$result);

  let assemblyFormat = [{
    ` ` operands ` ` attr-dict `:` qualified(type(operands))
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ListInsert
//===----------------------------------------------------------------------===//

def MGP_ListInsertOp : MGP_Op<"list.insert", [
  // The input list elements and inserted element are all shallow copied into
  // the output list.
  DeclareOpInterfaceMethods<GRTMayAliasInterface>,
]> {
  let summary = "List insert Operation.";
  let description = [{
    Creates a new list that's the result of inserting an element at given (base
    zero) index of a copy of the list (i.e., the op is value semantics and
    doesn't affect the input list).

    The index can be negative, with numpy indexing semantics, i.e., `-1` means
    the last element, `-2` means the second to last element, etc.

    It will push any existing element at the index to after the inserted element
    (which will now be at the index). If the index equals the length of the
    list, the op effectively appends the element to the list.

    Example:

    ```mlir
    %lst : !mgp.list<!mgp.tensor<3x3xf32>>,
    %tns : !mgp.tensor<3x4xf32>,
    %idx : !mgp.tensor<si32>
    %t = mgp.list.insert(%lst, %tns, %idx) : (
      !mgp.list<!mgp.tensor<3x3xf32>>, !mgp.tensor<3x4xf32>, !mgp.tensor<si32>
    ) -> !mgp.list<!mgp.tensor<3x?xf32>>
    ```
  }];

  let arguments = (ins MGP_List:$input,
                       AnyType:$element,
                       // TODO: Should be IndexType
                       MGP_DeviceTensorType:$index);
  let results = (outs MGP_List:$result);

  let assemblyFormat = [{
    ` ` operands ` ` attr-dict `:` functional-type(operands, results)
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ListInsert
//===----------------------------------------------------------------------===//

def MGP_ListSliceOp : MGP_Op<"list.slice", [
  // The input list elements and inserted element are all shallow copied into
  // the output list.
  DeclareOpInterfaceMethods<GRTMayAliasInterface>,
]> {
  let summary = "List slice operation.";
  let description = [{
    Creates a new list that's the result of slicing the input list.

    The slicing semantics follow those of Python

    ```mlir
      %lst   : !mgp.list<!mgp.tensor<3x3xf32>>,
      %start : !mgp.tensor<si32>
      %end   : !mgp.tensor<si32>
      %step  : !mgp.tensor<si32>
      %t = mgp.list.insert(%lst, %start, %end, %step) : (
        !mgp.list<!mgp.tensor<3x3xf32>>,
        !mgp.tensor<si32>, !mgp.tensor<si32>, !mgp.tensor<si32>
      ) -> !mgp.list<!mgp.tensor<3x3xf32>>
    ```
  }];

  let arguments = (ins MGP_List:$input,
                       MGP_DeviceTensorType:$start,
                       MGP_DeviceTensorType:$end,
                       MGP_DeviceTensorType:$step);
  let results = (outs MGP_List:$result);

  let assemblyFormat = [{
    ` ` operands ` ` attr-dict `:` functional-type(operands, results)
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// GuardOp
//===----------------------------------------------------------------------===//

def GuardOp : MGP_Op<"guard", [
  InferTypeOpAdaptor,
  DeclareOpInterfaceMethods<GRTMayAliasInterface>,
]>
{
  let summary = "SSA-value guard operation";

  let description = [{
    This operation consumes a chain and a variadic list of values and forwards
    the non-chain inputs through.

    This can be used with !mef.chain returning ops in order to establish
    execution dependencies between the chain producing op and any downstream
    operations.

    Example:

    ```mlir
      %X: !mgp.tensor<?x?xf32>
      %Y: !mgp.tensor<?x?xf32>

      %spec1_1 = mgp.tensor_spec.get_dim[1] %0 : !mgp.tensor_spec<?x?xf32>
      %spec2_0 = mgp.tensor_spec.get_dim[0] %2 : !mgp.tensor_spec<?x?xf32>

      %cond = index.cmp.eq %spec1_1, %spec2_0

      %chain = mgp.assert %cond, "D1 != D2"
      %Xp, %Yp = mgp.guard %chain, %X, %Y : !mgp.tensor<?x?xf32>, !mgp.tensor<?x?xf32>
    ```
  }];

  let arguments = (ins ChainType:$chain,
                       Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$results);

  let assemblyFormat = [{
    ` ` operands ` ` attr-dict `:` qualified(type($inputs))
  }];

  let hasVerifier = 1;
}


def ExtendLifetimeOp : MGP_Op<"extend.lifetime",
  [DeclareOpInterfaceMethods<MGP_LiftableChainCreateOpInterface>]> {
  let summary = "SSA-value guard operation";

  let description = [{
    This operation consumes a chain and a variadic list of values. Returns a
    chain when all inputs are ready. This extends the lifetime of the input
    values to match the lifetime of the input chain.
  }];

  let arguments = (ins ChainType:$gate, Variadic<AnyType>:$inputs);
  let results = (outs ChainType:$results);
  let assemblyFormat = [{
    ` ` operands ` ` attr-dict `:` functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// ModelOp
//===----------------------------------------------------------------------===//

def ModelFrameworkAttr : ConfinedAttr<StrAttr, [
    AttrConstraint<
      CPred<"CompiledFrameworkLabel::isValidFrameworkName("
            "cast<StringAttr>($_self).getValue())">,
      "must be a recognized framework name.">]>;

def ModelOp : MGP_Op<"model", [HasParent<"ModuleOp">,
                               IsolatedFromAbove, RegionKindInterface,
                               HasOnlyGraphRegion, Symbol]> {
  let summary = [{
    Represents an executable model with 'init' and 'execute' entry points.
  }];
  let description = [{
    This op creates an executable model from single-block regions
    representing its 'init' and 'execute' entry points.

    The 'init' entry point has the implicit signature:
      (!mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>, ...) ->
      (!mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>, ...)
    where:
      mgp.opaques are the weights registry and device contexts.

    The 'execute' entry point has the implicit signature:
      (!mef.chain, !mgp.context, weights_registry, dev0, ... devn, in1, ..., inn) ->
      (!mef.chain, res1, ..., resm)
    where:
      (in1, ..., inn) -> (res1, ..., resm)
    is given as the op's explicit signature. And dev0, ..., devn are the
    devices passed to init, while weights_registry is the weights registry
    passed to init.

    The Modular runtime implicitly calls 'init' when the model executable is
    first loaded, and re-uses the resulting context value for each subsequent
    call to 'execute'.

    This op is represented by a subclass of M::Model in the Modular runtime.
    The actual class is determined from the `frameworkName` attribute, which
    tools can inspect to determine which API to establish to mediate
    between framework-specific values and representations used by the model.

    Additional attributes may be supplied. Currently the following attribute
    names have special meaning:
     - argument_names
     - result_names

    Example:
    ```mlir
    // At the top-level
    mgp.model @example_model : mof { attr="value" }
      init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
        %ch = mef.chain.new
        %ctx = mgp.context.create 0
        mef.output %ch, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque, !mgp.opaque
      }
      execute (%ch: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque, %cpu: mgp.opaque, %arg: tensor<?x4x16xf32>)
              -> (!mef.chain, tensor<?x10xf32>) {
        mef.output %ch, %arg : !mef.chain, tensor<?x4x16xf32>
      }
    ```
  }];

  // CAUTION: There are three constraints on these attribute names and their
  // order:
  //  - 'sym_name' must match the standard symbol name attribute name (as given
  //    by mlir::SymbolTable::getSymbolAttrName()). Note however that we could
  //    probably choose a more consistent name and implement
  //    DerivedAttributeOpInterface to recover name if we really wanted
  //    that.
  //  - The attributes are accessed by index within M::Module subclasses and
  //    MEF metadata extraction, and that index matches the MLIR canonical
  //    asciibetical order.
  //  - The extra attributes must appear last since optional.
  let arguments = (ins M_DeviceRefArrayAttr:$devices,                   // 0
                       ModelFrameworkAttr:$frameworkName,               // 1
                       TypeAttrOf<FunctionType>:$inputOutputTypes,      // 2
                       SymbolNameAttr:$sym_name,                        // 3
                       UnitAttr:$isSubgraph,                            // 4
                       OptionalAttr<DictionaryAttr>:$theExtraAttrs);    // 5
  let regions = (region SizedRegion<1>:$init, SizedRegion<1>:$execute);

  let extraClassDeclaration = [{
    Block *getInitBlock() { return &getInit().front(); }
    Block *getExecuteBlock() { return &getExecute().front(); }

    /// Get the execute block. This function is required to use with
    /// our internal callgraph utilities.
    Region &getBodyRegion() { return getExecute(); }

    MutableArrayRef<BlockArgument> getInitDevices() {
      return getInitBlock()->getArguments().drop_front(kMgpModelNumFixedImplicitInitArgs).take_front(getDevices().size());
    }
    OperandRange getInitOutputDevices() {
      return getInitBlock()->getTerminator()->getOperands().drop_front(
        // Fixed implicit execute args are also fixed implicit init output args.
        kMgpModelNumFixedImplicitExecArgs
      ).take_front(getDevices().size());
    }

    BlockArgument getInitWeightsRegistry() {
      return getInitBlock()->getArgument(0);
    }

    MutableArrayRef<BlockArgument> getExecuteInputs() {
      return getExecuteBlock()->getArguments().drop_front(kMgpModelNumFixedImplicitExecArgs + getDevices().size());
    }
    MutableArrayRef<BlockArgument> getExecuteChainContext() {
      return getExecuteBlock()->getArguments().take_front(2);
    }
    MutableArrayRef<BlockArgument> getExecuteDevices() {
      return getExecuteBlock()->getArguments().drop_front(kMgpModelNumFixedImplicitExecArgs).take_front(getDevices().size());
    }
    MutableArrayRef<BlockArgument> getExecuteImplicitArguments() {
      return getExecuteBlock()->getArguments().take_front(kMgpModelNumFixedImplicitExecArgs + getDevices().size());
    }
    BlockArgument getExecuteChain() { return getExecuteBlock()->getArgument(0); }
    BlockArgument getExecuteWeightsRegistry() { return getExecuteBlock()->getArgument(2); }
    BlockArgument getSubgraphIndex() { return getExecuteBlock()->getArguments().back(); }

    /// Retrieve the single ContextCreateOp from the init block.
    ContextCreateOp getInitContext();

    /// Retrieve the context for the execute block.
    TypedValue<MGP::ContextType> getExecuteContext();


    StringAttr getName() { return getSymNameAttr(); }
  }];

  let builders = [
    /// Creates a model with empty 'init' and 'execute' bodies and
    /// with `inputTypes` and `outputTypes` for the op's explicit signature.
    /// `inputArgLocs` specifies the location information for the
    /// explicit arguments to the 'execute' entry point, and thus should
    /// have the same length as `inputTypes`.
    OpBuilder<(ins "StringAttr":$sym_name,
                   "StringRef":$frameworkName,
                   "ArrayRef<DeviceRef>":$devices,
                   "ArrayRef<Type>":$inputTypes,
                   "ArrayRef<Type>":$outputTypes,
                   "ArrayRef<Location>":$inputArgLocs,
                   CArg<"bool", "false">:$isSubgraph,
                   CArg<"ArrayRef<NamedAttribute>", "{}">:$extraAttrs)>
  ];

  let assemblyFormat = [{
    $sym_name (`subgraph` $isSubgraph^)?`:` custom<ClassName>($frameworkName)
      ($theExtraAttrs^)?
      `devices` $devices
      `init` custom<InitRegion>($init)
      `execute` custom<ExecuteRegion>($inputOutputTypes, $execute, ref($devices))
      attr-dict
  }];
  let hasRegionVerifier = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// RuntimeCreateOp
//===----------------------------------------------------------------------===//

def RuntimeCreateOp : MGP_Op<"runtime.create", [
    DeclareOpInterfaceMethods<MGP_HasRuntimeSlotOpInterface>,]> {
  let summary = "Creates a runtime to coordinate execution.";
  let description = [{
    Creates a runtime within an `!mgp.context` slot.

    Note that this operation accepts and returns `!mef.chain`s since it is
    part of device setup on the host.

    Example:
    ```mlir
      // Establish a runtime in slot 5.
      %ch0 : !mef.chain
      %ctx : !mgp.context
      %ch1 = mgp.runtime.create %ch0, %ctx[5]
      %ch1 : !mef.chain
    ```
  }];

  // CAUTION: The attribute arguments must be in alphabetical order by their
  // names.
  let arguments = (ins ChainType:$inChain,
                       MGP_ContextType:$theContext,
                       UI64Attr:$aRuntimeSlot);         // 1
  let results = (outs ChainType:$outChain);
  let assemblyFormat = [{
    $inChain `,` $theContext `[` $aRuntimeSlot `]` attr-dict
  }];

  let extraClassDefinition = [{
    uint64_t RuntimeCreateOp::getSlot() { return getARuntimeSlot(); }

  }];
}

//===----------------------------------------------------------------------===//
// BufferGetCachedOp
//===----------------------------------------------------------------------===//

def BufferGetCachedOp : MGP_Op<"buffer.get_cached", [Pure]> {
  let summary = "Retrieves an `!mgp.buffer` from a dedicated state slot.";
  let description = [{
    Returns (a borrow of) the `!mgp.buffer` in the state slot with the given
    index. The buffer should have already been cached by `mgp.buffer.cache`.
    Returns an error if the cached buffer's size (if any) or device do not
    agree with the `!mgp.buffer` type annotation.

    Note this op is gated by a `!mef.chain` since the cache is held on the
    host.

    Example:
    ```mlir
    %ctx : !mgp.context
    %ch0 : !mef.chain
    %idx : index
    // Returns the buffer cached in slot index.
    %buf = mgp.buffer.get_cached %ch0, %ctx[%idx] : !mgp.buffer<<"gpu", 3>>
    %buf : !mgp.buffer<<"gpu", 3>>
    ```
  }];

  let arguments = (ins ChainType:$inChain,
                       MGP_ContextType:$theContext,
                       Index:$bBufferSlot);         // 1
  let results = (outs MGP_DeviceBufferType:$outBuffer);
  let assemblyFormat = [{
    $inChain `,` $theContext `[` $bBufferSlot `]` attr-dict
    `:` qualified(type($outBuffer))
  }];

  let extraClassDefinition = [{
  }];
}

//===----------------------------------------------------------------------===//
// BufferRemoveCachedOp
//===----------------------------------------------------------------------===//

def BufferRemoveCachedOp : MGP_Op<"buffer.remove_cached", [
  DeclareOpInterfaceMethods<MGP_SelectiveLowerAttrOpInterface, ["getLiftableAttributes"]>]> {
  let summary = "Removes the buffer stored at the specified state slot.";
  let description = [{
    Removes the buffer stored at the specified state slot.
    The buffer is freed if there are no more uses.

    WARNING: LowerAlignmentsPass cannot track buffer alignments through the
    cache, so do not replace a buffer in a cache with a new buffer that has
    different alignment.
    This op is currently only inserted by DynamicMemoryPlanningPass which runs
    after LowerAlignmentsPass, which is OK.
  }];

  // CAUTION: The attribute arguments must be in alphabetical order by their
  // names.
  let arguments = (ins ChainType:$inChain,
                       MGP_ContextType:$theContext,
                       UI64Attr:$bufferSlot          // 1
                       );
  let results = (outs ChainType:$outChain);
  let assemblyFormat = [{
    $inChain `,` $theContext `[` $bufferSlot `]`
    attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// BufferGetSizeOp
//===----------------------------------------------------------------------===//

def BufferGetSizeOp : MGP_Op<"buffer.get_size", [
    DeclareOpInterfaceMethods<InferTypeOpInterface>,
]> {
  let summary = "Get the runtime size of a buffer in bytes.";
  let description = [{
    Returns the size of the input buffer in bytes.

    Example:
    ```mlir
    %in_buf : !mgp.buffer
    %buf_size = mgp.buffer.get_size %in_buf
    %buf_size : index
    ```
  }];

  let arguments = (ins ChainType:$inChain, MGP_DeviceBufferType:$buffer);
  let results = (outs  Index:$size);
  let assemblyFormat = [{
    $inChain `,` $buffer attr-dict
    `:` qualified(type($buffer))
  }];
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Async Ref Ops
//===----------------------------------------------------------------------===//

def MGP_GetInputAsyncValuesOp : MGP_Op<"get_input_async_vals", [Pure]> {
  let results = (outs Variadic<AnyType>:$result);
  let summary = "Returns pointers for each of the async value inputs ";
}

def MGP_GetOutputAsyncValuesOp : MGP_Op<"get_output_async_vals", [Pure]> {
  let results = (outs Variadic<AnyType>:$result);
  let summary = "Returns pointers for each of the async value outputs ";
}

def MGP_AllocateAsyncValuesOp : MGP_Op<"allocate_for_async_vals", [Pure]> {
  let results = (outs Variadic<AnyType>:$result);
  let summary = "Allocates pointers for each of the async value";
}

//===----------------------------------------------------------------------===//
// DebugPrintOp
//===----------------------------------------------------------------------===//

def DebugPrintOp : MGP_Op<"debug.print"> {
  let summary = "Prints a debug string. For debugging and testing only.";
  let description = [{
    Prints a debug string. If a label attribute is supplied
    the string is printed with that label.

    For debugging and testing only.

    Note this op accepts and returns `!mef.chain`s since printing is managed
    by the host.

    Example:
    ```mlir
    %ch0: !mef.chain
    %ctx: !mgp.context
    %ch1 = mgp.debug.print %ch0, %ctx, "debug string", "label"
    %ch1: !mef.chain
    ```
  }];

  // CAUTION: The attribute arguments must be in alphabetical order by their
  // names.
  let arguments = (ins ChainType:$inChain,
                       MGP_ContextType:$theContext,
                       StrAttr: $aDebugString,                        // 0
                       DefaultValuedAttr<StrAttr, "\"\"">: $bLabel);  // 1
  let results = (outs ChainType: $outChain);
  let assemblyFormat = [{
    $inChain `,` $theContext `,` $aDebugString ( `,` $bLabel^ )? attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

def MGP_CallOp : MGP_Op<"call"> {
  let summary = "Call a subgraph.";
  let description = [{
    Placeholder for the execute block of another graph, which will be inlined by
    `InlineMGPCallsPass`.
    The graph is referenced via symbol name of the target !mgp.model.

    Operands and results must match the signature of the target model's execute block.

    Example:
    ```mlir
    mgp.model @subgraph : mof {
      init (...) {
        // ...
      }
      execute (%ch: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque, %cpu: mgp.opaque, %arg: tensor<?x4x16xf32>)
              -> (!mef.chain, tensor<?x10xf32>) {
        // ...
      }
    }

    mgp.model @main : mof {
      init (...) {
        // ...
      }
      execute (%ch: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque, %cpu: mgp.opaque, %arg: tensor<?x4x16xf32>)
              -> (!mef.chain, tensor<?x10xf32>) {
        %sub_res = mgp.call @subgraph(%ch, %ctx, %weights, %cpu, %arg)
      }
    }
    ```
  }];
  let arguments = (ins FlatSymbolRefAttr:$model,
                       DefaultValuedAttr<StrAttr, "\"\"">:$prefix,
                       Variadic<AnyType>:$operands,
                       DefaultValuedAttr<BoolAttr, "false">:$isGlobalizedIndex);
  let results = (outs Variadic<AnyType>:$results);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $model `(` $operands `)` attr-dict `:` functional-type($operands, results)
  }];

  let extraClassDeclaration = [{
    FlatSymbolRefAttr getCalleeSymbol() {
      return cast<FlatSymbolRefAttr>(getModelAttr());
    }
  }];
}

include "GenericML/MGPDialect/MGPDeviceOps.td"

#endif  // GENERICML_MGPDIALECT_MGPOPS_TD
