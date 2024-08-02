//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This defines operations in the MOGG dialect.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOOPS_TD
#define GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOOPS_TD

include "GenericML/GraphCompiler/MOGGDialect/MOGGDialect.td"
include "GenericML/GraphCompiler/MOGGDialect/MOGGInterfaces.td"
include "GenericML/GraphCompiler/MOGGDialect/MOGGTypes.td"
include "GenericML/MGPDialect/MGPTypes.td"
include "GenericML/GraphCompiler/MODialect/MOTypes.td"
include "GenericML/GraphCompiler/MODialect/MOAttrs.td"
include "GenericML/GraphCompiler/MODialect/MOInterfaces.td"
include "GenericML/GraphCompiler/Interfaces/LowerableToMGPInterface.td"
include "GenericML/MGPDialect/MGPOpInterfaces.td"
include "KGEN/KGENDialect/KGENInterfaces.td"
include "Support/MDialect/MAttrs.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "GraphRT/MEFDialect/MEFTypes.td"

// Base class for ops in this dialect.
class MOGG_Op<string mnemonic, list<Trait> traits = []>
    : Op<MOGGDialect, mnemonic, traits>;

class MOGG_LowerableOp<string mnemonic, list<Trait> traits = []>
    : MOGG_Op<mnemonic, traits # [DeclareOpInterfaceMethods<LowerableToMGPInterface, ["lower"]>]>;

//===----------------------------------------------------------------------===//
// Internal helper ops
//===----------------------------------------------------------------------===//

def MOGG_ConcatOp : MOGG_LowerableOp<"concat", []> {
  let summary = "Concat Operation along axis zero";
  let description = [{
    Concat input tensors along axis zero.

    Example:

    ```mlir
    %arg0: !mo.tensor<[3, 3], f32>
    %arg1: !mo.tensor<[2, 3], f32>
    %res = mogg.concat(%arg0, %arg1) : (
      !mo.tensor<[3, 3], f32>, !mo.tensor<[2, 3], f32>
    ) -> !mo.tensor<[5, 3], f32>
    ```

    Currently only for the 'host' device.
    TODO: Generalize to arbitrary devices.
  }];

  let arguments = (ins Variadic<MO_Tensor>:$inputs);
  let results = (outs MO_Tensor:$result);

  let assemblyFormat = [{
    `(` $inputs `)` attr-dict `:` functional-type($inputs, results)
  }];
}

def MOGG_SplitOp : MOGG_LowerableOp<"split", []> {
  let summary = "Split Operation along axis zero";
  let description = [{
    Split input tensors into multiple tensors along axis zero.

    Each of the output tensors has the same shape as `input` except along axis
    zero where the sum of the zero dimensions from all output tensors should be
    equal to the one of the input tensor. Note that this op is a special case
    of `mo.split` where `axis` is zero. The `split_sizes` is implicitly defined
    by `mogg.split`'s output tensors.

    Example:

    ```mlir
    %input: !mo.tensor<[9, 8], f32>
    %res = mogg.split(%input) : (!mo.tensor<[9, 8], f32> ) ->
    (!mo.tensor<[2, 8], f32>, !mo.tensor<[3, 8], f32>, !mo.tensor<[4, 8], f32>)
    ```

    Currently only for the 'host' device.
    TODO: Generalize to arbitrary devices.
  }];

  let arguments = (ins MO_Tensor:$input);
  let results = (outs Variadic<MO_Tensor>:$results);

  let assemblyFormat = [{
    `(` $input `)` attr-dict `:` `(`
      qualified(type($input)) `)` `->` `(` type($results) `)`
  }];
}

//===----------------------------------------------------------------------===//
// OutputOp
//===----------------------------------------------------------------------===//

def MOGG_OutputOp
    : MOGG_Op<"output", [Terminator]> {
  let summary = "MOGG output operation";
  let description = [{
    The "mogg.output" operation specifies the output values for a region in a
    MOGG operation. It is intendend to allow the regions of mogg operations to
    grow in complexity with specific outputs.

    ```mlir
      mogg.op foo() {
      ... mogg.output e0, % 1 : i32, f8
    } : i32, f8
    ```
  }];

  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}


//===----------------------------------------------------------------------===//
// MOGG abstract compiled components representation
//===----------------------------------------------------------------------===//

def MOGG_KernelOp : MOGG_LowerableOp<"kernel", [
    IsolatedFromAbove,
    SingleBlock,
    SymbolTable,
    DeclInterface,
    DeclareOpInterfaceMethods<MO_ParamDeclaration>,
    DeclareOpInterfaceMethods<MOGG_CompilableKernel>,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    MO_HasDeviceInterface]> {
  let summary = "A representation of a kernel which can be fused into and compiled";
  let description = [{
    This kernel represents the handover from the MO graph to the KGEN land.

    The operation itself is a fully legal graph semantic operation which
    composes with all other operations at the MO level.

    Inside the kernel however the semantics are mirrors of the KGEN semantics.
    We operate on lambdas and kgen function calls. This lets us have both inline
    in the same graph while keeping the hard dependencies contained to lowering.

    The kernel abstraction also exposes a number of hooks. These hooks are
    details of the fusion algorithms. They allow this kernel to express the
    semantics by which its outputs or inputs may be folded by being fused into.

    Example:

    ```mlir
    %1 = mogg.kernel (%0, %arg1) : (%arg2: !mo.tensor<[2, 3], f32>, %arg3: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], f32>{
      ...
    } {fusionBodyLambda = "elemwise_compute_lambda", inputFusionInterface = ["input_0_fn", "input_1_fn"], outputFusionInterface = ["output_0_fn"]}
    ```

    This above kernel has a graph semantic %1 and inputs %0, %arg1 which work
    with the rest of the MO/MOGG operations.  Internally it works on `(%arg2:
    !mo.tensor<[2, 3], f32>, %arg3: !mo.tensor<[2, 3], f32>)` which essentially
    are the KGEN view of those values.

    `outputParamDecls` allows this compiled operation to define parameter
     values. These values are explicit parameter returns of the kernel. They
     cannot be referred to within the kernel other than to set them. When the
     kernel is executed they will be updated in the MGP parameter map. This is
     completely seperate from the normal kernel tensor returns. Their values
     are set within the kernel using `SetParameterOp`.

    The attributes at the end.
    ```mlir
    { fusionBodyLambda = "elemwise_compute_lambda",
      inputFusionInterface = ["input_0_fn", "input_1_fn"],
      outputFusionInterface = ["output_0_fn"]
    }
    ```
    `fusionBodyLambda` Represents the fusable elementwise / view operation body
    contained within. This signifies that this kernel is fully implemented by
    applying this lambda to an otherwise identity operation, therefore we can
    safely fuse it into a user.  Most ops will not have this, only elementwise
    and view ops. After being fused into an op that doesn't have `fuseHook` it
    will be lost and can no longer be fused. E.G elementwise into matmul.

    `inputFusionInterface` Represents the lambdas within this kernel that are
    responsible for loading each input. The name doesn't matter, we know which
    input this matches by the index in the attr array vs the index of the
    operand in the block.

    `outputFusionInterface` Is the same as the above but for storing.

    `isViewOp` Is an easy way for us to check if something can be represented as
    a view, e.g., a broadcast in the form of a view change followed by a copy. We
    can drop the copy and keep the view change op.

    `isKGEN` Is an attribute which marks the MOGG kernel as being represented
    using KGEN ops rather than a kernel composed of mogg ops.

    `allocIndices` is a list attribute containing the indices of the output tensors
    that have in-place allocation. For example, allocIndices = [0] means the first
    output tensor has been allocated in the kernel.

    `isInPlace` Is a bool attribute marking whether the kernel has a `mo.buffer` input
    it operates on in-place.

    Kernels preserve the device of their original MO op.
  }];

  let arguments = (ins Variadic<AnyTypeOf<[MO_Tensor, MO_List, MO_Opaque, MO_Buffer, MO_Chain]>>:$operands,
    MO_ParamDecls:$outputParamDecls,
    OptionalAttr<ArrayAttr>:$inputFusionInterface,
    OptionalAttr<ArrayAttr>:$outputFusionInterface,
    OptionalAttr<StrAttr>:$fusionBodyLambda,
    OptionalAttr<BoolAttr>:$isViewOp,
    OptionalAttr<BoolAttr>:$isKGEN,
    OptionalAttr<StrAttr>:$summary,
    OptionalAttr<BoolAttr>:$isAsync,
    OptionalAttr<ArrayAttr>:$allocIndices,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$isInPlace
  );
  let results = (outs Variadic<AnyTypeOf<[MO_Tensor, MO_Opaque, MO_Buffer, MO_Chain]>>:$results);
  let regions = (region SizedRegion<1>:$body);
  let hasVerifier = 1;
  let hasCanonicalizeMethod = 1;

  let assemblyFormat = [{
    custom<Kernel>($operands, type($operands), type($results), $body, $outputParamDecls) attr-dict
  }];


  let extraClassDeclaration = [{

    /// Sometimes it is necessary to temporarily pretend that a kernel has
    /// fusion enabled however we still need to distinguish between those
    /// which have fusion disabled by the user and those which we have
    /// temporarily marked. These must be removed before the verifier runs.
    static constexpr StringLiteral getTempLambdaName() {
      return "__TMP__FUSE_HOOK";
    }

    /// Strip any of the temporary fusion lambda markers.
    void removeTemporaryFusionMarks();

    /// Returns the OutputPlaceholder for this operation or null if it doesn't
    /// exist.
    OutputPlaceholdersOp getOutputPlaceholder();

    // Returns the StackAllocateTensorsOp for this operation or null if it
    // does not exist.
    StackAllocateTensorsOp getStackAllocateTensor();

    // Returns the CreateEmptyTensorsOp for this operation or null if it does
    // not exist.
    CreateEmptyTensorsOp getCreateEmptyTensor();

    // Returns the OutputOp for this operation or null if it does not exist.
    OutputOp getOutputOp();

    /// Disables future fusion on this input.
    void disableInputFusionOnInput(size_t index);

    /// Removes an operand at the given index and disables input fusion on that
    /// position if this kernel otherwise supports it.
    void removeOperandAtIndex(size_t index);

    /// Add an operand to the kernel and adds an argument to the kernel region.
    /// Fixes up the input fusion interfaces to disallow input fusion on this
    /// position. Returns the added kernel argument.
    Value addOperand(Value operand);

    /// Mark a given input as having a lambda fusion interface
    void setInputFusionInterface(size_t index, StringRef lambdaName);

    /// Fuse this body into the given builder and symbol table, using the
    /// remapper to update the SSA values and tracking the changed lambda
    /// names in `oldLambdaNamesToNew`.
    void fuseBodyIntoNew(OpBuilder& builder, mlir::SymbolTable& symTable,
        IRMapping& remapper, llvm::StringMap<StringRef>& oldLambdaNamesToNew);

    /// The same as the above but adds the option to skip an operation in the
    /// body.
    void fuseBodyIntoNewWithSkip(OpBuilder& builder, mlir::SymbolTable& symTable,
        IRMapping& remapper, llvm::StringMap<StringRef>& oldLambdaNamesToNew,
        std::function<bool(Operation*)> shouldSkip);

    /// Sets the insertion point to the start of the kernel. This may not be
    /// the precise start as it will place the operation after the output
    /// placeholders if they exist.
    void setInsertionPointToTop(OpBuilder& builder);

    /// Returns the KGEN input parameters this depends on. Necessary to
    /// implement the DeclInterface.
    ArrayRef<ParamDeclAttr> getInputParams() { return {}; }
    ArrayRef<ParamDeclAttr> getResultParams() { return {}; }

    /// Finds a lambda in the body of the op.
    KGEN::ParamDeclareRegionOp findLambda(StringAttr name);

    /// Returns an input lambda or an empty string if it does not exist.
    StringAttr getInputFusionInterfaceAt(size_t index);

    /// Returns an output lambda or an empty string if it does not exist.
    StringAttr getOutputFusionInterfaceAt(size_t index);

    /// Get the indices of the output tensors that are in-place allocated
    DenseSet<int> getInplaceAllocIndicesSet();

    /// Returns whether the kernel is a in-place allocated kernel
    bool isInPlaceAlloc();

    /// Returns whether the kernel operates on one of its inputs in-place.
    bool isInPlace();

    /// Returns the sole input chain (if present) and std::nullopt otherwise.
    std::optional<Value> getInChain();

    /// Returns the sole output chain (if present) and std::nullopt otherwise.
    std::optional<Value> getOutChain();
  }];
}

def MOGG_GenericCompiledRegionOp : MOGG_Op<"generic.compiled.region", [
    IsolatedFromAbove,
    SingleBlock,
    SymbolTable,
    DeclareOpInterfaceMethods<MOGG_CompilableKernel>,
    MO_HasDeviceInterface,
    DeclareOpInterfaceMethods<MGP_InOutSemanticsOpInterface>,
    DeclInterface
]> {
  let summary = "An abstract function representing a compiled body of code";
  let description = [{
    This function takes and returns any number of AsyncValues and operates on
    them by compiling the code contained within its body.

    Currently this is always blocking and the code will run to completion
    within the body.

    This is a more generic version of the MOGG.kernel which specifically
    operates on Tensors and other MO level structures. It is intended that
    eventually we lower mogg.kernel into this form directly. The first
    return output is always a mef.chain for now.

    `canFuseIntoOther`, `canBeFusedInto`, and `canDuplicate` dictate how
    sequence fusion (e.g. fusion which just concatenates code without
    additional optimization) can interact with the compiled region.

    Sequence fusion looks at potential `parent --> child` relationships
    between compiled regions and tries to fuse the parent into the child.
    `canFuseIntoOther` determines if this region can fuse as `parent`.
    `canBeFusedInto` determines if this region can fuse as `child`.
    `canDuplicate` determines whether we can duplicate kernels in the
    fusion process (usually small ones). E.g. consider a region A with
    children B and C, if `canDuplicate` for A, then we may allow A
    to inline into B AND C, effectively copying the code for A twice.
  }];

  let arguments = (ins
    Variadic<AnyType>:$operands,
    MO_ParamDecls:$outputParamDecls,
    M_InOutSignatureAttr:$inOutSignature,
    OptionalAttr<BoolAttr>:$isKGEN,
    OptionalAttr<StrAttr>:$summary,
    OptionalAttr<BoolAttr>:$isAsync,
    OptionalAttr<BoolAttr>:$hasCuda,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$canFuseIntoOther,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$canBeFusedInto,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$canDuplicate
  );
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{
    custom<Kernel>($operands, type($operands), type($results), $body, $outputParamDecls) attr-dict
  }];

  let extraClassDeclaration = [{
    /// Returns the KGEN input parameters this depends on. Necessary to
    /// implement the DeclInterface.
    ArrayRef<ParamDeclAttr> getInputParams() { return {}; }
    ArrayRef<ParamDeclAttr> getResultParams() { return {}; }
  }];
}

def MOGG_ParamShapeToValueOp : MOGG_Op<"param_shape_to_value", [Pure]> {
  let summary = "Returns a runtime representation of a given shape";
  let description = [{
    Returns a runtime representation for a shape.
  }];

  let results = (outs MOGG_Shape:$result);

  let assemblyFormat = [{
    attr-dict `->` type(results)
  }];
}

def MOGG_ShapeFromKGENOp : MOGG_Op<"shape_from_kgen", [Pure]> {
  let summary = "Gets a MOGG shape construct from KGEN";
  let description = [{
    Given a KGEN value which represents a shape launders that into MOGG
    representation.
  }];


  let arguments = (ins AnyType:$shape);
  let results = (outs MOGG_Shape:$result);

  let assemblyFormat = [{
    attr-dict `(` $shape `:` type($shape) `)` `->` type(results)
  }];
}

def MOGG_SetParameterOp : MOGG_Op<"set_param"> {
  let summary = "Assign a param a value from the shape at index";
  let description = [{
    Gives the param the value from the shape at the constant index.
    `$param = $shape[$index]`

    ```
    %shape = mogg.param_shape_to_value -> !mogg.shape<[50, 150]>
    mogg.set_param(%shape : <[50, 150]>) {index = 0 : i64, paramName = "M"}
    mogg.set_param(%shape : <[50, 150]>) {index = 1 : i64, paramName = "N"}
    ```
    This sets `M` to the value from `%shape` at the index `0` which is `50`
    and set `N` to the value at index `1` which is `150`.
  }];

  let arguments = (ins
     MOGG_Shape:$shape,
     Builtin_StringAttr:$paramName,
     Builtin_IntegerAttr:$index
  );

   let assemblyFormat = [{
      `(` $shape `:` type($shape) `)` attr-dict
   }];
}

def MOGG_ShapeToTensorOp : MOGG_Op<"shape_to_tensor"> {
  let summary = "Convert a MOGG shape value to a tensor value";
  let description = [{
    Convert the input `!mogg.shape` value to a `!mo.tensor`, where elements of
    the output tensor will have the same value as individual dimensions in the
    input shape.

    Example:
    ```
    %shape  = mogg.param_shape_to_value -> !mogg.shape<[50, D1, 150]>
    %tensor = mogg.shape_to_tensor(%shape) : !mogg.shape<[50, D1, 150]> -> !mo.tensor<[3], si64>
    ```
  }];

  let arguments = (ins MOGG_Shape:$shape);
  let results = (outs MO_Tensor:$result);

  let assemblyFormat = [{
     `(` $shape`)` attr-dict `:` type($shape) `->` type($result)
  }];
  let hasVerifier = 1;
}

def MOGG_GetShapeOfOp : MOGG_Op<"get_shape_of"> {
  let summary = "Get the MOGG shape value from a KGEN tensor value";
  let description = [{
    Get the `!mogg.shape` value from the output of a mogg.to_kgen(`!mo.tensor`).
    This intrinsic has direct support from mojo that helps us handling in-place
    allocated value semantic dynamic tensors.

    Example:
    ```
    %shape = mogg.get_shape_of(%tensor) : !kgen.pointer<...> -> !mogg.shape<[D1, D2]>
    ```
  }];

  let arguments = (ins AnyType:$tensor);
  let results = (outs MOGG_Shape:$shape);
}


def MOGG_CallOp : MOGG_Op<"call"> {
  let summary = "Represents a KGEN function to call";
  let description = [{
    This represents a call to a KGEN function. The kernel target will be a
    function we retrieve from the registered ops. The call can also take an
    optional list of symbols in the from of `$params`. These are lambdas within
    this kernel which should be passed as parameters to the KGEN call.

    Example:
    ```mlir
    mogg.call "fake_op" [@test_lambda, @test_lambda2](%input, %output) : !mo.tensor<[2, 3], f32>, !mo.tensor<[2, 3], f32>
    ```

    The above will look up `fake_op` in the mojo op register to find the actual
    mojo symbol for that function and will compile down to a kgen call to that
    symbol with the lambda functions passed as parameters.

   ```mlir
    %0 = mogg.call "simd_load"(%input, %indices) : !mo.tensor<[2, 3], f32>, !mogg.indices -> !mogg.simd<f32>
   ```

    These calls can operate on tensors, simd values, and indices, and can return
    any of those types.
  }];

  let arguments = (ins
     StrAttr:$mojoKernelTarget,
     Variadic<AnyType>:$operands,
     OptionalAttr<ArrayAttr>:$params,
     OptionalAttr<DictionaryAttr>:$kgenParams
  );

   let results = (outs Variadic<AnyType>:$results);

   let assemblyFormat = [{
     $mojoKernelTarget ($params^)? `(`($operands^)?`)` ($kgenParams^)? attr-dict
       (`:` type($operands)^)? (`->` type($results)^)?
   }];

  let extraClassDeclaration = [{
    /// Changes the value of a given KGEN parameter if it exists. Returns true
    /// if the parameter exists and the value is now the same as value, false
    /// if the parameter doesn't exist.
    bool setKGENParamToValueIfPresent(StringRef name, Attribute value);
  }];
}

def MOGG_LambdaInvokeOp : MOGG_Op<"invoke",
   [MemoryEffectsOpInterface]> {
  let summary = "Represents an inlined call to a lambda";
  let description = [{
    Lambda invoke is a direct call of that lambda. This will lower to a
    kgen.call_param operation. It cannot take in any parameters. It can take
    in/return only on simd values or indices, not tensors directly. Tensors are
    just captured from above in the invoked lambda.

    Example:
    ```mlir
    %0 = mogg.invoke @test_lambda(%input, %output) : !mogg.indices, !mogg.simd<f32> -> !mogg.simd<f32>
    ```
  }];

  let arguments = (ins
     SymbolRefAttr:$lambda,
     Variadic<AnyTypeOf<[MOGG_Simd, MOGG_Indices]>>:$operands,
     OptionalAttr<BoolAttr>:$readOnly
  );
  let results = (outs Variadic<MOGG_Simd>:$outs);

  let assemblyFormat = [{
    $lambda `(`($operands^)?`)` attr-dict
      (`:` type($operands)^)? (`->` type($outs)^)?
  }];

  // TODO(10098): We will want to come up with a proper side effect analysis here.
  let extraClassDeclaration = [{
    void getEffects(
      llvm::SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects) {
      // If we don't have attr or it is false assume this has a side effect.
      bool isReadOnly = getReadOnly().value_or(false);
      if (!isReadOnly)
        effects.emplace_back(mlir::MemoryEffects::Write::get());
    }
  }];
 }

def MOGG_LambdaOp : MOGG_Op<"lambda",
  [SingleBlockImplicitTerminator<"OutputOp">,
    Symbol]> {
  let summary = "Represents non-isolated from above lambda";
  let description = [{
    These lambdas can capture tensors from above by reference and apply KGEN
    function calls to them. Lambdas can take and return indices and SIMD values
    only. They compose with other lambdas by `LambdaInvokeOp` and can be used by
    KGEN calls by the CallOp.

    Example:
    ```mlir
    mogg.lambda @test_lambda = (%arg4: !mogg.indices, %arg5 : !mogg.simd<f32>) -> !mogg.simd<f32> {
      ...
    }
    ```

    The above exposes a lambda with symbol name `test_lambda` which can then be
    used in calls to kgen functions or invoked directly.
  }];

  let arguments = (ins SymbolNameAttr:$sym_name, TypeAttrOf<Builtin_Function>:$func);
  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{
    custom<Lambda>($sym_name, $func, $body) attr-dict
  }];
}

def MOGG_StackAllocateTensorsOp : MOGG_Op<"stack_allocate_tensors"> {
  let results = (outs Variadic<MO_Tensor>:$results);

  let summary = "Allocates statically sized NDBuffers/Tensor on the host's stack.";
  let description = [{
    Each result of this op is an NDBuffer/Tensor:
     - Only statically shaped `!mo.tensor` types are allowed.
     - The shapes should be 'small' so as to not blow the stack.
     - The NDBuffer/Tensor will be backed by a stack allocated buffer/tensor
       on the host stack.
     - Obviously these NDBuffers/Tensor may only be used for host-side temporaries
       within the kernel and cannot be returned to the caller.

    Example:
    ```mlir
      %tensors:2 = mogg.stack_allocate_tensors : (!mo.tensor<[2, 3], f32>, !mo.tensor<[4], f32>)
    ```
  }];

  let assemblyFormat = [{
   attr-dict `:` type($results)
  }];
}

def MOGG_StackAllocateOpaquesOp : MOGG_Op<"stack_allocate_opaques"> {
  let results = (outs Variadic<AnyTypeOf<[MO_Opaque, MGP_OpaqueType]>>:$results);

  let summary = "Allocates opaque types on the host's stack.";
  let description = [{
    Each result of this op is an opaque object.
    It represents the outputs of a kernel call.
    This op does a stack allocation for the outputs of the kernel.
    The data is unitialized, and will be filled by the kernel.

    Example:
    ```mlir
      %vals:2 = mogg.stack_allocate_opaques : (!mgp.opaque, !mo.opaque<"my_fp">)
    ```
  }];

  let assemblyFormat = [{
   attr-dict `:` type($results)
  }];
}

def MOGG_CreateEmptyTensorsOp : MOGG_Op<"empty_tensors"> {
  let results = (outs Variadic<MO_Tensor>:$results);

  let summary = "This creates some tensors which have only shape information but no value";
  let description = [{
    These tensors will be created with the correct shape and type but will have
    no memory allocated. These are a temporary construct to allow shape functions
    to create null tensors without needing to deal with outputs etc.

    Example:
    ```mlir
    %alloc = mogg.create_empty_tensors : !mo.tensor<[2, 3], f32>
    ```
    This returns a tensor with an internal memory allocation.
  }];

  let assemblyFormat = [{
   attr-dict `:` type($results)
  }];
}

def MOGG_DimFromShapeOp : MOGG_Op<"dim_from_shape", [Pure]> {
  let summary = "Gets dims from a MOGG shape";
  let description = [{
    Given a MOGG shape value which represents a shape and we get the dimensions
    of it as multiple `index`.
  }];

  let arguments = (ins MOGG_Shape:$shape);
  let results = (outs Variadic<Index>:$dims);

  let assemblyFormat = [{
    attr-dict `(` $shape `:` type($shape) `)` `->` type(results)
  }];
}

def MOGG_BufferBorrowOp : MOGG_Op<"buffer.borrow", [Pure]> {
  let arguments = (ins AnyType:$buf, AnyType:$borrowee);
  let results = (outs AnyType:$result);
  let summary = "Intrinsic to create a buffer but extend lifetime of another";
  let description = [{
    Represents a placeholder operation which on lowering to will produce a
    new buffer which extends a shared lifetime with the other.
  }];

  let assemblyFormat = [{
    `(` operands `)` attr-dict `:`
        functional-type(operands, results)
  }];
}


def MOGG_ToTensorFromMGPOp : MOGG_Op<"to_tensor_from_mgp", [Pure]> {
  let arguments = (ins AnyType:$dataPointer, Variadic<Index>:$dims);
  let results = (outs MO_Tensor:$result);
  let summary = "Represents the construction of an NDBuffer or Tensor";
  let description = [{
    Takes a data pointer and a list of dimensions. Will lower this into a call
    to NDBuffer or Tensor constructor under the hood.
  }];

  let assemblyFormat = [{
    `(` operands `)` attr-dict `:`
        functional-type(operands, results)
  }];
}

def MOGG_ToMGPFromTensorOp : MOGG_Op<"to_mgp_from_tensor", [Pure]> {
  let arguments = (ins AnyType:$tensor);
  let results = (outs AnyType:$dataPointer);
  let summary = "Get the data pointer from the kgen tensor";
  let description = [{
    Get the data pointer from the kgen tensor. The input tensor is the output of
    a `to_kgen` taking a `mo.tensor`.
  }];
}

def MOGG_ToOpaqueFromMGPOp : MOGG_Op<"to_opaque_from_mgp", [Pure]> {
  let arguments = (ins AnyType:$dataPointer);
  let results = (outs AnyType:$result);
  let summary = "Represents the construction of an opaque type from a pointer";
  let description = [{
    Takes a pointer to the opaque object.
    `dataPointer` is an !mgp.opaque object and `result` is an !mo.opaque.
    During lowering, the pointer is passed to the kernel.
    At the implementation level, it's just a bitcast / rebind / no-op.
    But it helps preserving the correct types in the IR during lowering.
  }];

  let assemblyFormat = [{
    `(` operands `)` attr-dict `:`
        functional-type(operands, results)
  }];
}

def MOGG_ToListFromMGPOp : MOGG_Op<"to_list_from_mgp", [Pure]> {
  let arguments = (ins AnyType:$dataPointer);
  let results = (outs AnyType:$result);
  let summary = "Represents the construction of an list type from a pointer";
  let description = [{
    Takes a pointer to !mgp.list to !mo.list.

    `dataPointer` is an !mgp.list object and `result` is a !mo.list.

    At the implementation level, it's just a rebind since we define the ABI for
    both objects the same way (ABIList).
  }];

  let assemblyFormat = [{
    `(` operands `)` attr-dict `:`
        functional-type(operands, results)
  }];
}

def MOGG_ToMGPFromOpaqueOp : MOGG_Op<"to_mgp_from_opaque", [Pure]> {
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$dataPointer);
  let summary = "Get the opaque pointer from the kgen value";
  let description = [{
    `input` is a KGEN object, and `dataPointer` is an !mgp.opaque
    Get the opaque pointer from the kgen opaque tensor.
    At the implementation level, it's just a bitcast / rebind / no-op.
    But it helps preserving the correct types in the IR during lowering.
  }];

  let assemblyFormat = [{
    `(` operands `)` attr-dict `:`
        functional-type(operands, results)
  }];
}

def MOGG_GetAsyncValuesOp : MOGG_Op<"get_async_vals", [Pure]> {
  let results = (outs Variadic<AnyType>:$result);
  let summary = "Returns pointers for each of the async value inputs ";
  let description = [{
    Each input and output are represented by an async value. This returns a
    pointer to each of the inputs and outputs. Inputs appear first followed
    by the outputs.
  }];

  let assemblyFormat = [{
    `(` operands `)` attr-dict `:`
      functional-type(operands, results)
  }];
}

def MOGG_CreateAsyncMojoValueOp : MOGG_Op<"create_async_mojo_value"> {
  let arguments = (ins
      AnyType:$mojoPointer,
      AnyType:$asyncVal,
      AnyType:$runtime
  );
  let summary = "Create an AsyncValue from a mojo pointer";
  let description = [{
    Create an AnySyncValue<MojoValue> from a mojo pointer.
    This can't just be a mogg.call like for other asyncs, as the logic is a bit more complex.
  }];

  let assemblyFormat = [{
    `(` operands `)` attr-dict `:`
      functional-type(operands, results)
  }];
}

def MOGG_ReleaseMojoValueOp : MOGG_Op<"release_mojo_value"> {
  let arguments = (ins
      AnyType:$mojoValue
  );
  let summary = "Call the mojo destructor on a mojo object";
  let description = [{
    Call the destructor method on a KGEN mojo value.
    Mojo objects are usually stored in an AnySyncValue<MojoValue>.
    But when fusing regions, some leftover objects are never released.
    This op takes care of realising them.
  }];

  let assemblyFormat = [{
    `(` operands `)` attr-dict `:`
      functional-type(operands, results)
  }];
}

def MOGG_RuntimePlaceholderOp : MOGG_Op<"rt_placeholder", [Pure]> {
  let results = (outs AnyType:$result);
  let summary = "Returns a placeholder for the asyncrt runtime pointer";
  let description = [{
    This is resolved to an ABI argument during lowering.
  }];

  let assemblyFormat = [{
   attr-dict `:` type($result)
  }];
}

def MOGG_OutputPlaceholdersOp : MOGG_Op<"output_placeholders"> {
  let results = (outs Variadic<AnyTypeOf<[MO_Tensor, MO_Opaque, MO_Buffer, MO_Chain]>>:$results);

  let summary = "Represents buffer semantic SSA values for the outputs";
  let description = [{
    As the KGEN kernels require the output to be a buffer semantic input to the
    kernel we need to represent this within the kernel. This op produces a
    placeholder we can use for each output. During lowering these become inputs
    to the kernel as KGEN expects.

    Example:
    ```mlir
      %out = mogg.output_placeholders : !mo.tensor<[2, 3], f32>
    ```
    This provides a `%out` SSA value which corresponds to the first output which
    we can now pass into the KGEN calls which expect a buffer.
  }];

  let assemblyFormat = [{
   attr-dict `:` type($results)
  }];
}

def MOGG_CreateReadyChainOp : MOGG_Op<"create_ready_chain", [Pure]> {
  let results = (outs AnyType:$result);

  let summary = "Represents a chain output placeholder";
  let description = [{
    This is to help resolve any chain dependencies for data-dependent operations
    and help elide chains during fusion when lowering down to a compiled region.

    This is a special-case to the output placeholders because we remove this much lower
    in the kgen pipeline compared to the output_placeholders

    Example:
    ```mlir
      %out = mogg.create_ready_chain : !mo.tensor<[2, 3], f32>
    ```
    This provides a `%out` SSA value which corresponds to the chain output that kernels
    want to output
  }];
}

def MOGG_TensorCopyOp : MOGG_Op<"tensor_copy"> {
  let summary = "Represents the copy intrinsic whose counterpart is in kgen library";
  let description = [{
    Represents the copy intrinsic that will be directly lowered to a call of
    the copy function in mogg's kgen library. The `src` (a tensor view) is
    materialized to `dst`.

    Example:
    ```mlir
      %3 = "mogg.to_kgen"(%1) : (!mo.tensor<[2], si32>) -> !kgen.pointer<struct<...>>
      %4 = pop.stack_allocation 1 x struct<(...) memoryOnly>
      "mogg.tensor_copy"(%3, %4) : (!kgen.pointer<struct<...>>, !kgen.pointer<struct<...>>)
    ```
  }];

  let arguments = (ins AnyType:$dst, AnyType:$src);
}

//===----------------------------------------------------------------------===//
// MOGG Runtime compiled code representation ops.
//===----------------------------------------------------------------------===//

def MOGG_MojoCallContextPlaceholderOp: MOGG_Op<"runtime_mojocallcontext_placeholder", [Pure]> {
  let summary = "Launders the runtime MojoCallContext into KGEN land";
  let description = [{
    Returns the MojoCallContextPtr passed to the Mojo kernel by the C++ runtime.
  }];
  let results = (outs AnyType:$results);
}

def MOGG_ToKGENOp: MOGG_Op<"to_kgen", [Pure,
  DeclareOpInterfaceMethods<MOGG_MojoObjectInfosInterface>]> {
  let summary = "Converts a MOGG value to a KGEN value";
  let description = [{
    Converts a mogg/mo value to a kgen value. Includes the shape and optionally,
    the stride parameter that appears in the kernel.
  }];

  let arguments = (ins AnyTypeOf<[MOGG_Simd, MO_Tensor, MO_Buffer, MOGG_Indices, MOGG_Shape, MGP_OpaqueType, MGP_List, MO_List, MO_Opaque]>:$input,
    OptionalAttr<TypedAttrInterface>:$shape,
    OptionalAttr<TypedAttrInterface>:$strides,
    OptionalAttr<DictionaryAttr>:$typeMethodsDict,
    UnitAttr:$isRegisterType
  );
  let results = (outs AnyType:$results);

  let extraClassDeclaration = [{
    /// Necessary to implement MojoObjectInfosInterface.
    Value getKGENMojoValueImpl() { return getResults(); }
  }];
}

def MOGG_ToMOGGOp: MOGG_Op<"to_mogg", [Pure,
  DeclareOpInterfaceMethods<MOGG_MojoObjectInfosInterface>]> {
  let summary = "Converts a KGEN value to a MOGG value";
  let description = [{
    Converts a KGEN value to a MOGG value. This is the converse of to_kgen. It
    helps do partial lowering of the IR.
  }];

  let arguments = (ins AnyType:$input,
    OptionalAttr<TypedAttrInterface>:$shape,
    OptionalAttr<TypedAttrInterface>:$strides,
    OptionalAttr<DictionaryAttr>:$typeMethodsDict,
    UnitAttr:$isRegisterType
  );
  let results = (outs AnyTypeOf<[MOGG_Simd, MO_Tensor, MOGG_Indices, MOGG_Shape, MO_Opaque]>:$results);

  let extraClassDeclaration = [{
    /// Necessary to implement MojoObjectInfosInterface.
    Value getKGENMojoValueImpl() { return getInput(); }
  }];
}

def MOGG_AssertOp: MOGG_Op<"assert"> {
  let summary = "Placeholder for inserting a try catch block";
  let description = [{
    Placeholder operations that will eventually be lowered to a try catch block.
    This is necessary because mogg.kernel can't contain the try catch block
    until after JITing.
  }];

  let arguments = (ins AnyType:$condition, AnyType:$errorHandle);
}

def MOGG_LoadCompiledKernelOp : MOGG_LowerableOp<"load_compiled_kernel"> {
  let summary = "Loads pairs of (symbol, compiled object) into the runtime "
                "for a given device.";
  let description = [{
    For each of the input symbol/resource pair, associates the symbol with
    the resource (a blob of compiled kernel object) in the runtime (so the
    symbol names can be used by other ops like `ExecuteCompiledOp`).

    The compiled objects are with respect to the op's device attribute. The
    final representation for the compiled objects, and how they are registered
    with the runtime, depends on the device.
  }];

  let arguments = (ins ArrayAttr:$symbols,
    ConfinedAttr<
      ElementsAttr,
      [AnyAttrOf<[Builtin_DenseResourceElementsAttr]>]
    >:$value
  );
}


def MOGG_ExecuteCompiledOp : MOGG_LowerableOp<"execute_kernel", [
   DeclareOpInterfaceMethods<MO_ParamDeclaration>,
   MO_HasDeviceInterface]> {
  let summary = "Executes a compiled kernel on a given device.";
  let description = [{
    Executes a compiled kernel with the given name and operands on the given
    device.

    This will use the kernel name to look up the compiled object (loaded by
    `LoadCompiledKernelOp`), and call into the kernel with the lowered operands and
    other values required by the ABI.

    The `usedParams` attribute represent unique shape parameters used inside the
    compiled kernel; they will be lowered to runtime values and passed into the
    kernel, which will use them to reconstruct the tensor shapes.

    `outputParamDecls` allows this compiled operation to define a parameter
    values. These are exported into MGP land where they will be translated
    into calls into the parameter system.

    If `isAsync` is true, the kernel is implemented using the Mojo async/await
    scheme. Otherwise the kernel is synchronous.
  }];

  let arguments = (ins Variadic<AnyType>:$operands,
                       StrAttr:$kernelName,
                       ArrayAttr:$usedParams,
                       MO_ParamDecls:$outputParamDecls,
                       StrAttr:$summary,
                       BoolAttr:$isAsync,
                       BoolAttr:$hasCuda);
  let results = (outs Variadic<AnyType>:$results);
}

#endif // GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOGGOPS_TD
