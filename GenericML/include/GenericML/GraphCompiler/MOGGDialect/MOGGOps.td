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
include "GenericML/GraphCompiler/MODialect/MOTypes.td"
include "GenericML/GraphCompiler/Interfaces/LowerableToMGPInterface.td"
include "Support/MDialect/MAttrs.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/BuiltinTypes.td"

// Base class for ops in this dialect.
class MOGG_Op<string mnemonic, list<Trait> traits = []>
    : Op<MOGGDialect, mnemonic, traits>;

class MOGG_LowerableOp<string mnemonic, list<Trait> traits = []>
    : MOGG_Op<mnemonic, traits # [DeclareOpInterfaceMethods<LowerableToMGPInterface, ["lower"]>]>;

//===----------------------------------------------------------------------===//
// MOGG subgraph ops
//===----------------------------------------------------------------------===//

def MOGG_ExtOp : MOGG_LowerableOp<"ext.op", [Pure]> {
  let summary = "MOGG representation of an external library kernel";
  let description = [{
    This is a generic subgraph to represent calls to kernel libraries.

    It contains the name of the kernel, variadic inputs and outputs, and a body
    region for holding the externally called library for lowering.
  }];

  let arguments = (ins
    StrAttr:$opName,
    Variadic<MO_Tensor>:$operands
  );

  let results = (outs Variadic<MO_Tensor>:$results);

  let assemblyFormat = [{
    $opName attr-dict `(`($operands^)?`)`
      (`:` type($operands)^)? (`->` type($results)^)?
  }];
}


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
    zero, where the size is given by the corresponding element in `splitSizes`.

    Example:

    ```mlir
    %input: !mo.tensor<[9, 8], f32>
    %split_sizes = mo.constant {
      value = #M.dense_array<2, 3, 4> : tensor<3xsi64>} : !mo.tensor<[], si64>
    %res = mogg.split(%input, %split_sizes) : (
      !mo.tensor<[9, 8], f32>, !mo.tensor<[3], si64>
    ) -> (!mo.tensor<[2, 3], f32>, !mo.tensor<[3, 3], f32>, !mo.tensor<[4, 3], f32>)
    ```
  }];

  let arguments = (ins
    MO_Tensor:$input,
    MO_ShapeLikeIfRanked:$splitSizes);
  let results = (outs Variadic<MO_Tensor>:$results);

  let assemblyFormat = [{
    `(` $input `,` $splitSizes `)` attr-dict `:` `(`
      qualified(type($input)) `,` qualified(type($splitSizes))
    `)` `->` `(` type($results) `)`
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

def MOGG_MojoRegisterOp : MOGG_Op<"mojo_register", [HasParent<"ModuleOp">]> {
  let summary = "A mapping of all the registered mojo functions";
  let description = [{
    A module wide library of avaliable mojo functions. This is populated by
    loading the target Mojo library/libraries. The `dict` attribute maps names
    of MO ops or special instrinsic ops to the symbol name of the corresponding
    mojo function.

    Note that this is meant to be inserted and used by internal transformations.
    In the future this will be populated by a direct Mojo binding.
  }];

  let arguments = (ins DictionaryAttr:$dict);

  let assemblyFormat = [{
    attr-dict
  }];

}

def MOGG_KernelOp : MOGG_Op<"kernel", [IsolatedFromAbove,
                                        SingleBlock,
                                        SymbolTable, Pure]> {
  let summary = "A representation of a kernel which can be fused into and compiled";
  let description = [{
    This kernel represents the handover from the MO graph to the KGEN land.

    The operation itself is a fully legal graph semantic operation which
    composes with all other operations at the MO level.

    Inside the kernel however the semantics are mirrors of the KGEN semantics.
    We operate on lambdas and kgen function calls. This lets us have both inline
    in the same graph while keeping the hard dependencies contained to lowering.

    The kernel abstraction also exposes a number of hooks. These hooks are
    details of the fusion algorithms. They allow this kenrel to express the
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

    The attributes at the end.
    ```mlir
    { fusionBodyLambda = "elemwise_compute_lambda",
      inputFusionInterface = ["input_0_fn", "input_1_fn"],
      outputFusionInterface = ["output_0_fn"]
    }
    ```
    `fusionBodyLambda` Represents the fusable elementwise / view operation body
    contained with in. This signfies that this kernel is fully implemented by
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
  }];

  let arguments = (ins Variadic<MO_Tensor>:$operands,
    OptionalAttr<ArrayAttr>:$inputFusionInterface,
    OptionalAttr<ArrayAttr>:$outputFusionInterface,
    OptionalAttr<StrAttr>:$fusionBodyLambda,
    OptionalAttr<BoolAttr>:$isViewOp
  );
  let results = (outs Variadic<MO_Tensor>:$results);
  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{
    custom<Kernel>($operands, type($operands), type($results), $body) attr-dict
  }];
}

def MOGG_CallOp : MOGG_Op<"call"> {
  let summary = "Represents a KGEN function to call";
  let description = [{
    This represents a call to a KGEN function. The kernel target will be a
    function we retrieve from the MojoRegisterOp. The call can also take an
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
     Variadic<AnyTypeOf<[MOGG_Simd, MO_Tensor, MOGG_Indices]>>:$operands,
     OptionalAttr<ArrayAttr>:$params
  );

   let results = (outs Variadic<AnyTypeOf<[MOGG_Simd, MO_Tensor]>>:$results);

   let assemblyFormat = [{
     $mojoKernelTarget ($params^)? attr-dict `(`($operands^)?`)`
       (`:` type($operands)^)? (`->` type($results)^)?
   }];
 }

def MOGG_SimdLoadOp : MOGG_Op<"simd_load"> {
  let summary = "Represents a simd load";
  let description = [{
    Represents the special case of a simd_load. Simd load is a special case
    because we can lower it to different functions depending on the tensor
    details. If the tensor is a scalar then it becomes a splat, broadcasted last
    dim might need a splat, etc.

    Each load is in a performance critical function so we have to do this ahead
    of time.

    Example:
    ```mlir
    mogg.simd_load(%indices, %tensor) : !mogg.indices, !mo.tensor<[2, 3], f32>, !mo.tensor<[2, 3], f32>
    ```
  }];

  let arguments = (ins
    MO_Tensor:$tensor,
    MOGG_Indices:$indices
  );

  let results = (outs MOGG_Simd:$result);

  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
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
    [HasParent<"MOGG::KernelOp">, SingleBlockImplicitTerminator<"OutputOp">,
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

def MOGG_OutputPlaceholdersOp : MOGG_Op<"output_placeholders"> {
  let results = (outs Variadic<MO_Tensor>:$results);

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

//===----------------------------------------------------------------------===//
// MOGG Runtime compiled code representation ops.
//===----------------------------------------------------------------------===//

def MOGG_LoadJitKernelOp : MOGG_LowerableOp<"load_compiled_kernel"> {
  let summary = "Loads pairs of (symbol, compiled object) into the JIT runtime";
  let description = [{
    For each of the input symbol/resource pair, and associates the symbol with
    the resource (a blob of compiled kernel object) in the JIT runtime (so the
    symbol names can be used by other ops like `ExecuteCompiledOp`).
  }];

  let arguments = (ins ArrayAttr:$symbols,
    ConfinedAttr<
      ElementsAttr,
      [AnyAttrOf<[Builtin_DenseResourceElementsAttr]>]
    >:$value);
}


def MOGG_ExecuteCompiledOp : MOGG_LowerableOp<"execute_kernel"> {
  let summary = "Executes a compiled kernel";
  let description = [{
    Executes a compiled kernel with the given name and operands.

    This will use the kernel name to look up the compiled object (loaded by
    `LoadJitKernelOp`), and call into the kernel with the lowered operands and
    other values required by the ABI.

    The `usedParams` attribute represent unique shape parameters used inside the
    compiled kernel; they will be lowered to runtime values and passed into the
    kernel, which will use them to reconstruct the tensor shapes.
  }];

  let arguments = (ins Variadic<MO_Tensor>:$operands,
                          StrAttr:$kernelName,
                          ArrayAttr:$usedParams);
  let results = (outs Variadic<MO_Tensor>:$results);
}

#endif // GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOGGOPS_TD
