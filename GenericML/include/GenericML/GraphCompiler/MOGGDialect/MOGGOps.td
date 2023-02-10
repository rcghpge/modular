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
include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/BuiltinAttributes.td"

// Base class for ops in this dialect.
class MOGG_Op<string mnemonic, list<Trait> traits = []>
    : Op<MOGGDialect, mnemonic, traits>;

class MOGG_LowerableOp<string mnemonic, list<Trait> traits = []>
    : MOGG_Op<mnemonic, traits # [DeclareOpInterfaceMethods<LowerableToMGPInterface, ["lower"]>]>;

//===----------------------------------------------------------------------===//
// MOGG subgraph ops
//===----------------------------------------------------------------------===//

def MOGG_ExtOp : MOGG_LowerableOp<"ext.op", [HasParent<"MO::GraphOp">]> {
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

def MOGG_ElementwiseOp : MOGG_Op<"elementwise", [HasParent<"MO::GraphOp">, SingleBlock]> {
  let summary = "Block of fused elementwise operations";
  let description = [{
    This represents an operation we want to create a fused elementwise JIT kernel for.
    Additional MOGG passes will fuse the regions of these operations together to create
    fused elementwise blocks.
  }];

  let arguments = (ins Variadic<MO_Tensor>:$operands);
  let results = (outs Variadic<MO_Tensor>:$results);
  let regions = (region SizedRegion<1>:$body);
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
// MOGG jit ops
//===----------------------------------------------------------------------===//

def MOGG_LitOp : MOGG_Op<"lit", [SingleBlock]> {
  let summary = "Represents a target to call";
  let description = [{
    This represents an operation which is implemented by the kernel name.
    It will be jitted by the jit compiler pass into a compiled object.
  }];

   let arguments = (ins
     StrAttr:$litKernelTarget,
     StrAttr:$originalOpName,
     Variadic<MO_Tensor>:$operands
   );

   let results = (outs Variadic<MO_Tensor>:$results);

   let assemblyFormat = [{
     $litKernelTarget attr-dict `(`($operands^)?`)`
       (`:` type($operands)^)? (`->` type($results)^)?
   }];
 }


def MOGG_LoadJitKernelOp : MOGG_LowerableOp<"load_compiled_kernel"> {
  let summary = "Loads a compiled object and all symbols into the JIT runtime.";
  let description = [{
  }];

  let arguments = (ins ArrayAttr:$symbols,
    ConfinedAttr<
      ElementsAttr,
      [AnyAttrOf<[Builtin_DenseResourceElementsAttr]>]
    >:$value);
}


def MOGG_ExecuteCompiledOp : MOGG_LowerableOp<"execute_kernel", [HasParent<"MO::GraphOp">]> {
  let summary = "A compiled KGEN ELF file attached to an op";
  let description = [{}];

  let arguments = (ins Variadic<MO_Tensor>:$operands, StrAttr:$kernelName, StrAttr:$originalOpName);
  let results = (outs Variadic<MO_Tensor>:$results);
}

#endif // GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOGGOPS_TD
