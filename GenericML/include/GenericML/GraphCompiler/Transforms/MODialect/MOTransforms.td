//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MODIALECT_MOTRANSFORMS_TD
#define GENERICML_GRAPHCOMPILER_MODIALECT_MOTRANSFORMS_TD

include "mlir/Pass/PassBase.td"

def ShapeInferenceMaterialization :
    Pass<"materialize-shape-funcs", "ModuleOp"> {
  let summary = "Materializes the shape inference functions.";
  let description = [{
    This pass materializes shape inference functions for parameter-declaring ops
    that have them, and add invocations to the graph, so that the parameters
    will be declared and bound to the results of shape inference function
    invocations, instead of declared by the ops.

    The ops need to be fully parametric (i.e. unknown shapes or dimensions in
    the arguments and results are not allowed), and the shape inference
    functions are not evaluated.
  }];

  let dependentDialects = [
    "mlir::LLVM::LLVMDialect",
    "mlir::index::IndexDialect",
    "M::Cache::CacheDialect",
    "M::DebugInfo::DebugInfoDialect",
    "M::HLCF::HLCFDialect",
    "M::KGEN::KGENDialect",
    "M::KGEN::LIT::LITDialect",
    "M::KGEN::POP::POPDialect",
    "M::MDialect",
  ];
}

def AddFallbackShapeFunctions :
    Pass<"add-fallback-shape-funcs", "GraphOp"> {
  let summary = "Adds fallback shape functions to unmaterialized mo ops";
  let description = [{
    This ensures all MO operations will have parameterized outputs by checking
    if they have already been parameterized and if not materializing a fallback
    shape function which will give them those parameters. 
  }];
}

def InlineShapeFunctions : Pass<"inline-shape-funcs", "ModuleOp"> {
  let summary = "Inlines shape functions into the graph.";
  let description = [{
    This pass inlines (already materialized) shape functions invocations into
    the graph as parameter computations.
  }];
}

def PropagateShapes : Pass<"propagate-shapes", "GraphOp"> {
  let summary = "Propagates shape parameters through the graph.";
  let description = [{
    This pass evaluates shape parameter logic in the graph, simplifying and
    reducing shape logic as much as it can.
  }];
}

def ResolveUnknownParameters : Pass<"resolve-unknown-parameters", "ModuleOp"> {
  let summary = "Resolves unknown tensor shape and dimension parameters.";
  let description = [{
    This pass parameterizes `mo` ops that can be parameterized. This involves 2
    things:
    1. replacing unknown shape and dimension parameters (i.e. `?`) in `!mo.tensor`
    instances with parameter references.
    2. letting newly parameterized ops declare their new parameters that appear
    in their results.
  }];
}

def StaticizeShapes : Pass<"staticize-shapes", "ModuleOp"> {
  let summary = "Lower shape-like tensors to `!mosh.ape`.";
  let description = [{
    This pass attempts to lower shape-like MO tensors (i.e. rank-1, signed 32 or
    64 bit integer) to the special `!mosh.ape` type, and lower computations on
    them to parameter computations or static operations (as opposed to
    operations that take in dynamic values, like reshape or broadcast). This
    essentially "lifts" the shape logic encoded as tensor computations into the
    symbolic shape inference realm.

    The pass currently only resolves shapes that originate from `mo.shape_of`
    ops or constant shape-like tensors. In particular shape-like (or dim-like)
    arguments provided to the graph are not supported for now.
  }];

  // TODO(#16974) remove once shape materialization is removed from this pass
  let dependentDialects = [
    "mlir::LLVM::LLVMDialect",
    "mlir::index::IndexDialect",
    "M::Cache::CacheDialect",
    "M::DebugInfo::DebugInfoDialect",
    "M::HLCF::HLCFDialect",
    "M::KGEN::KGENDialect",
    "M::KGEN::LIT::LITDialect",
    "M::KGEN::POP::POPDialect",
    "M::MDialect",
  ];
}

def SymbolicOptimization : Pass<"symbolic-optimize", "GraphOp"> {
  let summary = "Optimize the graph using symbolic reasoning.";
  let description = [{
    This pass leverages symbolic information present in the graph (usually as a
    result of shape inference) to optimize computations. It includes
    transformations such as:
    - moving certain operations within the graph
    - rewriting a series of operations into more efficient ones
    - canonicalizing certain operations into simpler form to enable
      downstream optimizations
  }];
}

def InferLayouts : Pass<"infer-layouts", "GraphOp"> {
  let summary = "Infer layouts of tensor values";
  let description = [{
    This pass tries to infer the layout of tensor values in a `mo.graph`, based
    on constraints of relevant MO ops, e.g., before layout transformations,
    `mo.conv` assumes input has `NHWC` layout. At the end of the inference, the
    pass will annotate the tensor types with inferred layout attributes.
  }];
}

def HoistParameterExpressions : Pass<"hoist-param-exprs", "GraphOp"> {
  let summary = "Move parameter expressions out of ops and into a preceding alias.";
  let description = [{
    Transforms
    ```
    %1 = mo.op(%arg: !mo.tensor<[add(N,M)], f32>) : !mo.tensor<[add(N,M)], f32>
    ```
    into
    ```
    kgen.param.declare param_expr = <add(N,M)>
    %arg_rebound = mo.rebind(%arg:!mo.tensor<[add(N,M)], f32>) :!mo.tensor<[param_expr], f32>
    %tmp = mo.op(%arg_rebound: !mo.tensor<[param_expr], f32>) : !mo.tensor<[param_expr], f32>
    %1 = %mo.rebind(tmp: !mo.tensor<(N,M), f32>): !mo.tensor<[add(N,M)], f32>
    ```
    `rebind`s are inserted around the block to ensure that the input and outputs of the block remain the same type.

    This pass is nessesary for MOGG to jit kernels with parameter expressions in thier arguments.
  }];
}

def ConstantFold : Pass<"constant-fold", "ModuleOp"> {
  let summary = "Partially evaluate constant subgraphs";
  let description = [{
    This pass identifies constant subgraphs and executes them at compile time
    using existing kernel infrastructure. Does not explicitly handle control flow.

    Example:
    ```
    mo.graph @const_program() -> !mo.tensor<[2], si32> {
      %0 = mo.constant {value = #M.dense_array<0, 1> : tensor<2xsi32>} : !mo.tensor<[2], si32>
      %1 = mo.constant {value = #M.dense_array<6, 3> : tensor<2xsi32>} : !mo.tensor<[2], si32>
      %2 = mo.mul(%0, %1) : !mo.tensor<[2], si32>
      %3 = mo.sub(%2, %1) : !mo.tensor<[2], si32>
      mo.output %3 : !mo.tensor<[2], si32>
    }
    ```
    becomes
    ```
    mo.graph @const_program() -> !mo.tensor<[2], si32> {
      %0 = mo.constant {value = #M.dense_array<0, 2> : tensor<2xsi32>} : !mo.tensor<[2], si32>
      mo.output %0 : !mo.tensor<[2], si32>
    }
    ```

    This pass may be non-determininstic in ordering of output nodes.

    This pass run on ModuleOps instead of GraphOps because it 1. needs to call
    runPipeline(MOToMOGG) which cannot be nested in a GraphOp, and 2. it needs
    to avoid overhead of doing lowering and model setup multiple times.
  }];
  // All these dialects are required to run MOToMOGG from within this pass.
  let dependentDialects = [
    "M::MDialect",
    "M::MOGG::MOGGDialect",
    "M::MGP::MGPDialect",
    "M::GRT::GRTDialect",
    "M::KGEN::KGENDialect",
    "M::KGEN::LIT::LITDialect",
    "M::KGEN::POP::POPDialect",
    "M::HLCF::HLCFDialect",
    "M::Cache::CacheDialect",
    "M::DebugInfo::DebugInfoDialect",
    "mlir::LLVM::LLVMDialect",
    "mlir::index::IndexDialect",
  ];
  let options = [
    Option<"iterations", "iterations", "int", "2", "Number of times to run constant folding back to back">,
  ];
}

def ConstantFoldSubgraphTest : Pass<"constant-fold-subgraph-test", "ModuleOp"> {
  let summary = "Test constant subgraph detection";
  let description = [{
    This is a testing only pass. It prints out the constant subgraphs found
    within a graph so that they can be checked by filecheck. Printed output
    looks like this:
    ```
    mo.graph @const_program() -> !mo.tensor<[2], si32> {
      %0 = mo.constant {value = #M.dense_array<0, 1> : tensor<2xsi32>} : !mo.tensor<[2], si32>
      %1 = mo.constant {value = #M.dense_array<6, 3> : tensor<2xsi32>} : !mo.tensor<[2], si32>
      %2 = mo.mul(%0, %1) : !mo.tensor<[2], si32>
      %3 = mo.sub(%2, %1) : !mo.tensor<[2], si32>
      mo.output %3 : !mo.tensor<[2], si32>
    }
    subgraph %0 %1 %2 %3
    ```
    which is the full graph followed by zero or more occurrences of word
    'subgraph' with the values that compose a constant subgraph.
  }];
}

#endif // GENERICML_GRAPHCOMPILER_MODIALECT_MOTRANSFORMS_TD
