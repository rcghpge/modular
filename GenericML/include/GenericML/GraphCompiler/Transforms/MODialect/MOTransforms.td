//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MODIALECT_MOTRANSFORMS_TD
#define GENERICML_GRAPHCOMPILER_MODIALECT_MOTRANSFORMS_TD

include "mlir/Pass/PassBase.td"

def AddFallbackShapeFunctions :
    Pass<"add-fallback-shape-funcs", "GraphOp"> {
  let summary = "Adds fallback shape functions to unmaterialized mo ops";
  let description = [{
    This ensures all MO operations will have parameterized outputs by checking
    if they have already been parameterized and if not materializing a fallback
    shape function which will give them those parameters.
  }];

  let options = [
    ListOption<"kernelLibraryPaths", "kernel-library-paths", "std::string",
      "Paths to mojo packages containing mojo shape functions to be loaded. ">,
    Option<"withDebuginfo", "with-debuginfo", "bool",
      /*default=*/"false",
      "Should kernels and stubs be built with debuginfo.">,
  ];

  let dependentDialects = [
    "mlir::LLVM::LLVMDialect",
    "mlir::index::IndexDialect",
    "M::DebugInfo::DebugInfoDialect",
    "M::HLCF::HLCFDialect",
    "M::KGEN::Custom::CustomDialect",
    "M::KGEN::KGENDialect",
    "M::KGEN::LIT::LITDialect",
    "M::KGEN::POP::POPDialect",
    "M::MDialect",
    "mlir::NVVM::NVVMDialect",
  ];
}

def SymbolicizeFallbackShapeFunctions :
    Pass<"symbolicize-fallback-shape-funcs", "mlir::ModuleOp"> {
  let summary = "Symbolically fold fallback shape functions in the graph";
  let description = [{
    This pass examines all fallback shape function calls in the `mo.graph` ops,
    and attempts to interpret the shape function body symbolically; if
    succeeded, replaces the shape function call with `kgen.param.declare`s for
    the shape function call's output parameters (and assertions in the shape
    function).
  }];

  let options = [
    ListOption<"kernelLibraryPaths", "kernel-library-paths", "std::string",
      "Paths to mojo packages containing mojo shape functions to be loaded. ">,
    Option<"withDebuginfo", "with-debuginfo", "bool",
      /*default=*/"false",
      "Should kernels and stubs be built with debuginfo.">,
  ];

  let dependentDialects = [
    "mlir::LLVM::LLVMDialect",
    "mlir::index::IndexDialect",
    "M::DebugInfo::DebugInfoDialect",
    "M::HLCF::HLCFDialect",
    "M::KGEN::Custom::CustomDialect",
    "M::KGEN::KGENDialect",
    "M::KGEN::LIT::LITDialect",
    "M::KGEN::POP::POPDialect",
    "M::MDialect",
    "M::MOGG::MOGGDialect",
    "mlir::NVVM::NVVMDialect",
  ];
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

  // NOTE: this is only needed as we transition `?` to live in RMO.
  // Once all frontends only use RMO, this is superfluous as this will only
  // affect RMO operations.
  let options = [
    Option<"resolveRMOOnly", "resolveRMOOnly", "bool", "false",
           "Whether to only resolve unknown parameters for RMO operations.">,
  ];
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

  let options = [
    Option<"symbolicFfastMath", "symbolic-ffast-math", "bool", "true",
      "If the symbolic optimizer is allowed to (1) operate under the finite"
      "math assumption, (2) re-order floating point operations (3) and apply"
      "reciprocal optimizations">,
      Option<"symbolicNoUndefinedMath", "symbolic-no-undefined-math", "bool", "true",
      "If the symbolic optimizer is allowed to assume that there will be no"
      "undefined math operations such as divisions by 0">
  ];
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

def GenVizSpec : Pass<"gen-viz", "GraphOp"> {
  let summary = "Generates a Modular Visualization Spec.";
  let description = [{
    This pass generates a visualization spec which can be used as input to
    Netron (https://github.com/modularml/netron) to visualize the MO graph.
  }];
  let options = [
    Option<"vizOutputPath", "viz-output-path", "std::string", "",
           "Specifies the output path for the "
           "visualization spec.">,
  ];
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
    "M::KGEN::Custom::CustomDialect",
    "M::KGEN::KGENDialect",
    "M::KGEN::LIT::LITDialect",
    "M::KGEN::POP::POPDialect",
    "M::HLCF::HLCFDialect",
    "M::DebugInfo::DebugInfoDialect",
    "mlir::LLVM::LLVMDialect",
    "mlir::NVVM::NVVMDialect",
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

def SupportConversionsTest : Pass<"support-conversions-test", "ModuleOp"> {
  let summary = "Testing only pass for testing MO Conversion support library.";
  let description = [{
    This is a testing only pass. It looks for hard-coded, unregistered ops and
    expands them using helper libraries located in
    GenericML/GraphCompiler/Conversion/MO/Support/*.
    ```
    mo.graph @const_program() -> (!mo.tensor<[2], f32>) {
      ...
      // Pattern rewrite looks for "prelu" and expands op with decomposed
      // operators in MO to execute a "prelu" operation:
      %2 = "prelu"(%0, %1) : (!mo.tensor<[2], f32>, !mo.tensor<[1], f32>) -> !mo.tensor<[2], f32>
      mo.output %2 : !mo.tensor<[5], f32>
    }
    ```
    See SupportConversionsTest.cpp for more registered rewrite patterns.
  }];

  let dependentDialects = [
    "M::RMO::RMODialect"
  ];
}

def AddDevicesPass : Pass<"add-devices", "GraphOp"> {
  let summary = "Adds `device_specs` and `device` attributes to the `mo.graph`.";
  let description = [{
    This pass ensures the `mo.graph` has a `device_specs` attribute describing
    the intended runtime target. By default that will be identical to the
    current host machine, including the CPU 'host' device specification and
    any additional non-CPU devices (if `enable-extra-devices` is on).

    If `default-device-label` is non-empty, also add a `device` attribute to
    the `mo.graph` if it does not already have one. This can be used to
    establish the graph-wide default device on which ops should be executed.

    TODO(#21801): TEMPORARY until we have proper compilation configuration handling.
  }];

  let dependentDialects = ["M::MDialect"];

  let options = [
    Option<"enableExtraDevices", "enable-extra-devices", "bool", "false",
           "Include all host devices instead of just the CPU during compilation.">,
    Option<"enableFakeCudaDevice", "enable-fake-cuda-device", "bool", "false",
           "Include a simulated CUDA device for testing on hosts without an "
           "actual GPU.">,
    Option<"defaultDeviceLabel", "default-device-label", "std::string", "\"\"",
           "If non-empty, assign as many ops as possible to the device "
           "with the given label (and device id zero).">,
  ];
}

def AssignDevicesPass : Pass<"assign-devices", "GraphOp"> {
  let summary = "Assigns an `#M.device_spec` annotation to every operation "
                "and tensor.";
  let description = [{
    This pass assigns a `#M.device_spec` annotation to every operation and
    tensor in the graph, respecting any existing annotations. New `mo.transfer`
    ops are inserted where required for consistency. The pass uses a simple
    eager assignment algorithm and does not attempt any form of global
    optimization.

    The rules are:
     - Ops are processed in reverse dataflow order, ie from outputs to inputs.
     - Unless already annotated, graph input and output tensors are annotated
       to be stored on the host device.
     - Unless already annotated, an operation is annotated to execute on the
       device found by the following search:
         - the first result or operand tensor with a device annotation;
         - the annotation on the overall graph op;
         - the 'host' device
     - Unless already annotated, the operand and result tensors of an operation
       are annotated to match the operation.
     - Conflicts are resolved by inserting a mo.transfer op.
     - The results of a `mo.transfer` op may be shared, and `mo.transfer` has
       the expected transitivity and reflexivity rules.

    Note that this pass has no understanding of the devices themselves, and
    simply works on uninterpreted `#M.device_ref`s.
  }];

  let options = [
    Option<"smallTensorLimit", "small-tensor-limit", "int64_t", "1024",
           "Maximum number of elements a tensor may have to be considered "
           "'small'. Ops which take and return only 'small' tensors are "
           "themselves considered 'small', and need not be placed on the "
           "overall graph's default device.">,
    ListOption<"kernelLibraryPaths", "kernel-library-paths", "std::string",
          "Paths to mojo packages containing mojo shape functions to be loaded. ">,
    Option<"withDebuginfo", "with-debuginfo", "bool",
      /*default=*/"false",
      "Should kernels and stubs be built with debuginfo.">,
    Option<"targetDevice", "target-device",  "std::string", "\"\"",
           "If non-empty, assign devices using this device">,
    Option<"copyFromHost", "copy-from-host", "bool", "true",
           "If true make inputs / outputs host tensors with copies inserted.">,
  ];

  // We need to include all the dialects as we access the mojo library.
  let dependentDialects = [
    "mlir::LLVM::LLVMDialect",
    "mlir::index::IndexDialect",
    "M::DebugInfo::DebugInfoDialect",
    "M::HLCF::HLCFDialect",
    "M::KGEN::Custom::CustomDialect",
    "M::KGEN::KGENDialect",
    "M::KGEN::LIT::LITDialect",
    "M::KGEN::POP::POPDialect",
    "M::MDialect",
    "mlir::NVVM::NVVMDialect",
  ];
}

def DebugPrintAllTensorsPass : Pass<"debug-print-all-tensors", "GraphOp"> {
  let summary = "Print all intermediate tensors as they are produced.";
  let description = [{
    This pass emits a mo.debug.tensor.print for every mo.tensor typed value
    in the graph, including the inputs. The graph is topologically sorted to
    ensure the model won't end up with a cycle due to the chains which must
    serialize the underlying prints. For debugging only.
  }];
}

def SplatLargeConstants :
    Pass<"splat-large-constants", "GraphOp"> {
  let summary = "Will splat a large constant for development purposes";
  let description = [{
    This pass will replace a large constant with a splatted version by
    broadcasting a single scalar value. This pass does not preserve correctness
    so it should be used for development purposes only (especially for large
    models).
  }];

  let options = [
    // Default is 100M elements.
    Option<"numElements", "num-elements", "int", "100000000",
      "Threshold number of elements in the constant's dense array">,
  ];
}

def TopologicalSort :
    Pass<"mo-topological-sort", "GraphOp"> {
  let summary = "Topologically sort ops in a graph based on SSA and parameter dependencies.";
  let description = [{
    This pass topologically sort ops within a `mo.graph` based on both SSA and
    parameter dependencies. It also handles nested regions properly (e.g.,
    `mo.if`).
  }];
}

def ElideRedundantMutableOps :
    Pass<"elide-redundant-mutable-ops", "GraphOp"> {
  let summary = "Eliminates redundant mo.mutable.store and mo.mutable.load ops.";
  let description = [{
    This pass contains patterns meant to eliminate mo.mutable.store and
    mo.mutable.load ops under appropriate conditions they are:

    For mo.mutable.load ops:
      Erase the load when it is immediately preceded by a store operating on
      the same buffer and the load is the sole consumer of the store's output
      chain.

      e.g.

      ```
      %ch1 = mo.mutable.store(%ch0, %buffer, %tensor0)
      %ch2, %tensor1 = mo.mutable.load(%ch1, %buffer)

      %res = mo.op(%tensor1, ...)

      %ch3 = mo.mutable.store(%ch2, %buffer, %res)

      ==>

      %ch1 = mo.mutable.store(%ch0, %buffer, %tensor0)

      %res = mo.op(%tensor0, ...)

      %ch3 = mo.mutable.store(%ch1, %buffer, %res)
      ```

    For mo.mutable.store ops:
      Erase a store when it is followed by another store operating on the same
      buffer and the first store's output chain is solely consumed by the second
      store.


      e.g.

      ```
      %ch1 = mo.mutable.store(%ch0, %buffer, %tensor)
      %res = mo.op(%tensor, ...)
      %ch3 = mo.mutable.store(%ch1, %buffer, %res)

      ==>

      %res = mo.op(%tensor, ...)
      %ch3 = mo.mutable.store(%ch0, %buffer, %res)
      ```

    This enables the fusion passes in MOGG to work with in-place operations.
  }];
}

#endif // GENERICML_GRAPHCOMPILER_MODIALECT_MOTRANSFORMS_TD
