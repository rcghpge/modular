//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This defines the device-graph operations in the MGP dialect: the
// `mgp.device_graph.create`/`mgp.device_graph.execute` surface, its
// `mgp.device_graph.yield` build-body terminator, and the transient
// `mgp.device_graph.staging`/`mgp.device_graph.dependencies` scaffolds (with
// their yield terminators) used during device-graph synthesis.
//
// This file is included by MGPOps.td; the ops below use the local
// `MGP_DeviceGraphOp` base class, so they do not depend on `MGP_Op`.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_MGPDIALECT_MGPDEVICEGRAPHOPS_TD
#define GENERICML_MGPDIALECT_MGPDEVICEGRAPHOPS_TD

include "GraphCompiler/GraphCompiler/MGPDialect/Dialect/MGPDialect.td"
include "GraphCompiler/GraphCompiler/MGPDialect/Dialect/MGPTypes.td"
include "MLRT/GraphRT/MEFDialect/MEFTypes.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Base class for device-graph ops in the MGP dialect. Prepends `device_graph.`
// to the mnemonic, so `MGP_DeviceGraphOp<"create">` yields the op name
// `mgp.device_graph.create`.
class MGP_DeviceGraphOp<string mnemonic, list<Trait> traits = []> :
  Op<MGPDialect, "device_graph." # mnemonic, traits>;

//===----------------------------------------------------------------------===//
// DeviceGraphCreateOp
//===----------------------------------------------------------------------===//

def DeviceGraphCreateOp : MGP_DeviceGraphOp<"create", [
  IsolatedFromAbove,
  SingleBlockImplicitTerminator<"DeviceGraphYieldOp">,
]> {
  let summary = "Builds a device graph by recording into a builder.";
  let description = [{
    Returns an `!mgp.device_graph` built by invoking the build body. The op is
    stateless: it does no caching, takes no chain, and has no chain result -- the
    produced graph value carries the downstream dependency
    (`mgp.device_graph.execute` consumes it). (Caching is future work.)

    The `inputs` operands are the read-only buffers/tensors the graph treats as
    frozen inputs at replay time; they form the input half of the result
    `!mgp.device_graph` signature. The graph's outputs are produced inside the
    graph (value-returning, not destination-passing), so they have no operand --
    the build body yields them via `mgp.device_graph.yield`, and their types form
    the output half of the signature. The `!mgp.context` operand is build-time
    only (it feeds in-body `mgp.buffer.get_cached` reads) and is deliberately not
    part of the signature.

    The op owns the device-graph builder lifecycle. It creates a
    `!mgp.device_graph.builder` from its `deviceContext` operand and passes that
    builder to the build body as the body's first block argument; after the body
    returns, the op instantiates the recorded builder into its result
    `!mgp.device_graph`. The build body records nodes into that builder and
    yields the graph's output values; it does not create the builder or produce
    the graph object itself.

    The body's block arguments are, in order: the `!mgp.device_graph.builder`,
    the `!mgp.context`, then the `inputs` (carrying the input operand types
    verbatim -- bare `T`, no `!mogg.async` wrapping). The body must terminate with
    a `mgp.device_graph.yield` whose operand types match the result graph
    signature's results.

    Example:
    ```mlir
      %graph = mgp.device_graph.create
          %ctx, %gpu_ctx
          inputs(%in, %bias, %wt
              : !mgp.buffer<<"gpu", 0>>, !mgp.buffer<<"gpu", 0>>,
                !mgp.buffer<<"gpu", 0>>) ({
        ^bb0(%builder: !mgp.device_graph.builder,
             %ctx_b: !mgp.context,
             %in_b: !mgp.buffer<<"gpu", 0>>,
             %bias_b: !mgp.buffer<<"gpu", 0>>,
             %wt_b: !mgp.buffer<<"gpu", 0>>):
          // ... record nodes into %builder, produce %out ...
          mgp.device_graph.yield %out : !mgp.buffer<<"gpu", 0>>
      }) : !mgp.device_context<<"gpu", 0>> -> !mgp.device_graph<...>
    ```
  }];

  let arguments = (ins
    MGP_ContextType:$theContext,
    MGP_DeviceContextType:$deviceContext,
    Variadic<AnyType>:$inputs
  );
  let results = (outs MGP_DeviceGraphType:$outGraph);
  let regions = (region SizedRegion<1>:$buildBody);

  let assemblyFormat = [{
    $theContext `,` $deviceContext
    `inputs` `(` ( $inputs^ `:` qualified(type($inputs)) )? `)`
    $buildBody attr-dict
    `:` qualified(type($deviceContext)) `->` qualified(type($outGraph))
  }];

  let builders = [
    // Builds the op and its build body in one shot. Creates the body block with
    // its `!mgp.device_graph.builder` block argument, then the `!mgp.context`
    // block arg, then one bare-`T` argument per input, and invokes `bodyBuilder`
    // with the builder arg, the context arg, and the input args. The callback is
    // expected to record nodes into the builder and emit the
    // `mgp.device_graph.yield` carrying the graph's output values; the result
    // `!mgp.device_graph`'s signature is then inferred as
    // `(type(inputs)) -> (type(yielded outputs))` (the context is build-time and
    // is not part of the signature). If `bodyBuilder` is null or leaves the body
    // unterminated, an operandless `mgp.device_graph.yield` (no graph outputs) is
    // supplied.
    OpBuilder<(ins
      "Value":$theContext,
      "Value":$deviceContext,
      "ValueRange":$inputs,
      CArg<"function_ref<void(OpBuilder &, Location, Value, Value, ValueRange)>",
           "nullptr">:$bodyBuilder)>
  ];

  let extraClassDeclaration = [{
    Block *getBodyBuilderBlock() {
      return &getBuildBody().front();
    }
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// DeviceGraphYieldOp
//===----------------------------------------------------------------------===//

def DeviceGraphYieldOp : MGP_DeviceGraphOp<"yield", [
  HasParent<"DeviceGraphCreateOp">,
  Pure,
  ReturnLike,
  Terminator,
]> {
  let summary = "Terminator for a `mgp.device_graph.create` build body.";
  let description = [{
    Marks the end of the recording build body of an enclosing
    `mgp.device_graph.create`, yielding the graph's output values. The operand
    types form the result `!mgp.device_graph` signature's results; the parent op
    instantiates the recorded builder into that graph after the body returns. A
    body with no graph outputs yields nothing (the operand list is empty).

    Example:
    ```mlir
      mgp.device_graph.yield %out : !mgp.buffer<<"gpu", 0>>
    ```
  }];

  let arguments = (ins Variadic<AnyType>:$outputs);
  let assemblyFormat = "attr-dict ($outputs^ `:` type($outputs))?";

  // No-operand builder for the parent's `SingleBlockImplicitTerminator` trait
  // (operands are populated after insertion); see DeviceGraphStagingYieldOp.
  let builders = [OpBuilder<(ins)>];
}

//===----------------------------------------------------------------------===//
// DeviceGraphExecuteOp
//===----------------------------------------------------------------------===//

def DeviceGraphExecuteOp : MGP_DeviceGraphOp<"execute", []> {
  let summary = "Replays a constructed device graph.";
  let description = [{
    Triggers replay of the given `!mgp.device_graph` on `deviceContext`.

    - `inputs`: read-only buffers/tensors whose contents are fed to the graph's
      frozen-input addresses at replay time. Types must match `graph`'s
      signature inputs.

    Replay produces the graph's outputs internally and surfaces them as the op's
    `outputs` results (value-returning, not destination-passing): one result per
    graph output, whose types match `graph`'s signature results. The op neither
    consumes nor produces a chain -- the `graph` operand orders the replay after
    the graph is built, and the output values carry the downstream data
    dependency.

    Example:
    ```mlir
      %out = mgp.device_graph.execute
          %graph, %gpu_ctx
          inputs(%in, %bias
              : !mgp.tensor<2x4xf32, <"gpu", 0>>, !mgp.tensor<2x3xf32, <"gpu", 0>>)
          : !mgp.device_graph<...>, !mgp.device_context<<"gpu", 0>>
            -> !mgp.tensor<2x3xf32, <"gpu", 0>>
    ```
  }];

  let arguments = (ins
    MGP_DeviceGraphType:$graph,
    MGP_DeviceContextType:$deviceContext,
    Variadic<AnyType>:$inputs
  );
  let results = (outs Variadic<AnyType>:$outputs);

  let assemblyFormat = [{
    $graph `,` $deviceContext
    `inputs` `(` ( $inputs^ `:` qualified(type($inputs)) )? `)`
    attr-dict
    `:` qualified(type($graph)) `,` qualified(type($deviceContext))
    ( `->` qualified(type($outputs))^ )?
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// DeviceGraphDependenciesOp
//===----------------------------------------------------------------------===//

def DeviceGraphDependenciesOp : MGP_DeviceGraphOp<"dependencies", [
  HasOnlyGraphRegion,
  RecursiveMemoryEffects,
  RegionKindInterface,
  SingleBlockImplicitTerminator<"DeviceGraphDependenciesYieldOp">,
]> {
  let summary = "A compiled region with explicit device-graph dependency edges.";
  let description = [{
    An operation used to model dependencies between regions of IR. In the
    incremental lowering process towards device-graph based execution, one
    `mpg.device_graph.dependencies` operation is created per
    `mogg.generic.compiled.region`.

    It separates the two things a compiled region entangles -- dataflow and
    ordering. It allows us to gradually merge the bodies of the
    `mogg.generic.compiled.region` operations in a model, while still tracking
    their execution order dependencies.

    Important features of this operation:

    - It is **not** `IsolatedFromAbove`: the body's operations
      reference enclosing SSA values directly (the region's dataflow boundary is
      dissolved at conversion time rather than re-threaded through operands).
      Execution order dependencies are encoded purely through the `dependencies`
      operands.
    - It takes the **device-graph builder as the `graphBuilder` operand** -- the
      enclosing builder recording into the graph under construction (it maps onto
      the `mgp.device_graph.staging` / `mgp.device_graph.create` builder). The
      operand makes the builder dependency explicit in the def-use graph.
    - Its body has exactly **one block argument**, a `!mgp.device_graph.builder`
      (arg #0), bound to the `graphBuilder` operand. The body's device-context
      uses are re-sourced from this block-arg builder via a
      `builtin.unrealized_conversion_cast` rather than referencing the outer
      device context, so the body is self-contained (local lowering).
    - Its remaining **operands are the dependency edges** --
      `!mgp.device_graph.node` values produced by the units this one must run
      after. That operand list is *only* those edges, not dataflow.
    - It **produces a `!mgp.device_graph.node`** representing this unit's
      contribution (successors depend on it), plus the values the former region
      yielded (so existing users are untouched). The node is synthesized by the
      op; it is not yielded by the body.

    The body terminates with `mgp.device_graph.dependencies.yield`, whose operand
    types must match the op's `bodyResults`.

    Example:
    ```mlir
      %node, %r = mgp.device_graph.dependencies builder(%b)
          dependencies(%depA, %depB) {
      ^bb0(%builder: !mgp.device_graph.builder):
        // ops copied from the former mogg.generic.compiled.region, referencing
        // enclosing SSA directly
        mgp.device_graph.dependencies.yield %v : !mgp.buffer<<"gpu", 0>>
      } : (!mgp.device_graph.node, !mgp.device_graph.node)
        -> (!mgp.device_graph.node, !mgp.buffer<<"gpu", 0>>)
    ```
  }];

  let arguments = (ins
    MGP_DeviceGraphBuilderType:$graphBuilder,
    Variadic<MGP_DeviceGraphNodeType>:$dependencies
  );

  let results = (outs
    MGP_DeviceGraphNodeType:$node,
    Variadic<AnyType>:$bodyResults
  );

  let regions = (region SizedRegion<1>:$body);

  let builders = [
    OpBuilder<(ins
      "Value":$graphBuilder,
      "ValueRange":$dependencies,
      CArg<"function_ref<void(OpBuilder &, Location, Value)>",
           "nullptr">:$bodyBuilder)>
  ];

  let assemblyFormat = [{
    `builder` `(` $graphBuilder `)`
    `dependencies` `(` ($dependencies^)? `)`
    $body attr-dict (`:` type($bodyResults)^)?
  }];

  let extraClassDeclaration = [{
    /// Returns the `mgp.device_graph.dependencies.yield` terminating the body.
    DeviceGraphDependenciesYieldOp getYieldOp();

    /// Returns the body's `!mgp.device_graph.builder` block argument (bound to
    /// the `graphBuilder` operand) -- the builder the body's recording work
    /// uses.
    TypedValue<MGP::DeviceGraphBuilderType> getBuilderBlockArgument();

    /// Returns true if the body region _may_ enqueue new device graph nodes,
    /// otherwise returns false. This is best effort based on the uses of the
    /// body region's builder operand.
    bool producesNewNodes();
  }];

  let hasCanonicalizer = 1;
  let hasFolder = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// DeviceGraphDependenciesYieldOp
//===----------------------------------------------------------------------===//

def DeviceGraphDependenciesYieldOp : MGP_DeviceGraphOp<"dependencies.yield", [
  HasParent<"DeviceGraphDependenciesOp">,
  Pure,
  ReturnLike,
  Terminator,
]> {
  let summary = "Terminator for a `mgp.device_graph.dependencies` body.";
  let description = [{
    Yields the former region's result values out of an enclosing
    `mgp.device_graph.dependencies` op. The operand types must match the parent
    op's `bodyResults`.
  }];

  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";

  // No-operand builder for the `SingleBlockImplicitTerminator` trait on the
  // parent, which inserts a default terminator into an empty body; the operands
  // are populated afterward.
  let builders = [OpBuilder<(ins)>];
}

//===----------------------------------------------------------------------===//
// DeviceGraphStagingOp
//===----------------------------------------------------------------------===//

def DeviceGraphStagingOp : MGP_DeviceGraphOp<"staging", [
  HasOnlyGraphRegion,
  IsolatedFromAbove,
  RecursiveMemoryEffects,
  RegionKindInterface,
  SingleBlockImplicitTerminator<"DeviceGraphStagingYieldOp">,
]> {
  let summary = "Container operation for staging the device-graph build body.";
  let description = [{
    A staging operation used by `--mgp-stage-device-graphs` to hold the
    device-graph build body *while it is being assembled* -- the
    `mgp.device_graph.dependencies` ops plus the `mogg.async` pack/unpack ops that
    mediate the `!mgp.*`/`!mogg.async<...>` and `!mo.tensor` type domains. It is
    eliminated when its cleaned contents are lifted into the
    `mgp.device_graph.create`, so it never reaches EmitMojo.

    Structurally it mirrors `mogg.generic.compiled.region`: it is
    `IsolatedFromAbove`, so the values its body needs come in as `inputs`
    operands of type `T` and are passed to the body as block arguments of the
    same bare type `T`. The body yields bare `T` values and the op's
    `stagedResults` are those same types. The boundary carries no
    `!mogg.async<T>` -- the verifier requires the block-arg and yield types to
    match the operand/result types exactly. The async pack/unpack needed to
    dissolve a `mogg.generic.compiled.region` body is injected locally by
    `mgp-compute-device-graph-dependencies` when it lowers each region.

    The body's first block argument is the `!mgp.device_graph.builder` (it has no
    corresponding operand); the builder the recording calls thread maps onto the
    `mgp.device_graph.create` builder block arg at finalization. It remains a
    graph region (the staged `mgp.device_graph.dependencies` ops carry forward
    node edges).

    Example:
    ```mlir
      %out = mgp.device_graph.staging inputs(%in : !mgp.tensor<2x3xf32, <"gpu", 0>>) {
      ^bb0(%builder: !mgp.device_graph.builder,
           %in_t: !mgp.tensor<2x3xf32, <"gpu", 0>>):
        // dependencies ops (each injecting its own async pack/unpack mediation)
        mgp.device_graph.staging.yield %t : !mgp.tensor<2x3xf32, <"gpu", 0>>
      } -> !mgp.tensor<2x3xf32, <"gpu", 0>>
    ```
  }];

  let arguments = (ins
    Variadic<AnyType>:$inputs,
    // As mogg.generic.compiled.region ops are inlined away, we will need a
    // place to store the constant resource attributes.
    // This operation is eventually synthesized back into a
    // mogg.generic.compiled.region, so it is the natural place.
    DefaultValuedAttr<ArrayAttr, "{}">:$constantResources
  );

  let results = (outs Variadic<AnyType>:$stagedResults);
  let regions = (region SizedRegion<1>:$body);

  let builders = [
    // Builds the op and its body in one shot: adds the `inputs` operands, creates
    // the body block with its `!mgp.device_graph.builder` block argument followed
    // by one `!mogg.async<T>` argument per input, and invokes `bodyBuilder` with
    // the builder arg and the async input args. The callback is expected to emit
    // the `staging.yield` carrying the (async-wrapped) staged values; the op's
    // result types are inferred by unwrapping those yield operand types. If
    // `bodyBuilder` is null or leaves the body unterminated, an operandless
    // `staging.yield` (zero results) is supplied.
    OpBuilder<(ins
      "ValueRange":$inputs,
      CArg<"function_ref<void(OpBuilder &, Location, Value, ValueRange)>",
           "nullptr">:$bodyBuilder)>
  ];

  let assemblyFormat = [{
    `inputs` `(` ($inputs^ `:` qualified(type($inputs)))? `)`
    $body attr-dict (`->` qualified(type($stagedResults))^)?
  }];

  let extraClassDeclaration = [{
    /// Returns the `mgp.device_graph.staging.yield` terminating the body.
    DeviceGraphStagingYieldOp getYieldOp();

    Block &getBodyBlock() {
      return getBody().front();
    }

    TypedValue<DeviceGraphBuilderType> getBodyBuilder() {
      return cast<TypedValue<DeviceGraphBuilderType>>(
        getBodyBlock().getArgument(0));
    }
  }];

  let hasCanonicalizer = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// DeviceGraphStagingYieldOp
//===----------------------------------------------------------------------===//

def DeviceGraphStagingYieldOp : MGP_DeviceGraphOp<"staging.yield", [
  HasParent<"DeviceGraphStagingOp">,
  Pure,
  ReturnLike,
  Terminator,
]> {
  let summary = "Terminator for a `mgp.device_graph.staging` body.";
  let description = [{
    Yields the values the rest of `execute` consumes out of an enclosing
    `mgp.device_graph.staging` op. The operand types must match the parent op's
    `stagedResults`.
  }];

  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";

  // No-operand builder for the `SingleBlockImplicitTerminator` trait on the
  // parent (see DeviceGraphDependenciesYieldOp).
  let builders = [OpBuilder<(ins)>];
}

#endif // GENERICML_MGPDIALECT_MGPDEVICEGRAPHOPS_TD
