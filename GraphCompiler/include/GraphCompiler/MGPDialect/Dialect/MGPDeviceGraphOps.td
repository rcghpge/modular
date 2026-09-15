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

include "GraphCompiler/Interfaces/EmitsMojoInterface.td"
include "GraphCompiler/MGPDialect/Dialect/MGPDialect.td"
include "GraphCompiler/MGPDialect/Dialect/MGPTypes.td"
include "GraphCompiler/MOGGDialect/MOGGTypes.td"
include "MLRT/GraphRT/MEFDialect/MEFTypes.td"
include "mlir/IR/OpAsmInterface.td"
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
  AttrSizedOperandSegments,
  DeclareOpInterfaceMethods<EmitsMojoInterface, ["isThrows"]>,
  IsolatedFromAbove,
  SingleBlockImplicitTerminator<"DeviceGraphYieldOp">,
]> {
  let summary = "Builds a device graph by recording into a builder.";
  let description = [{
    Returns an `!mgp.device_graph` built by invoking the build body. The op takes
    no chain and has no chain result -- the produced graph value carries the
    downstream dependency (`mgp.device_graph.execute` consumes it).

    When the optional `cache` operand is present, the graph is looked up in that
    cache under a key derived from the `inputs`, and the build body runs only on
    a miss; the newly built graph is then stored under the same key. Without the
    operand the op is stateless and rebuilds the graph on every execution.

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
    Optional<MGP_DeviceGraphCacheType>:$cache,
    Variadic<AnyType>:$inputs
  );
  let results = (outs MGP_DeviceGraphType:$outGraph);
  let regions = (region SizedRegion<1>:$buildBody);

  let assemblyFormat = [{
    $theContext `,` $deviceContext
    (`cache` `(` $cache^ `:` qualified(type($cache)) `)`)?
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
    // supplied. A null `cache` builds the op without the cache operand.
    OpBuilder<(ins
      "Value":$theContext,
      "Value":$deviceContext,
      "Value":$cache,
      "ValueRange":$inputs,
      CArg<"function_ref<void(OpBuilder &, Location, Value, Value, ValueRange)>",
           "nullptr">:$bodyBuilder)>
  ];

  let extraClassDeclaration = [{
    Block &getBodyBuilderBlock() {
      return getBuildBody().front();
    }

    BlockArgument getBodyBuilder() {
      return getBodyBuilderBlock().getArgument(0);
    }

    BlockArgument getBodyContext() {
      return getBodyBuilderBlock().getArgument(1);
    }

    MutableArrayRef<BlockArgument> getBodyInputs() {
      return getBodyBuilderBlock().getArguments().drop_front(2);
    }
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// DeviceGraphCacheOp
//===----------------------------------------------------------------------===//

def DeviceGraphCacheOp : MGP_DeviceGraphOp<"cache", [
  DeclareOpInterfaceMethods<EmitsMojoInterface, ["isThrows"]>,
  Pure,
]> {
  let summary = "Returns a model's device-graph cache from its state context.";
  let description = [{
    Returns the `!mgp.device_graph.cache` held by the `!mgp.context`, for
    `mgp.device_graph.create` to reuse graphs across executions. The cache is
    created on first use and lives as long as the state context, so it is shared
    by every execution of the model.

    Example:
    ```mlir
      %cache = mgp.device_graph.cache %ctx : !mgp.device_graph.cache
    ```
  }];

  let arguments = (ins MGP_ContextType:$theContext);
  let results = (outs MGP_DeviceGraphCacheType:$outCache);

  let assemblyFormat = [{
    $theContext attr-dict `:` qualified(type($outCache))
  }];
}

//===----------------------------------------------------------------------===//
// DeviceGraphYieldOp
//===----------------------------------------------------------------------===//

def DeviceGraphYieldOp : MGP_DeviceGraphOp<"yield", [
  DeclareOpInterfaceMethods<EmitsMojoInterface>,
  ParentOneOf<["DeviceGraphCreateOp", "DeviceGraphCollectiveCreateOp"]>,
  Pure,
  ReturnLike,
  Terminator,
]> {
  let summary = "Terminator for a device-graph build body.";
  let description = [{
    Marks the end of the recording build body of an enclosing
    `mgp.device_graph.create` or `mgp.device_graph.collective.create`, yielding
    the graph's output values. The operand types form the result
    `!mgp.device_graph` signature's results; the parent op instantiates the
    recorded builder into that graph after the body returns. A body with no
    graph outputs yields nothing (the operand list is empty).

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
    graph output, whose types match `graph`'s signature results. Device graphs
    have no notion of a chain, so `!mef.chain` never appears in the graph
    signature; instead this op produces its own `chain` result that represents
    completion of the replay, threading host-side ordering after the graph runs.

    Example:
    ```mlir
      %chain, %out = mgp.device_graph.execute
          %graph, %gpu_ctx
          inputs(%in, %bias
              : !mgp.tensor<2x4xf32, <"gpu", 0>>, !mgp.tensor<2x3xf32, <"gpu", 0>>)
          : !mgp.device_graph<...>, !mgp.device_context<<"gpu", 0>>
            -> !mef.chain, !mgp.tensor<2x3xf32, <"gpu", 0>>
    ```
  }];

  let arguments = (ins
    MGP_DeviceGraphType:$graph,
    MGP_DeviceContextType:$deviceContext,
    Variadic<AnyType>:$inputs
  );
  let results = (outs ChainType:$chain, Variadic<AnyType>:$outputs);

  let assemblyFormat = [{
    $graph `,` $deviceContext
    `inputs` `(` ( $inputs^ `:` qualified(type($inputs)) )? `)`
    attr-dict
    `:` qualified(type($graph)) `,` qualified(type($deviceContext))
    `->` qualified(type($chain)) ( `,` qualified(type($outputs))^ )?
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// DeviceGraphAllocOp
//===----------------------------------------------------------------------===//

def DeviceGraphAllocOp : MGP_DeviceGraphOp<"alloc", [
  DeclareOpInterfaceMethods<EmitsMojoInterface, ["isThrows"]>,
]> {
  let summary = "Allocates a buffer against a device-graph builder.";
  let description = [{
    The device-graph counterpart of `mgp.buffer.alloc`: allocates an
    uninitialized buffer of `runtimeSize` bytes, but takes the recording
    `!mgp.device_graph.builder` instead of a `!mgp.device_context`. Keeping
    allocation a builder operation lets the builder track device-graph
    allocations (e.g. a device-graph memory pool) in the future; today it lowers
    to a Mojo primitive that allocates via the builder's device context.

    It appears only inside a `mgp.device_graph.create` build body and is emitted
    to Mojo (never C++-lowered).

    Example:
    ```mlir
      %buf = mgp.device_graph.alloc %ch, %size, %builder, align 256
          : !mgp.buffer<<"gpu", 0>>
    ```
  }];

  let arguments = (ins
    ChainType:$inChain,
    Index:$runtimeSize,
    MGP_DeviceGraphBuilderType:$graphBuilder,
    DefaultValuedAttr<UI64Attr, "kUnknownSize">:$rawAlign
  );
  let results = (outs MGP_DeviceBufferType:$outBuffer);

  let assemblyFormat = [{
    $inChain `,` $runtimeSize `,` $graphBuilder ( `,` `align` $rawAlign^ )?
    attr-dict `:` qualified(type($outBuffer))
  }];
}

//===----------------------------------------------------------------------===//
// DeviceGraphRecordingContextOp
//===----------------------------------------------------------------------===//

def DeviceGraphRecordingContextOp : MGP_DeviceGraphOp<"recording_context", [
  DeclareOpInterfaceMethods<EmitsMojoInterface, ["isThrows"]>,
]> {
  let summary = "Views a device-graph builder as a recording device context.";
  let description = [{
    Extracts a recording `!mogg.context` from a `!mgp.device_graph.builder`.
    Operations dispatched against the returned context record into the builder
    (in enqueue order) instead of executing eagerly.

    This is how an execute-only kernel -- one without a node-free `build_graph`
    method -- is recorded during device-graph synthesis: the recording stub
    obtains this context and passes it as the launch argument of the kernel's
    `mogg.call.execute`, so the kernel's ordinary `execute` dispatches record
    into the graph. It appears only inside a device-graph build body and is
    emitted to Mojo (never C++-lowered), lowering to
    `DeviceGraphBuilder.recording_context()`.

    Example:
    ```mlir
      %ctx = mgp.device_graph.recording_context %builder : !mogg.context
    ```
  }];

  let arguments = (ins MGP_DeviceGraphBuilderType:$graphBuilder);
  let results = (outs MOGG_Context:$context);

  let assemblyFormat = "$graphBuilder attr-dict `:` qualified(type($context))";
}

//===----------------------------------------------------------------------===//
// DeviceGraphDeviceContextOp
//===----------------------------------------------------------------------===//

def DeviceGraphDeviceContextOp : MGP_DeviceGraphOp<"device_context", [
  DeclareOpInterfaceMethods<EmitsMojoInterface>,
  Pure,
]> {
  let summary = "Extracts the device context backing a device-graph builder.";
  let description = [{
    Reads the `!mgp.device_context` a `!mgp.device_graph.builder` was created
    from, lowering to `DeviceGraphBuilder.context()`.

    This is the *backing* context, not a recording view: work dispatched against
    it runs eagerly while the graph is being built instead of being recorded as
    a graph node. It is therefore only for in-body ops that need a real device
    context to construct or register a host-side view of device memory -- e.g.
    `mgp.buffer.bulk_slice`, which uses it for sub-buffer pointer math and
    pointer registration and enqueues nothing. An op that must record its work
    takes a `mgp.device_graph.recording_context` instead.

    Example:
    ```mlir
      %ctx = mgp.device_graph.device_context %builder
          : !mgp.device_context<<"gpu", 0>>
    ```
  }];

  let arguments = (ins MGP_DeviceGraphBuilderType:$graphBuilder);
  let results = (outs MGP_DeviceContextType:$deviceContext);

  let assemblyFormat =
    "$graphBuilder attr-dict `:` qualified(type($deviceContext))";
}

//===----------------------------------------------------------------------===//
// DeviceGraphDependenciesOp
//===----------------------------------------------------------------------===//

def DeviceGraphDependenciesOp : MGP_DeviceGraphOp<"dependencies", [
  DeclareOpInterfaceMethods<EmitsMojoInterface, ["isThrows"]>,
  HasOnlyGraphRegion,
  RecursiveMemoryEffects,
  RegionKindInterface,
  SingleBlockImplicitTerminator<"DeviceGraphDependenciesYieldOp">,
]> {
  let summary = "A compiled region with explicit device-graph dependency edges.";
  let description = [{
    An operation used to model dependencies between regions of IR. In the
    incremental lowering process towards device-graph based execution, one
    `mgp.device_graph.dependencies` operation is created per
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
    - Its body's nodes are **recorded in order**: `DeviceGraphBuilder.region`
      chains each node it records after the previous one, and the `dependencies`
      edges gate the first. The node result is therefore the body's last node.

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

    /// Returns the single body region block.
    Block &getBodyBlock();

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
  DeclareOpInterfaceMethods<EmitsMojoInterface>,
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

    MutableArrayRef<BlockArgument> getBodyInputs() {
      return getBodyBlock().getArguments().drop_front(1);
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

//===----------------------------------------------------------------------===//
// DeviceGraphStableCreateOp
//===----------------------------------------------------------------------===//

def DeviceGraphStableCreateOp : MGP_DeviceGraphOp<"stable.create", [
  InferTypeOpAdaptor,
  Pure,
]> {
  let summary = "Marks a tensor as requiring a stable address.";
  let description = [{
    Designates `$tensor` as a value that must be assigned a stable address when
    the device graph is partitioned across devices. The result has the same
    shape, device, and mutability as `$tensor`, but its `stable` flag is set.

    Example:
    ```mlir
      %shared = mgp.device_graph.stable.create %t
          : !mgp.tensor<4xf32, <"gpu", 0>>
    ```
  }];

  let arguments = (ins MGP_DeviceTensorType:$tensor);
  let results = (outs MGP_DeviceTensorType:$result);

  let assemblyFormat = [{
    $tensor attr-dict `:` qualified(type($tensor))
  }];

  let extraClassDeclaration = [{
    /// Get the stable address type that will back this tensor.
    MGP::StableAddressType getStableType() {
      return MGP::StableAddressType::get(getTensor().getType());
    }

    /// The device for the stable tensor.
    M::DeviceRefAttr getDevice() {
      return getTensor().getType().getDevice();
    }
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// DeviceGraphStableGetOp
//===----------------------------------------------------------------------===//

def DeviceGraphStableGetOp : MGP_DeviceGraphOp<"stable.get", [
  DeclareOpInterfaceMethods<EmitsMojoInterface, ["isThrows"]>,
  InferTypeOpAdaptor,
]> {
  let summary = "Reads a tensor view from a stable address; optional builder.";
  let description = [{
    Records a node that, at replay, reads the value currently published at
    `$addr`. When `$graphBuilder` is present, the node is recorded against that
    builder; when absent, the operation is recorded against an implicit
    device-graph context (used while the per-region builder plumbing is being
    worked out).

    At construction time the stable-address mapping is recorded on the device
    graph itself, so that the patched pointer resolves at launch time.

    A `!mgp.stable_addr<T>` produces a `!mgp.device_tensor<T, ..., stable>`
    whose shape/device are taken from the slot's element type `T`. The returned
    tensor is marked `stable` because its backing address is provided by the
    stable-address slot.
  }];

  let arguments = (ins
    MGP_StableAddressType:$addr,
    Optional<MGP_DeviceGraphBuilderType>:$graphBuilder
  );
  let results = (outs MGP_DeviceTensorType:$result);
  let assemblyFormat = [{
    $addr `:` qualified(type($addr))
     (`,` $graphBuilder^ `:` qualified(type($graphBuilder)))?
     attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// DeviceGraphStablePlaceOp
//===----------------------------------------------------------------------===//

def DeviceGraphStablePlaceOp : MGP_DeviceGraphOp<"stable.place", [
  DeclareOpInterfaceMethods<EmitsMojoInterface, ["isThrows"]>,
  InferTypeOpAdaptor,
  TypesMatchWith<"stable address type wraps the tensor type",
                 "tensor", "addr",
                 "::M::MGP::StableAddressType::get($_self)">,
]> {
  let summary = "Publishes a value into a stable address; optional builder.";
  let description = [{
    Records a node that, at replay, publishes `$tensor` at `$addr`. When
    `$graphBuilder` is present, the node is recorded against that builder; when
    absent, the operation publishes against an implicit device-graph context
    (used while the per-region builder plumbing is being worked out).

    At construction time the stable-address mapping is recorded on the device
    graph itself, so that the patched pointer resolves at launch time. A
    subsequent `mgp.device_graph.stable.get` in the same region that reads the
    same `$addr` sees this value at replay when enqueued after the place.

    The operation also returns a `!mgp.device_tensor` view backed by the stable
    address, identical to the value produced by `mgp.device_graph.stable.get`
    on the same `$addr`. This lets consumers on the source device reuse the
    placed value directly instead of emitting a separate `stable.get`.
  }];

  let arguments = (ins
    MGP_StableAddressType:$addr,
    MGP_DeviceTensorType:$tensor,
    Optional<MGP_DeviceGraphBuilderType>:$graphBuilder
  );
  let results = (outs MGP_DeviceTensorType:$placedTensor);
  let assemblyFormat = [{
    $addr `,` $tensor `:` qualified(type($tensor))
    (`,` $graphBuilder^ `:` qualified(type($graphBuilder)))?
    attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// DeviceGraphCollectiveOp
//===----------------------------------------------------------------------===//

def DeviceGraphCollectiveOp : MGP_DeviceGraphOp<"collective", [
  SingleBlockImplicitTerminator<"DeviceGraphCollectiveYieldOp">,
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmBlockArgumentNames"]>
]> {
  let summary =
    "Bundles a set of per-device device-graph bodies that must launch together.";
  let description = [{
    Holds one region per device. The op is the allocation point for the shared
    stable addresses used to coordinate the device graphs.

    The `$devices` attribute carries one `M_DeviceRefAttr` per region in
    region order, recording which device each region's body represents.

    Each region's body records a single device-graph on its respective device.
    Each region receives the same entry-block arguments: a
    `!mgp.device_graph.builder` followed by the shared stable-address slots.
    Each region terminates with `mgp.device_graph.collective.yield`, whose
    operands append to the parent op's `$outputs` in region order.

    The printed form hoists the entry-block arguments into two clauses:
    `builder(...)` for the shared `!mgp.device_graph.builder` block argument,
    followed by `stable(...)` for the shared stable-address block arguments.
    Each per-device region is introduced by the `DeviceRefAttr` of the device it
    runs on, and the region body is printed without its entry-block arguments
    because they are determined by the `builder(...)` and `stable(...)` clauses.

    Example:
    ```mlir
    %0:2 = mgp.device_graph.collective builder(%b) stable(
      %s0: !mgp.stable_addr<<4xf32, <"gpu", 0>>>,
      %s1: !mgp.stable_addr<<4xf32, <"gpu", 1>>>
    ) <"gpu", 0> {
      mgp.device_graph.collective.yield %val : !mgp.tensor<4xf32>
    }, <"gpu", 1> {
      mgp.device_graph.collective.yield %val : !mgp.tensor<4xf32>
    } : !mgp.tensor<4xf32>, !mgp.tensor<4xf32>
    ```
  }];

  let arguments = (ins
    M_DeviceRefsAttr:$devices
  );

  let results = (outs
    Variadic<AnyType>:$outputs
  );

  let regions = (region
    VariadicRegion<SizedRegion<1>>:$graphs
  );

  let extraClassDeclaration = [{
    /// Returns the number of regions.
    unsigned getNumRegions() { return getGraphs().size(); }

    /// Returns the block of region `idx`.
    Block &getRegionBlock(unsigned idx) {
      return getGraphs()[idx].front();
    }

    /// Returns the builder block argument for region `idx`.
    BlockArgument getRegionBuilder(unsigned idx) {
      return getRegionBlock(idx).getArgument(0);
    }

    /// Returns the stable-address block arguments for region `idx`.
    MutableArrayRef<BlockArgument> getRegionStableAddresses(unsigned idx) {
      return getRegionBlock(idx).getArguments().drop_front(1);
    }

    auto getGraphBlocks() {
      return llvm::map_range(getGraphs(), [](Region &region) {
        return &region.front();
      });
    }
  }];

  let builders = [
    OpBuilder<(ins
      "llvm::ArrayRef<M::DeviceRefAttr>":$devices,
      "llvm::ArrayRef<StableAddressType>":$stableTypes,
      CArg<"llvm::function_ref<void(::mlir::OpBuilder &, M::DeviceRefAttr, BlockArgument, llvm::ArrayRef<::mlir::BlockArgument>)>", "nullptr">:$bodyBuilder
    )>
  ];

  let hasCustomAssemblyFormat = 1;

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// DeviceGraphCollectiveYieldOp
//===----------------------------------------------------------------------===//

def DeviceGraphCollectiveYieldOp
    : MGP_DeviceGraphOp<"collective.yield", [
  HasParent<"DeviceGraphCollectiveOp">,
  Pure,
  ReturnLike,
  Terminator,
]> {
  let summary = "Terminator for a `mgp.device_graph.collective` device region.";
  let description = [{
    Yields the values this region contributes to the parent op's `$outputs`.
    The parent op appends operands across regions in region order.
  }];

  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat =
    "attr-dict ($operands^ `:` type($operands))?";

  // No-operand builder for the parent's `SingleBlockImplicitTerminator` trait
  // so the auto-generated terminator builder has the right signature; operands
  // are populated after insertion.
  let builders = [OpBuilder<(ins)>];
}

//===----------------------------------------------------------------------===//
// DeviceGraphCollectiveCreateOp
//===----------------------------------------------------------------------===//

def DeviceGraphCollectiveCreateOp : MGP_DeviceGraphOp<"collective.create", [
  AttrSizedOperandSegments,
  DeclareOpInterfaceMethods<EmitsMojoInterface, ["isThrows"]>,
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmBlockArgumentNames"]>,
  IsolatedFromAbove,
]> {
  let summary = "Builds one device graph per device from a collective body.";
  let description = [{
    Returns a variadic set of `!mgp.device_graph` values, one per region,
    by running each region's build body against its own recorder.

    The op is the multi-device counterpart of `mgp.device_graph.create`:
    each region records a single device graph belonging to a `!mgp.device_context`
    from the `deviceContexts` operand list. The regions share the same
    `theContext` and the same optional `cache`. The `inputs` operand list is
    the flat concatenation of each region's graph inputs, in region order;
    the i-th region's block input arguments consume the next slice of that
    list.

    Each region's block arguments are, in order: the recorder, the model context,
    then the region's slice of the `inputs` operands. The region must terminate
    with `mgp.device_graph.yield`; the yielded value types define the output
    signature of that region's graph.

    The printed form hoists the per-region argument lists next to the region
    they belong to: each region is prefixed with `graph(...)`, the builder
    block argument name, and the input block argument names and types. The
    shared `theContext` operand appears after the op name; the `cache` operand,
    when present, is printed after it. The region bodies are printed without
    their entry-block arguments because those arguments are declared by the
    prefix clauses.

    Example:
    ```mlir
      %g0, %g1 = mgp.device_graph.collective.create %ctx
        graph(%d0: !mgp.device_context<<"gpu", 0>>) builder(%b0)
            inputs(%i0: !mgp.buffer<<"gpu", 0>>) {
          mgp.device_graph.yield %i0 : !mgp.buffer<<"gpu", 0>>
      }, graph(%d1: !mgp.device_context<<"gpu", 1>>) builder(%b1)
            inputs(%i1: !mgp.buffer<<"gpu", 1>>) {
          mgp.device_graph.yield %i1 : !mgp.buffer<<"gpu", 1>>
      } : !mgp.device_graph<(!mgp.buffer<<"gpu", 0>>) -> (!mgp.buffer<<"gpu", 0>>)>,
          !mgp.device_graph<(!mgp.buffer<<"gpu", 1>>) -> (!mgp.buffer<<"gpu", 1>>)>
    ```
  }];

  let arguments = (ins
    MGP_ContextType:$theContext,
    Variadic<MGP_DeviceContextType>:$deviceContexts,
    Optional<MGP_DeviceGraphCacheType>:$cache,
    Variadic<AnyType>:$inputs
  );

  let results = (outs Variadic<MGP_DeviceGraphType>:$outGraphs);
  let regions = (region VariadicRegion<SizedRegion<1>>:$graphs);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// DeviceGraphCollectiveExecuteOp
//===----------------------------------------------------------------------===//

def DeviceGraphCollectiveExecuteOp : MGP_DeviceGraphOp<"collective.execute", [
  AttrSizedOperandSegments,
]> {
  let summary = "Replays a set of device graphs built by collective.create.";
  let description = [{
    Triggers replay of the `!mgp.device_graph` values produced by the
    matching `mgp.device_graph.collective.create`.

    The `graphs` operands are the per-device graphs, in the same order as the
    regions of the creating op. The `inputs` operands are the union of the
    live inputs for all graphs, laid out flat as
    `[graph0_inputs..., graph1_inputs..., ...]`; each graph itself knows how
    many inputs it consumes, so no extra per-graph segmentation metadata is
    required by the runtime primitive.

    The op produces a fresh `!mef.chain` for host-side ordering and a variadic
    set of value outputs, one per graph-output in graph order.
  }];

  let arguments = (ins
    Variadic<MGP_DeviceGraphType>:$graphs,
    Variadic<AnyType>:$inputs
  );
  let results = (outs ChainType:$chain, Variadic<AnyType>:$outputs);

  let assemblyFormat = [{
    $graphs `inputs` `(` ($inputs^ `:` qualified(type($inputs)))? `)` attr-dict
    `:` type($graphs) (`->` type($outputs)^)?
  }];
  let hasVerifier = 1;
}

#endif // GENERICML_MGPDIALECT_MGPDEVICEGRAPHOPS_TD
