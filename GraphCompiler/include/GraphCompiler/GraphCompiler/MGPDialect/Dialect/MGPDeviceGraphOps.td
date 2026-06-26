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
  AttrSizedOperandSegments,
  InferTypeOpAdaptor,
  IsolatedFromAbove,
  SingleBlockImplicitTerminator<"DeviceGraphYieldOp">,
]> {
  let summary = "Builds or retrieves a cached device graph.";
  let description = [{
    Returns an `!mgp.device_graph`, either by invoking the build body and
    storing the result in the given state slot, or by retrieving a previously
    cached value from that slot.

    Operands are split into two groups, mirroring the eventual replay-time
    signature recorded on the result `!mgp.device_graph`:
    - `inputs`: read-only buffers/tensors that the graph will treat as
      frozen inputs at replay time.
    - `outputs`: caller-allocated buffers/tensors that the graph will write
      through at replay time.

    The op owns the device-graph builder lifecycle. It creates a
    `!mgp.device_graph.builder` from its `deviceContext` operand and passes that
    builder to the build body as the body's first block argument; after the body
    returns, the op instantiates the recorded builder into its result
    `!mgp.device_graph`. The build body therefore only records nodes -- it does
    not create the builder or produce the graph.

    The body's block arguments are, in order: the `!mgp.device_graph.builder`,
    then the `inputs`, then the `outputs`. As in `mogg.generic.compiled.region`,
    the input/output arguments arrive as `!mogg.async<T>` wrappers around the
    corresponding operand type `T` -- the region boundary wraps the
    buffers/tensors into async values on entry and the body unpacks them via
    `mogg.async.unpack` (a body that consumes an argument directly as the plain
    `T` is also accepted). On a cache miss the body runs and must terminate with
    an operandless `mgp.device_graph.yield`. On a cache hit the body is not
    invoked.

    Note: in the initial implementation every call is a cache miss -- the
    slot operand is reserved for the caching machinery added in Phase 2.

    Example:
    ```mlir
      %ch1, %graph = mgp.device_graph.create
          %ch0, %ctx[3], %gpu_ctx
          inputs(%in, %bias, %wt
              : !mgp.buffer<<"gpu", 0>>, !mgp.buffer<<"gpu", 0>>,
                !mgp.buffer<<"gpu", 0>>)
          outputs(%out : !mgp.buffer<<"gpu", 0>>) ({
        ^bb0(%builder: !mgp.device_graph.builder,
             %in_b: !mogg.async<!mgp.buffer<<"gpu", 0>>>,
             %bias_b: !mogg.async<!mgp.buffer<<"gpu", 0>>>,
             %wt_b: !mogg.async<!mgp.buffer<<"gpu", 0>>>,
             %out_b: !mogg.async<!mgp.buffer<<"gpu", 0>>>):
          // ... unpack args, record nodes into %builder ...
          mgp.device_graph.yield
      }) : !mgp.device_context<<"gpu", 0>> -> !mgp.device_graph<...>
    ```
  }];

  // CAUTION: attribute arguments must be in alphabetical order by name.
  let arguments = (ins
    ChainType:$inChain,
    MGP_ContextType:$theContext,
    MGP_DeviceContextType:$deviceContext,
    Variadic<AnyType>:$inputs,
    Variadic<AnyType>:$outputs,
    UI64Attr:$bGraphSlot
  );
  let results = (outs
    ChainType:$outChain,
    MGP_DeviceGraphType:$outGraph
  );
  let regions = (region SizedRegion<1>:$buildBody);

  let assemblyFormat = [{
    $inChain `,` $theContext `[` $bGraphSlot `]` `,` $deviceContext
    `inputs` `(` ( $inputs^ `:` qualified(type($inputs)) )? `)`
    `outputs` `(` ( $outputs^ `:` qualified(type($outputs)) )? `)`
    $buildBody attr-dict
    `:` qualified(type($deviceContext)) `->` qualified(type($outGraph))
  }];

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
    `mgp.device_graph.create`. It takes no operands: the build body only
    records nodes into the builder the parent op supplies, and the parent op
    instantiates that builder into its result `!mgp.device_graph` after the body
    returns -- the body does not produce the graph itself.

    Example:
    ```mlir
      mgp.device_graph.yield
    ```
  }];

  let assemblyFormat = "attr-dict";
}

//===----------------------------------------------------------------------===//
// DeviceGraphExecuteOp
//===----------------------------------------------------------------------===//

def DeviceGraphExecuteOp : MGP_DeviceGraphOp<"execute", [
  AttrSizedOperandSegments,
]> {
  let summary = "Replays a constructed device graph.";
  let description = [{
    Triggers replay of the given `!mgp.device_graph` on `deviceContext`.

    Operands are split into two groups:
    - `inputs`: read-only buffers/tensors whose contents are copied into the
      graph's frozen-input addresses at replay time. Types must match
      `graph`'s signature inputs.
    - `outputs`: caller-allocated buffers/tensors the graph writes through
      in-place. Types must match `graph`'s signature results.

    The op returns only a chain; result values are observable through the
    caller-supplied output buffers.

    Example:
    ```mlir
      %ch1 = mgp.device_graph.execute
          %ch0, %graph, %gpu_ctx
          inputs(%in, %bias, %wt
              : !mgp.buffer<<"gpu", 0>>, !mgp.buffer<<"gpu", 0>>,
                !mgp.buffer<<"gpu", 0>>)
          outputs(%out : !mgp.buffer<<"gpu", 0>>)
          : !mgp.device_graph<...>, !mgp.device_context<<"gpu", 0>>
    ```
  }];

  let arguments = (ins
    ChainType:$inChain,
    MGP_DeviceGraphType:$graph,
    MGP_DeviceContextType:$deviceContext,
    Variadic<AnyType>:$inputs,
    Variadic<AnyType>:$outputs
  );
  let results = (outs ChainType:$outChain);

  let assemblyFormat = [{
    $inChain `,` $graph `,` $deviceContext
    `inputs` `(` ( $inputs^ `:` qualified(type($inputs)) )? `)`
    `outputs` `(` ( $outputs^ `:` qualified(type($outputs)) )? `)`
    attr-dict
    `:` qualified(type($graph)) `,` qualified(type($deviceContext))
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
      dissolved at conversion time rather than re-threaded through operands). The
      body has no block arguments. Execution order dependencies are encoded purely
      through the `dependencies` operands.
    - Its **operands are the dependency edges** -- `!mgp.device_graph.node`
      values produced by the units this one must run after. The operand list is
      *only* those edges, not dataflow.
    - It **produces a `!mgp.device_graph.node`** representing this unit's
      contribution (successors depend on it), plus the values the former region
      yielded (so existing users are untouched). The node is synthesized by the
      op; it is not yielded by the body.

    The body terminates with `mgp.device_graph.dependencies.yield`, whose operand
    types must match the op's `bodyResults`.

    Example:
    ```mlir
      %node, %r = mgp.device_graph.dependencies dependencies(%depA, %depB) {
        // ops copied from the former mogg.generic.compiled.region, referencing
        // enclosing SSA directly
        mgp.device_graph.dependencies.yield %v : !mgp.buffer<<"gpu", 0>>
      } : (!mgp.device_graph.node, !mgp.device_graph.node)
        -> (!mgp.device_graph.node, !mgp.buffer<<"gpu", 0>>)
    ```
  }];

  let arguments = (ins Variadic<MGP_DeviceGraphNodeType>:$dependencies);
  let results = (outs
    MGP_DeviceGraphNodeType:$node,
    Variadic<AnyType>:$bodyResults
  );
  let regions = (region SizedRegion<1>:$body);

  let builders = [
    OpBuilder<(ins
      "ValueRange":$dependencies,
      CArg<"function_ref<void(OpBuilder &, Location)>",
           "nullptr">:$bodyBuilder)>
  ];

  let assemblyFormat = [{
    `dependencies` `(` ($dependencies^)? `)` $body attr-dict (`:` type($bodyResults)^)?
  }];

  let extraClassDeclaration = [{
    /// Returns the `mgp.device_graph.dependencies.yield` terminating the body.
    DeviceGraphDependenciesYieldOp getYieldOp();
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
    operands of type `T` and are passed to the body as block arguments of type
    `!mogg.async<T>` (the body unpacks them via `mogg.async.unpack`). The body
    yields `!mogg.async<T>` values and the op's `stagedResults` are the unwrapped
    `T`. This keeps the sync/async boundary explicit at the op edge rather than
    relying on implicit conversions inside the body.

    The body's first block argument is the `!mgp.device_graph.builder` (it has no
    corresponding operand); the builder the recording calls thread maps onto the
    `mgp.device_graph.create` builder block arg at finalization. It remains a
    graph region (the staged `mgp.device_graph.dependencies` ops carry forward
    node edges).

    Example:
    ```mlir
      %out = mgp.device_graph.staging inputs(%in : !mgp.tensor<2x3xf32, <"gpu", 0>>) {
      ^bb0(%builder: !mgp.device_graph.builder,
           %in_a: !mogg.async<!mgp.tensor<2x3xf32, <"gpu", 0>>>):
        // unpack inputs, dependencies ops + async pack/unpack mediation, pack outputs
        mgp.device_graph.staging.yield %t : !mogg.async<!mgp.tensor<2x3xf32, <"gpu", 0>>>
      } -> !mgp.tensor<2x3xf32, <"gpu", 0>>
    ```
  }];

  let arguments = (ins Variadic<AnyType>:$inputs);
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
