# Modular graph compiler pipeline description

## Introduction

This document focuses on describing the current state of the optimizations,
lowerings, and analyses in the graph compiler. It describes the system as-is,
using only the source code and a bit of git history without any prior knowledge
of the system.

### Format

The description of a pass should follow the schema:

```text
#### Description
A brief state of purpose and a high-level overview of the pass and how it works.

#### Dialects
Produced and consumed dialects or dependencies.

#### Performance
Notes on performance effects in a broad sense.

#### Additional notes
Optional. Any relevant information in the context of the document.
```

## Dialects

Dialects have extensive documentation. Here's a quick reference:

- **RMO (Relaxed Modular Operations)** - accepts incomplete type information
  from frontends and lowers to MO. Definition:
  `GraphCompiler/include/GraphCompiler/GraphCompiler/RMODialect/RMODialect.td`.
  Docs: `GraphCompiler/docs/GraphCompiler.md` (RMO section).
- **MO (Modular Operations)** - framework-agnostic, strongly-typed graph ops
  with value semantics. Definition:
  `GraphCompiler/include/GraphCompiler/GraphCompiler/MODialect/MODialect.td`.
  Docs: `GraphCompiler/docs/GraphCompiler.md` (MO section).
- **MOSH (Modular Shape)** - shape/parameter computation for MO graphs.
  Definition:
  `GraphCompiler/include/GraphCompiler/GraphCompiler/MOSHDialect/MOSHDialect.td`.
  Docs: `GraphCompiler/docs/GraphCompiler.md` (MOSH mention).
- **MOGG (Modular Operation Generator Graph)** - kernel selection/fusion and
  codegen-specific IR. Definition:
  `GraphCompiler/include/GraphCompiler/GraphCompiler/MOGGDialect/MOGGDialect.td`.
  Docs: `GraphCompiler/docs/GraphCompiler.md` (MOGG section).
- **MGP (Modular Glue Primitives)** - runtime glue dialect with buffer
  semantics; directly translatable to MEF. Definition:
  `GraphCompiler/include/GraphCompiler/GraphCompiler/MGPDialect/Dialect/MGPDialect.td`.
  Docs: `GraphCompiler/docs/GraphCompiler.md` (MGP section).

Other referenced dialects, not directly part of the compiler:

- **KGEN** - kernel-generation dialect used during JIT compilation. Definitions:
  `KGEN/include/KGEN/KGENDialect/KGENDialect.td` and related KGEN dialect files.
  Docs: `KGEN/docs/manual/PassesAndIR.md`.
- **HLCF** - high-level control flow dialect used in kernel-generation
  pipelines. Definition: `KGEN/include/KGEN/HLCFDialect/HLCFDialect.td`. Docs:
  `KGEN/docs/manual/PassesAndIR.md`.
- **LIT** - Mojo frontend dialect used for parsed Mojo IR in kernel compilation
  pipelines. Definition: `KGEN/include/KGEN/LITDialect/LITDialect.td`. Docs:
  `KGEN/docs/manual/PassesAndIR.md`.
- **POP** - Mojo low-level/runtime support dialect used in kernel generation and
  lowering. Definition: `KGEN/include/KGEN/POPDialect/POPDialect.td`. Docs:
  `KGEN/docs/manual/PassesAndIR.md`.
- **GRT (GraphRT)** - runtime primitive dialect for execution in GraphRT.
  Definition: `MLRT/include/MLRT/GraphRT/GRTDialect/GRTDialect.td`. Docs:
  `MLRT/docs/GraphRT/KeyConceptsInGraphRT.md`.
- **MEF** - low-level execution format dialect used by the MEF executor.
  Definition: `MLRT/include/MLRT/GraphRT/MEFDialect/MEFDialect.td`. Docs:
  `MLRT/docs/GraphRT/ModularExecutorFormat.md`.

## Analyses

### MojoLibraryAnalysis

#### Description

Loads and caches Mojo kernel libraries (standard + extra paths) and exposes
lookup helpers for registered kernel generators and symbols. It materializes a
module of exported generators and serves queries needed by kernel selection and
lowering.

#### Dialects

Consumes `mo` and depends on `lit` support for loading Mojo packages and
generators.

#### Performance

Amortizes library parsing/validation across passes by caching the loaded
libraries; reduces repeated kernel discovery costs during lowering/codegen.

#### Additional notes

Initialized with `kernelLibraryPaths`, then lazily loads Mojo packages and
builds a generator table used by lowering passes (e.g., MO to MOGG).

Custom kernels are registered on Mojo side through `@compiler.register(...)`.
The compiler will later use the symbol to match the kernel during lowering.
Custom ops can target these using `mo.custom {symbol = "MyOp"}`
(see <ExtensibilityUserGuide.md>).

The analysis parses Mojo packages/bytecode into MLIR, walks every
function/struct, and builds multiple lookup tables with registered kernels,
execute/shape/view functions, opaque-type map. By default, the main two
components are loaded:

- MOGGKernelAPI - a registry of hand‑fused/high‑level kernels and helpers.
  Normally, accessed via `mo.custom` and targeted in the PatternFusion pass.
- MOGGPrimitives - baseline Mojo kernels/primitives used by the graph compiler
  as standard kernel implementations (e.g., core ops like tensor construction
  `mgp.tensor.create` or allocation `mgp.buffer.alloc`, views, elementwise
  helpers, etc.). These are normally introduced in the MO->MGP lowering
  (MoToPrimitives and related lowering passes) and then included through kernel
  stubs produced by EmitMojo pass.

Source:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/include/GraphCompiler/GraphCompiler/MOGGDialect/Analysis/MojoRegistry.h>
Implementation:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/lib/GraphCompiler/MOGGDialect/Analysis/MojoRegistry.cpp>

### ParametricTopologyAnalysis

#### Description

Extends SSA dataflow with parametric dependencies (shape/parameter attributes)
and nested-region providers, enabling correct dependency queries and
topological sorting when attributes (not SSA values) drive ordering.

#### Dialects

Operates on `mo` graphs and their nested regions; models attribute-level
dependencies (e.g., `kgen` parameter attrs) alongside SSA.

#### Additional notes

This is a custom analysis (technically, not an MLIR analysis) that allows passes
to reason about true dependency order that are not expressed via SSA values.
Provides `users()`/`providers()` that union SSA and parametric dependencies, and
`topologicallySortAllOpsWithin`/`computeTopologicalSorting` helpers. Source:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/include/GraphCompiler/GraphCompiler/MODialect/Analysis/ParametricTopologyAnalysis.h>

### BufferLifetimeAnalysis

#### Description

Builds dependency trees over `mgp.model` blocks to reason about execution order
and buffer lifetimes, enabling safe memory planning and buffer reuse.

#### Dialects

Operates on `mgp` (ModelOp) and buffer operations.

#### Performance

Enables downstream memory planning passes to reuse buffers by providing fast
`executesAfter`/LCA queries over dependency trees.

#### Additional notes

Tracks a per-block `DependencyTree` with pre/post indices, supports dominance
and LCA queries, and includes alias-aware alloc-use discovery via
`getAllocUses`. Source:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/include/GraphCompiler/GraphCompiler/MGPDialect/Dialect/MGPBufferLifetimeAnalysis.h>

## Lowering passes

### MOToMOGG

#### Description

Converts eligible `mo` ops into `mogg` ops while selecting kernels from the
registered Mojo libraries. While lowering removes most `mo.rebind` ops with
known shapes to avoid blocking fusion.

#### Dialects

Consumes and produces `mo` and `mogg`, and depends on runtime/target dialects
used by kernel selection (`nvvm`, `rocdl`, `index`) and any other device
dialect, plus a transitive dependency on `lit`.

#### Additional notes

The pass consists of two phases. The first one replaces rebind ops with their
inputs unless they carry important information. This includes unknown shapes,
present layout attributes (matmul/conv), refined symbolic dims to static ones,
or the user is not being lowered to `mogg`. The redundant ops come from type
normalization/canonicalization and `rmo` legalization. The goal of the first
stage is to get rid of those to unblock fusion.

The second stage pulls up the `MojoLibraryAnalysis` and walks all the
operations in the module, lowering all the eligible ops. There are two groups
of `mo` ops that are preserved during the lowering: (1) any control flow ops
(deriving from `ControlOp`), and (2) a set of ops that are likely not converted
to a compute kernel. The latter includes `mo.graph` and some other runtime
plumbing ops like buffer transfers, materialization, external resources
management, views and list ops (controlled by
`PreservedDuringKernelLowering`). Those usually have either specialized
semantics or custom lowerings and survive until `MOGGToMGP` later in the
pipeline.

For eligible ops, a `MOGGKernelBuilder` controls the kernel selection. The
selection is performed by a Mojo kernel adaptor and consists of two steps:

1. Map the op to a kernel name + function label under which the kernel is
   registered in Mojo. Usually, the compiler searches the kernel by the op
   name with some exceptions when it has to synthesize it (e.g., custom ops use
   the symbol directly). The op arguments mapping (to kernel arguments) is
   type- (compatibility) and position-driven.
2. The target kernel is resolved via `MojoLibraryAnalysis`.

The adaptor becomes the handle that ties the selected kernel to the op in the
IR, and a `mogg.kernel` is created in the place of the original op. Depending
on the type of the selected kernel (e.g., view/elementwise/shape/...; the
interface depends on the type and later stages can use this as metadata to make
decisions such as fusion), an explicit call to the kernel is inserted into the
execute region of the `mogg.kernel` container (`mogg.call.execute` for normal
kernels). Finally, he lowering forwards kernel parameters and forward-declares
all the referenced library functions.

Source:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/lib/GraphCompiler/Conversion/MOToMOGG/MOToMOGG.cpp>
Pass definition:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/include/GraphCompiler/GraphCompiler/Conversion/MOToMOGG.td>

### MOGGToMGP

#### Description

Converts `mo`/`mogg` ops into `mgp`/`mef` primitives at the module level,
producing a runtime-oriented graph that can be further lowered and optimized.

#### Dialects

Consumes `mo`/`mogg` and produces `mgp`/`mef`, with runtime support from `grt`,
`pop`, and `index`.

#### Performance

Enables later MGP pipeline optimizations and kernel outlining by moving from MO
to buffer/primitive semantics. Supports module-level orchestraction
(topological dependency order).

#### Additional notes

Runs as a module pass because it invokes MO to MOGG lowering and needs to avoid
per-graph setup overhead. The pass lowers `mo.graph` into an `mgp.model` with
`init` and `execute` regions.

The pass collects all `mo.graph`s and builds a dependency map from `mo.call`.
Then lowers graphs in the topologically sorted order while detecting dependency
cycles. Each graph is lowered to a single `mgp` model, walking its body,
arguments and patching the control flow with `grt`. The lowering of ops goes
either through the `LowerableToMGPInterface` or dispatches to a specialized
handler if an op has one. Note that this includes `mogg.kernel` to
`mogg.generic.compiled.region` lowering. `mo.call`s are lowered to `mgp.call`s
with implicit operands that include chains, context, weight registry, device
contexts.

Within blocks, operations are sorted in levelwise topological order. This is a
workaround for `mo.if` preventing weight transformations from moving to init
region (`mgp.buffer.alloc` for weights is chained after `grt.if`, preventing
ExecInvariantCodeMotion pass from moving the weight transformation to model
init). Early operations reliably sit before control-flow.

At this stage, bufferization (value-semantic to buffer-semantic transition) is
performed. The logic lives in `MOOpLoweringState`. For inputs, the lowering
extracts the underlying buffer and spec for each tensor from an already-lowered
`mgp.tensor` value. Outputs are lowered in destination-passing style: create a
tensor spec, compute size, allocate a buffer, and then build an
`mgp.tensor.create`.

The difference between the shape of a tensor being static or dynamic is in the
materialization of the tensor spec. For dynamic, runtime dims are expressed via
`mosh`. Constants may introduce additional buffer allocation + transfer.

Source:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/lib/GraphCompiler/Conversion/MOGGToMGP/MOGGToMGP.cpp>
Pass definition:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/include/GraphCompiler/GraphCompiler/Conversion/MOGGToMGP.td>

## MGP passes

### LowerPrimitives

#### Description

Wraps eligible MGP/POP/index ops into mogg.generic.compiled.region kernels and
applies the MOGG ABI to prepare for fusion and Mojo emission.

#### Dialects

Consumes and produces `mgp` and depends on `kgen` (`lit`, `pop`) plus `mef`,
`mogg`, `index`.

#### Additional notes

The pass aids the sequence fusion pass by lowering some top-level model ops to
uniform compiled region kernels. It walks the ops that implement
`EmitsMojoInterface`, wraps them into a new `mogg.generic.compiled.region`, and
rewrites the complied region's ABI to MOGG ABI. The MOGG ABI means the
async‑value calling convention used for mogg.generic.compiled.region: inputs
are unpacked from async wrappers on entry, and outputs are packed back into
async wrappers on exit, with special handling for chains and borrowed buffers.
The ABI lowering does three things:

1. For arguments marked "shouldUnpack" (by emits mojo interface), it inserts
   `mogg.async.unpack` (or borrow‑aware variants) and rewrites uses to the
   unpacked values.
2. For results marked "shouldPack", it wraps them with `mogg.async.pack` /
   `mogg.async.pack.borrow` variants. Chains are replaced by
   `mogg.async.ready` and sunk with `mogg.chain.sink` to preserve side effects.
3. Any chain arguments are replaced with a dummy chain, so the body doesn’t
   accidentally use real chain inputs.

The pass also performs special lowering for `index.constant` ->
`mgp.buffer.constant`, `mgp.buffer.to_index` and a resource attribute (no kernel
hashing on constants).

Relies on `MojoLibraryAnalysis` but never uses it.

Source:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/lib/GraphCompiler/Conversion/MOGGToMGP/LowerPrimitives.cpp>
Pass definition:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/include/GraphCompiler/GraphCompiler/Conversion/MOGGToMGP.td>

### LookupConstantResources

#### Description

Replaces `mgp.buffer.constant` uses inside `mogg.kernel.stub` with
`mgp.resource.constant` indices that reference the stub’s constant resource
table. This makes constants comparable by index rather than by value.

#### Dialects

Consumes and produces `mgp` (within `mogg.kernel.stub`).

#### Performance

Improves kernel stub merging and downstream compilation by avoiding large
constant payloads in the IR and allowing identical kernels to share constants.

#### Additional notes

The pass runs late in the pipeline, right before emitting Mojo. It walks
`BufferConstantOp` in a `KernelStubOp`, resolves each constant against the
stub’s resource list, then replaces it with `ResourceConstantOp`, and resets
the resource array to a dummy list of the same size.

The pass expects all the resources to be found in the constant resources array
attached to the `KernelStubOp` by the `MO::ConstantOp` lowering. The resources
are referenced by index, so that multiple kernels can access the same resource.
This has many obvious benefits including memory efficiency, avoiding hashing
kernels by constant values, and compilation efficiency. Earlier in the
pipeline, index constants are converted to `mgp.buffer.constant` (by the
LowerPrimitives pass) with a resource attribute for the same reason.

Source:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/lib/GraphCompiler/MOGGDialect/Transforms/LookupConstantResources.cpp>
Pass definition:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/include/GraphCompiler/GraphCompiler/Conversion/MOGGToMGP.td>

### ExecInvariantCodeMotionPass (EICM)

#### Description

Moves computations from the `execute` block to the `init` block when they do
not depend on `execute` arguments. Uses `!mgp.context` slots to forward results,
with heuristics to avoid lifting trivial or awkward-to-cache ops.

#### Dialects

Consumes and produces `mgp` (ModelOp), with control-flow support from `grt` and
buffer plumbing from `mef`.

#### Performance

Reduces per-execution overhead by hoisting invariant work to initialization and
enables reuse of cached buffers.

#### Additional notes

The goal is to hoist execute‑invariant work to init, so it is computed once per
model rather than on every execution, while respecting side‑effect and
control‑flow safety. The pass runs early in the MGP pipeline, after MO has been
lowered to `mgp.model`(only available after MoToPrimitive) and after some basic
cleanup, but before memory planning and other buffer‑lifecycle optimizations
(these assume the init/execute split is already settled).

The pass defines "liftable" ops (buffer‑creating ops and chain‑creating ops) and
recomputable ops (pure ops) that can be moved or duplicated safely. The
condition is that an op does not:

- depend on execute‑block arguments,
- have side effects,
- is inside control‑flow,
- is explicitly unliftable (e.g., external constants that can be mutated
  outside the system; this can be a pytorch tensor).

The condition is determined by a set of analyses. A dense forward relation
analysis over values/operands propagates an actual dependency on the argument
or if an value is produced by an unliftable op, including control flow that is
considered intrinsically unliftable. A set of backward analyses make sure that:

- Ops whose results are mutated by unliftable ops are not lifted (values cached
  in `!mgp.context` are read‑only).
- Liftable ops whose results are required by unliftable ops which can not be
  conveyed via `!mgp.context` are cloned.
- Trivial ops with unliftable consumers are not lifted (recomputing is cheaper
  than caching).

The pass runs the analyses to convergence. Then the hoisting is performed. The
latter must consider preserving side-effect ordering (this involves some
gating placeholder chain creation until the pass can compute the real
dependency chain).

Source:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/lib/GraphCompiler/MGPDialect/Transforms/ExecInvariantCodeMotionPass.cpp>
Pass definition:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/include/GraphCompiler/GraphCompiler/MGPDialect/Transforms/MGPTransforms.td>

### FixupOverhangsPass

#### Description

Finds chain results with no uses and adds waits on the block terminator so all
chains are awaited before graph exit.

#### Dialects

Consumes and produces `mgp` (ModelOp).

#### Performance

Adds minimal extra dependencies to preserve correctness and ensure async errors
are not silently dropped.

#### Additional notes

Part of the JIT compilation pipeline. Walks each block in `mgp.model`, collects
unused results (one per op), and uses `alsoWaitForChains` to add dependencies on
the block output. An unused result is assumed to be a dangling chain (this is an
IR invariant at this stage). The pass fulfills the assumption of the model being
'overhang free' (see
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/include/GraphCompiler/GraphCompiler/MGPDialect/Transforms/AsyncDataflowDependencies.h>)

Source:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/lib/GraphCompiler/MGPDialect/Transforms/FixupOverhangsPass.cpp>
Pass definition:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/include/GraphCompiler/GraphCompiler/MGPDialect/Transforms/MGPTransforms.td>

### SimplifyExtendLifetimes

#### Description

Fuses `mgp.extend.lifetime` into a producer `mogg.generic.compiled.region` by
threading the extend-lifetime inputs directly into the kernel op and removing
the explicit lifetime extension op.

#### Dialects

Consumes `mgp`, rewriting uses of `mogg.generic.compiled.region` and `mef`
chain values.

#### Performance

Reduces IR size and avoids standalone lifetime-extension ops by folding them
into the kernel boundary.

#### Additional notes

`mgp.extend.lifetime` are created at MO to MGP lowering (during MOGGToMGP)
whenever a transfer is introduced (e.g., host->device or device->host copies;
so, both source and destination buffers). When buffer builder emits a transfer,
it adds an `mgp.sync` on the copy chain and then a `mgp.extend.lifetime` to keep
the relevant buffer alive until that sync completes.

The pass is run once the kernel boundaries and operand lists are settled, so
after the fusion, right before kernel outlining (after which there's not enough
information remaining).

Source:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/lib/GraphCompiler/MGPDialect/Transforms/SimplifyExtendLifetimesPass.cpp>
Pass definition:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/include/GraphCompiler/GraphCompiler/MGPDialect/Transforms/MGPTransforms.td>

### ConcatInPlace

#### Description

Rewrites `mgp.buffer.concat` into a single output allocation plus
`mgp.buffer.split.borrow`-style slices so that producers write directly into
the final output buffer.

#### Dialects

Consumes and produces `mgp` (ModelOp) with chains and `index` buffer-size values
used to compute offsets.

#### Performance

Eliminates explicit concat copies by turning concatenation into in-place writes
to slices of the output buffer.

#### Additional notes

Replace an explicit concat that writes into a freshly allocated output buffer
with a split‑borrow pattern: allocate one big buffer, slice it into per‑input
buffers, and redirect all writers to write directly into those slices. This
makes concat "in‑place" and removes the op itself.

Runs over the execute region, finds `mgp.buffer.concat` ops whose input buffers
and output buffer are all explicitly allocated and distinct, then computes
offsets from the runtime sizes. It inserts slices of the output buffer at the
input-buffer allocation sites, replaces each input buffer with its slice, and
erases the original allocs and concat op.

The pass assumes the concat input chain is only ready once all input buffers
are ready, so it replaces the concat output chain with the input chain.

Source:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/lib/GraphCompiler/MGPDialect/Transforms/ConcatInPlacePass.cpp>
Pass definition:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/include/GraphCompiler/GraphCompiler/MGPDialect/Transforms/MGPTransforms.td>

### OutlineKernels

#### Description

Outlines `mogg.generic.compiled.region` bodies into `mogg.kernel.stub` ops and
replaces the original region with `mogg.execute.compiled` calls.

#### Dialects

Consumes and produces `mogg`/`mgp` within a `ModuleOp`.

#### Performance

Outlining "stabilizes" the kernel boundaries before the Mojo codegen making
them explicit and deterministic. The entry points get uniform and addressable
(by symbol reference) which enables dedup/merge of identical stubs downstream.

#### Additional notes

Each `mogg.generic.compiled.region` is outlined into a named mogg.kernel.stub
(`region_<n>`), and the original site becomes a `mogg.execute.compiled` call.
Sets the device from the original op, and forwards summary/async/constant
resource attributes. Uses `ABIKind::MGP` (void-style; multiple results via
opaque buffers) for the stub.

Source:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/lib/GraphCompiler/MOGGDialect/Transforms/OutlineKernels.cpp>
Pass definition:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/include/GraphCompiler/GraphCompiler/Conversion/MOGGToMGP.td>

### AddStackTracesToAssertsPass

#### Description

Augments `mgp.assert` messages with build-time Python stack traces derived from
operation locations.

#### Dialects

Consumes and produces `mgp` (ModelOp); uses `rmo` location metadata.

#### Additional notes

Source:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/lib/GraphCompiler/MGPDialect/Transforms/AttachPythonStackTraces.cpp>
Pass definition:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/include/GraphCompiler/GraphCompiler/MGPDialect/Transforms/MGPTransforms.td>

### VerifyModelArguments

#### Description

Inserts runtime shape checks for `mgp.model` execute arguments to ensure
provided tensors match statically known dimensions.

#### Dialects

Consumes and produces `mgp` (ModelOp), using `index` comparisons and `pop`
boolean ops for predicates.

#### Performance

Adds runtime checks on model entry. Overhead is proportional to the number of
static dimensions in the execute arguments.

#### Additional notes

The pass scans execute arguments, skips non-tensors and fully dynamic shapes,
extracts the runtime tensor spec, and builds per-dimension equality checks.
A device tensor should have at least one static dim and a known rank. For those,
the pass combines the checks with `pop.and` and emits an `mgp.assert`.

Source:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/lib/GraphCompiler/MGPDialect/Transforms/VerifyModelArgumentsPass.cpp>
Pass definition:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/include/GraphCompiler/GraphCompiler/MGPDialect/Transforms/MGPTransforms.td>

### PrepareDevicePrimitives

#### Description

Specializes device-generic `mgp` primitives by renaming ops to include the
target device label and rewriting device-ref attributes into plain device ids.

#### Dialects

Consumes and produces `mgp` (ModelOp), using device metadata (`#M.device_ref`)
from `m`.

#### Performance

Avoids runtime device-label reconstruction and enables GraphRT dispatch to use
device-specific primitive names directly.

#### Additional notes

Source:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/lib/GraphCompiler/MGPDialect/Transforms/PrepareDevicePrimitivesPass.cpp>
Pass definition:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/include/GraphCompiler/GraphCompiler/MGPDialect/Transforms/MGPTransforms.td>

### DeferAllocsPass

#### Description

Defers `mgp.buffer.alloc.*` operations until just before the unique mutating
op that writes the buffer, and wires the allocation to the same input chains as
that mutator.

#### Dialects

Consumes and produces `mgp` (ModelOp).

#### Performance

Reduces allocations live ranges and preserves ordering.

#### Additional notes

The pass walks both execute and init sections. For each alloc (ops implementing
`MGPBufferAllocOpInterface`) it finds the unique mutating op (either a real
mutating op or a `mgp.buffer.slice` (these can come from the dynamic planning
pass that runs just before; the pass started handling slices after the
introduction of the planning pass); only single uses are considered) that writes
to the buffer and moves the allocation immediately before the mutating op. The
mutator's input chains are inserted into the alloc so it can't execute too
early. Moves each alloc before its unique mutator and adds chain dependencies
via `alsoWaitForChains`, asserting that allocs accept input chains. Uses
`ExecuteState` and applies to both `init` and `execute` regions. Source:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/lib/GraphCompiler/MGPDialect/Transforms/DeferAllocsPass.cpp>
Pass definition:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/include/GraphCompiler/GraphCompiler/MGPDialect/Transforms/MGPTransforms.td>

### DynamicMemoryPlanning

#### Description

Performs runtime memory planning by grouping eligible buffer allocations and
replacing them with slices of a single pooled buffer.

#### Dialects

Consumes and produces `mgp` (ModelOp), using `mgp.buffer.plan` and
`mgp.buffer.slice`.

#### Performance

Reduces device memory footprint by reusing a pooled allocation while preserving
buffer lifetimes.

#### Additional notes

The goal of the pass is to build a single device-local memory pool (sized) for
eligible device allocations that would replace all individual `mgp.buffer.alloc`
with a slice into that pool, based on a dynamic lifetime plan.

For some context, the direction of the memory planning evolution somewhat
follows this design doc draft:
<https://www.notion.so/modularai/Universal-Static-Memory-Planning-1672fa070a0845deab513e1678a0d6db>
(the idea from the design doc is still compatible with the current
implementation, but it is only a pure mgp rewrite). Aside from additional
optimization opportunities, the fact that device allocation during execution
requires host invocation makes the transformation necessary.
The pass uses `BufferLifetimeAnalysis` to build a dependency tree and compute
pre/post indices for lifetime ranges. With those, it constructs a
`mgp.buffer.plan` op that has an implementation in MGPPrimitives library (a
single-pass greedy algorithm that tries to reuse non-overlapping memory blocks).
The op stores a precomputed NxN sharing matrix (`canShare[i][j] = 1` iff
allocations `i` and `j` can occupy the same slot) rather than raw pre/post
indices, making the reuse structure directly readable and canonical. The op
returns a high watermark value and a set of offsets that are consumed by
`mgp.buffer.slice`s, replacing original allocations. The high watermark value
is computed at runtime and is used for the allocation. The alignment is computed
as the least common multiple of all the allocations to be compatible with all
the slices.

Only non-data-dependent allocations (so the sizes are trivially computable) in
the execute block with uses and without untraceable aliases are considered.

Source:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/lib/GraphCompiler/MGPDialect/Transforms/DynamicMemoryPlanningPass.cpp>
Pass definition:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/include/GraphCompiler/GraphCompiler/MGPDialect/Transforms/MGPTransforms.td>

### SimpleBufferReuse

#### Description

Reuses existing `mgp.buffer.alloc` buffers when a later allocation has
compatible size, device, alignment, and strictly ordered uses.

#### Dialects

Consumes and produces `mgp` (ModelOp).

#### Performance

Reduces allocation count and peak memory usage in straightforward reuse cases.

#### Additional notes

This is the first, low‑risk buffer‑reuse pass in the MGP pipeline. It runs after
EICM and before DynamicMemoryPlanning, so it can reduce the number of allocs
(and their lifetimes) that the planner has to consider. The pass reuses one
buffer alloc for another if it can prove all uses of the current buffer execute
after all uses of the candidate buffer. The pass groups allocs by
`(runtimeSize, device, alignment)`, uses `BufferLifetimeAnalysis` dependency
trees to ensure all uses of one buffer execute after the other, and avoids
reuse when allocations share a use. Skips allocs inside
`mogg.generic.compiled.region`.

The pass marks BufferLifetimeAnalysis preserved though the IR is modified (so
the analysis is invalid). The expectation is that the next pass (dynamic memory
planning) checks the uses of the allocs, but not allocs themselves.

Source:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/lib/GraphCompiler/MGPDialect/Transforms/SimpleBufferReusePass.cpp>
Pass definition:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/include/GraphCompiler/GraphCompiler/MGPDialect/Transforms/MGPTransforms.td>

### SimplifyChainsPass

#### Description

Removes redundant inputs from `grt.chain.create` by dropping chains that are
already implied ready by other inputs.

#### Dialects

Consumes and produces `mgp`/`grt` (ModelOp).

#### Performance

Reduces chain fan-in and unnecessary dependencies.

#### Additional notes

Uses implied-ready queries over transitive chain dependencies (via a local
backward walk over the operand graph to answer "does chain A imply chain B?";
this is less precise than the former implementation through
`AsyncDataflowDependencies` but gives much better compilation times), updates
chain.create operands in parallel (if threading is enabled), then re-runs chain
canonicalization patterns.

Some limitations of the local analysis:

- no equivalence/aliasing reasoning, pure structural reachability over operands
- no transitive semantic rules beyond operands
- no set simplification via relation algebra
- no "always ready" knowledge for empty merges folding

Source:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/lib/GraphCompiler/MGPDialect/Transforms/SimplifyChainsPass.cpp>
Pass definition:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/include/GraphCompiler/GraphCompiler/MGPDialect/Transforms/MGPTransforms.td>

### SequenceFusion

#### Description

Fuses sequences of `mogg.generic.compiled.region` ops into larger regions based
on dataflow and fusion-compatibility rules.

#### Dialects

Consumes and produces `mogg`/`mgp` (ModelOp), relying on MOGG sequence-fusion
analyses.

#### Performance

Reduces kernel count and overhead by merging compatible compiled regions.

#### Additional notes

SequenceFusion is the final MGP‑level kernel fusion pass. It fuses sequences of
mogg.generic.compiled.region ops into a single compiled region. The pass runs
after LowerPrimitives, so it doesn't work with `mogg.kernel`. It reuses
`sourceDestinationSequenceFusionAnalysis` to build a subgraph of fusible ops.
Fusion is done in three phases:

1. Fuse producers that are allowed to be duplicated into multiple consumers.
2. Fuse when the producer is single-use.
3. (optional) Fuse ops that are only transitively related (no
   producer-consumer edges in the graph) as long as no cycles are created and
   no new dependencies between computationaly significant (cannot be fused)
   ops.
The fusion chains are restricted by a maximum of 16 kernels as a compilation
time improvement (see
<https://linear.app/modularml/issue/GEX-3054/epic-mitigate-compilation-time-regression>).

The pass relies on `ParametricTopologyAnalysis` for deterministic ordering
(preserved), and preserves region attributes like async flags, resources, and
fusion capability.

Source:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/lib/GraphCompiler/MOGGDialect/Transforms/SequenceFusion.cpp>
Pass definition:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/include/GraphCompiler/GraphCompiler/Conversion/MOGGToMGP.td>

### EmitMojo

#### Description

Emits Mojo source for `mogg.kernel.stub` and compiled regions, stores it as a
module resource, and wires models to load the emitted kernels at init time.

#### Dialects

Consumes `mogg`/`mgp` and depends on `mef`/`grt` (from the nested cleanup) and
`kgen`/`lit`/`pop` (from library analysis).

#### Performance

Decouples kernel compilation between GC and MOCO. There's optional caching for
kernels with deduplication logic to limit compile times.

#### Additional notes

EmitMojo is the last step of the MO to MGP pipeline before JIT compilation. It
strips device IDs and dynamic shape params for stable hashes, merges identical
stubs (those are created by the kernel outlining pass), then runs nested cleanup
(raise-back index constants, fixup overhangs, canonicalize, symbol DCE, add
try/except). Inserts `mogg.load.compiled.kernel` ops in model init and removes
kernel stubs after emitting the Mojo blob.
The last step builds a `EmitMojoState` (holds contexts, metadata; responsible
for symbol resolution via `MojoLibraryAnalysis`, constant resolution, type
conversion, and lifetime management), imports Mojo stdlib together with custom
kernel packages and walks kernel stubs.
For each `mgp.model`, the pass gathers all kernel symbols references by the
execute section, and adds a `mogg.load_compiled_kernel` op into init that points
to the Mojo blob. The op is later lowered to `mgp` buffer load while
JitCompileKernels provides the pointer to the correct resource.

Source:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/lib/GraphCompiler/MOGGDialect/Transforms/EmitMojo.cpp>
Pass definition:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/include/GraphCompiler/GraphCompiler/Conversion/MOGGToMGP.td>

### LowerMOGGRegionCompiles

#### Description

Lowers `mogg.load.compiled.kernel` to `mgp.module.load` and rewrites
`mogg.execute.compiled` into `mgp.execute.generic` using module/kernel indices.

#### Dialects

Consumes `mogg` and produces `mgp`/`grt` within a `ModuleOp`.

#### Performance

Moves compiled-kernel execution into the runtime-friendly MGP form with
explicit module loading and indexing.

#### Additional notes

This is the final lowering step that turns the EmitMojo/JIT artifacts into
runtime MGP ops. It runs immediately after JitCompileKernels. After the pass,
all the symbolic information (e.g., kernel names, blob resources) accessed by
`mogg` ops is finalized into `mgp` runtime ops with index-based access.
Tracks module/kernel indices per load, rewrites executes in both init and
execute blocks (finding the correct `!mgp.context`), replaces the module-load
placeholder chain, removes `mogg.to_mgp_from_opaque` in execute, and fixes
isolated-from-above bindings. Module loads are forced to CPU.

Source:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/lib/GraphCompiler/MOGGDialect/Transforms/LowerMOGGRegionCompile.cpp>
Pass definition:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/include/GraphCompiler/GraphCompiler/Conversion/MOGGToMGP.td>

### RemoveRedundantOperands

#### Description

Removes redundant inputs/results on `mogg.generic.compiled.region` ops after
sequence fusion, collapsing duplicate operands and unused chain/results.

#### Dialects

Consumes and produces `mogg`(ModelOp), depends on `mef` for chain types.

#### Performance

Reduces async value count and kernel signatures by eliminating unused or
duplicate operands/results.

Source:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/lib/GraphCompiler/MOGGDialect/Transforms/RemoveRedundantOperands.cpp>
Pass definition:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/include/GraphCompiler/GraphCompiler/Conversion/MOGGToMGP.td>

### GlobalizeSlotIndices

#### Description

Globalizes `mgp` context slot indices across all models in a module, ensuring
unique slot ranges for subgraph calls.

#### Dialects

Runs on `mgp` modules (`ModuleOp`), updating `mgp.context.create` and
`mgp.call` ops; `kgen` utilities.

#### Performance

Prevents slot conflicts when models are combined or inlined; minimal overhead.

#### Additional notes

When a module contains multiple mgp.model ops that call each other, each model
may define local slot indices (from its `mgp.context.create`). This pass assigns
global slot offsets so that nested model calls use globally unique slot indices.
Builds an MGP call graph (represents a model op where edges are calls between
models; small wrapper around `KGEN::SCCGraph`, so internally it computes SCCs
and schedules analysis/rewrite in post-order, bottom-up), computes per‑model
slot usage, and inserts per‑call slot offsets (index constants) while marking
calls as globalized.

Source:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/lib/GraphCompiler/MGPDialect/Transforms/GlobalizeSlotIndicesPass.cpp>
Pass definition:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/include/GraphCompiler/GraphCompiler/MGPDialect/Transforms/MGPTransforms.td>

### InlineMGPCalls

#### Description

Inlines `mgp.call` subgraphs into their caller models, cloning init/execute
regions and removing the call operations.

#### Dialects

Runs on `mgp` modules (`ModuleOp`) with `mgp.model` and `mgp.call`.

#### Performance

Eliminates call overhead and enables downstream optimization across model
boundaries; increases IR size.

#### Additional notes

Part of the JIT compilation pipeline, run early after LowerMOGGRegionCompiles.
Builds a call graph, rejects recursion, offsets runtime slot indices in cloned
ops, prefixes placeholder weight names, updates init output chains, and erases
inlined subgraph models when unused. Also updates `mgp.context.create` slot
counts after inlining.

Source:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/lib/GraphCompiler/MGPDialect/Transforms/InlineMGPCallsPass.cpp>
Pass definition:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/include/GraphCompiler/GraphCompiler/MGPDialect/Transforms/MGPTransforms.td>

### LowerAlignments

#### Description

Infers and propagates buffer alignment through `mgp` ops and writes concrete
alignment values into `mogg.alignment.placeholder` via `kgen.param.declare`.

#### Dialects

Consumes and produces `mgp` (ModelOp), with `mogg` alignment placeholders and
`kgen` parameter declarations.

#### Additional notes

The pass resolves `mogg.alignment.placeholder` ops into concrete
`kgen.param.declare` alignment values by inferring alignment through MGP
dataflow. This gives kernel generation a concrete alignment parameter for
memory accesses. Memory planning introduces slices; slices change alignment
(GCD with offset), so the pass must run before slices are introduced. Defaults
to device preferred alignment when unknown.

Source:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/lib/GraphCompiler/MGPDialect/Transforms/LowerAlignmentsPass.cpp>
Pass definition:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/include/GraphCompiler/GraphCompiler/MGPDialect/Transforms/MGPTransforms.td>

## RMO passes

### LegalizeRMOOps

#### Description

Lowers `rmo` ops into `mo` ops, primarily via the `MOAnalogue` interface (maps
`rmo` to `mo` 1 to 1) and rebinds that align operand/result types with their MO
counterparts. It also expands composite `rmo` ops into simpler `rmo`/`mo`
sequences when needed.

#### Dialects

Consumes `rmo` and produces `mo` (GraphOp). Depends on both `rmo` and `mo`
dialects.

#### Performance

Primarily a correctness/legalization step; enables downstream MO pipeline
optimizations by producing strongly typed MO ops early.

#### Additional notes

Implemented as a greedy rewrite pipeline. Includes specialized lowerings for
mutable ops, broadcast/concat/conv/pool/matmul/reshape/select, and rebinds.
`skip-rmo-rebinds` can be used to keep rebinds for debugging.

The pass infers MO types (type inference implementation:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/lib/GraphCompiler/RMODialect/RMOOps.cpp#L367-L873>,
MO-analogue type mapping:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/lib/GraphCompiler/RMODialect/RMOMOAnalogueOps.cpp>)
and rebinds MO ops outputs back to the original RMO result type. Ops that don't
have a direct mapping or require some additional logic are handled in two main
ways:

- Specialized legalization patterns, e.g., `rmo.slice`.
- Lower to other RMO ops first, then get legalized later in the same pass.

Legalization via the interface assumes all the output values are tensors, so
any operation like `rmo.mo.transfer` has to have a separate pattern. Device
handling is done via output annotation. After the pass, all ops are in `mo`
(TODO: verify).

Source:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/lib/GraphCompiler/RMODialect/Transforms/LegalizeRMOOps.cpp>
Pass definition:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/include/GraphCompiler/GraphCompiler/RMODialect/Transforms/RMOTransforms.td>

## MO passes

Pass definitions:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/include/GraphCompiler/GraphCompiler/MODialect/Transforms/MOTransforms.td>

### AddFallbackShapeFunctions

#### Description

Materializes fallback shape functions for `mo` ops that still have unresolved
output parameters, ensuring every op has a shape function or an explicit
no-fallback declaration.

#### Dialects

Consumes and produces `mo` (GraphOp), using `kgen` shape-parameter attrs and
Mojo library metadata for `mo.custom`.

#### Additional notes

AddFallbackShapeFunctions is the safety net for MO shape inference: it ensures
that every MO op that still has unresolved output parameters gets a shape
function call inserted so shape parameters are materialized before lowering.
Its job is to make sure there are no remaining "unmaterialized" output param
decls.

For each `mo` op that:

- has the output param decls empty, meaning, params are already materialized -
  nothing to do.
- has unmaterialized dims and has a fallback rule - dispatch to a fallback
  builder to create shape computation and insert it before the op. The op is
  left intact and the implementation is handled by kernel selection as usual.
- has unmaterialized dims and is a custom op - query `MojoLibraryAnalysis` to
  find a registered shape function and build and invocation using those kernel
  params.
- needs no fallback (e.g., matmul, static broadcast) - nothing to do.

TODO: check phasing issue
<https://linear.app/modularml/issue/GEX-886/graphcompiler-make-symbolic-optimization-not-phase-issue-with-constant>.

Source:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/lib/GraphCompiler/MODialect/Transforms/AddFallbackShapeFuncs.cpp>
Pass definition:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/include/GraphCompiler/GraphCompiler/MODialect/Transforms/MOTransforms.td>

### MergeDuplicateShapeMaterializations

#### Description

Deduplicates shape materialization ops so identical ops with the same operands
share a single defining instance and parameters.

#### Dialects

Consumes and produces `mo` (GraphOp), using `kgen` parameter metadata and some
`mogg` utilities.

#### Performance

Reduces redundant shape-parameter construction and improves CSE effectiveness
(semantically identical ops, but different parameters references).

#### Additional notes

Finds shape‑materialization ops that define all their parameters. If two such
ops have the same op name + same operands, it keeps one and deletes the rest
(remaps references using `ParameterCollector` analysis and building a def-use
graph).

Source:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/lib/GraphCompiler/MODialect/Transforms/MergeDuplicateShapeMaterializations.cpp>
Pass definition:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/include/GraphCompiler/GraphCompiler/MODialect/Transforms/MOTransforms.td>

### InvokeShapePackingFunctions

#### Description

Evaluates packing-related `mo.invoke_shape_func` calls using a C++ shape
re-implementation and materializes concrete parameter declarations.

#### Dialects

Consumes and produces `mo` (GraphOp), with `kgen` parameter ops for emitted
declarations.

#### Performance

Precomputes packing shapes at compile time when possible, avoiding runtime
shape computation.

#### Additional notes

Pattern fusion may produce packing kernels (e.g., matmul B packing). The inputs
for kernels with packing are constructed via a sequence of shape functions +
`mo.layout_transform`. When the target is known, some of those shape functions
for packing can be resolved statically. The pass targets specific ones like
matmul/conv filter packing (matched by op name) and emits `kgen.param.declare`
for inferred dims. Optionally inserts sanity-check assertions.

When the target attribute is not present in the graph, the pass initializes
llvm native target and pull the information from current host.

Another instance of shape propagation (with CSE and canonicalization) is run
right after the pass, so that the newly introduced declarations are pushed
through the graph.

Source:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/lib/GraphCompiler/MODialect/Transforms/InvokeShapePackingFunctions.cpp>
Pass definition:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/include/GraphCompiler/GraphCompiler/MODialect/Transforms/MOTransforms.td>

### ValidateDevices

#### Description

Validates that tensor dtypes in graph inputs/outputs are supported by the
target device architecture.

#### Dialects

Consumes and produces `mo` (GraphOp).

#### Additional notes

In preparation for MO to MOGG lowering, acts as correctness gate. Runs after
pattern fusion, so that all the potentially introduced tensors or device
annotation changes are settled.

The limitations are only put on CPU targets:

- no support for fp8.
- no support for fp16/bf16 on arm (for others, the types allowed for casting).
- no support for float types with size < 32bit on x86 and arch.

Source:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/lib/GraphCompiler/MODialect/Transforms/ValidateDevicesPass.cpp>
Pass definition:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/include/GraphCompiler/GraphCompiler/MODialect/Transforms/MOTransforms.td>

### DecomposeViewOps

#### Description

Breaks complex view ops into simpler view primitives to enable cleaner lowering
and fusion.

#### Dialects

Consumes and produces `mo` (GraphOp), `mosh` for shape ops.

#### Additional notes

Rewrites `mo.static.reshape` into merge/split/add‑singleton sequences when
possible, preserving layouts and rebinding types as needed.

This aids downstream passes (e.g., FuseReshapesIntoEpilogues) that don't handle
complex cases. `mo.slice` is left intact and handled directly by FuseViews
(contiguous slices) or the copy‑kernel path (non‑contiguous slices).

Source:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/lib/GraphCompiler/MODialect/Transforms/DecomposeViewOps.cpp>
Pass definition:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/include/GraphCompiler/GraphCompiler/MODialect/Transforms/MOTransforms.td>

### ConstantFold

#### Description

Identifies constant subgraphs and executes them at compile time using the
existing compilation pipeline, replacing them with `mo.constant` results.

#### Dialects

Runs on `mo` modules (`ModuleOp`) and drives MO to MGP lowering internally.

#### Performance

Reduces runtime work by precomputing constant regions; trades extra compile time
for smaller runtime graphs.

#### Additional notes

ConstantFold is graph‑level constant subgraph evaluation. It finds subgraphs
whose outputs are fully determined by constants, lowers and executes them, then
replaces their outputs with mo.constant values in the original graph.
The compilation consists of a nested MO->MGP pipeline with constant folding
disabled to avoid recursion and a MGP to MEF. All kernels are forced to CPU with
sync execution. The replacement converts the computed results back into
`mo.constant` (or constraints) and replaces the uses.

The pass is run each time when the pipeline introduces or exposes new constant
subgraphs.

Depends on `ParameterCollector` analysis.

Source:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/lib/GraphCompiler/MODialect/Transforms/ConstantFold.cpp>
Pass definition:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/include/GraphCompiler/GraphCompiler/MODialect/Transforms/MOTransforms.td>

### HoistParameterExpressions

#### Description

Hoists complex parameter expressions out of tensor types into explicit
`kgen.param.declare` aliases and rewrites types to reference the new params.

#### Dialects

Consumes and produces `mo` (GraphOp) with `kgen` parameter attrs/ops and `mosh`
dependency.

#### Additional notes

Takes parameter expressions embedded inside tensor shapes (e.g.,
`KGEN::ParamOperatorAttr` like `mul(D0, D1)`) and hoists them into named
parameters. This is done to avoid repeated/hidden expressions inside types and
to make later passes simpler. Introduces `mo.rebind` on inputs/outputs to
preserve original types, creates one alias per unique param expression per
region, and folds consecutive rebinds.

Source:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/lib/GraphCompiler/MODialect/Transforms/HoistParameterExpressionsPass.cpp>
Pass definition:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/include/GraphCompiler/GraphCompiler/MODialect/Transforms/MOTransforms.td>

### ElideRedundantMutableOps

#### Description

Eliminates redundant `mo.mutable.load` and `mo.mutable.store` ops when their
chains prove the operations are unnecessary.

#### Dialects

Consumes and produces `mo` (GraphOp), using chain semantics.

#### Performance

Removes unnecessary memory round-trips.

#### Additional notes

Elides a load when its input chain comes from a store to the same buffer and
the load is the sole chain user. Same for the stores.

The pass is only enabled together with `mogg` fusion (TODO: see notes).

Source:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/lib/GraphCompiler/MODialect/Transforms/ElideRedundantMutableOps.cpp>
Pass definition:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/include/GraphCompiler/GraphCompiler/MODialect/Transforms/MOTransforms.td>

### HoistConstantSubgraphsIntoParent

#### Description

Hoists callsite-invariant, constant-derived ops from subgraphs into their
parent graphs so they can be further hoisted to init or constant-folded.

#### Dialects

Runs on `mo` modules (`ModuleOp`) and uses `mo.call` call graphs.

#### Performance

Reduces repeated work inside subgraphs and unlocks EICM/constant folding of
invariant computations.

#### Additional notes

If a subgraph is called multiple times with the same SSA value for some
arguments, then any ops derived solely from those invariant args can be hoisted
to the callsite. The pass builds a call graph, identifies callsite-invariant
args and runs a dataflow analysis to find invariant-derived ops. The dataflow
analysis is a non-standard small implementation that depends on
`ParametricTopologyAnalysis` to reason about providers/users in the graph. In
operates on two-point lattice (const/non-const), and thus, is cheap and easy.

The optimization is performed when there are multiple callsitest. It ignores
cycles and treats "constant" as invariant across callsitets.

Source:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/lib/GraphCompiler/MODialect/Transforms/HoistConstantSubgraphsIntoParent.cpp>
Pass definition:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/include/GraphCompiler/GraphCompiler/MODialect/Transforms/MOTransforms.td>

### InferLayouts

#### Description

Infers tensor layouts in `mo.graph` using layout constraints from ops like
`mo.conv`, `mo.conv_transpose`, and `mo.matmul`, then annotates tensor types
with the inferred layout attributes.

#### Dialects

Consumes and produces `mo` (GraphOp).

#### Performance

Enables layout-aware lowering and prevents later layout mismatches; can reject
unsupported conv ranks early.

#### Additional notes

The pass walks the mo.graph and adds hard layout constraints (seed constraints)
where the op semantics require a particular layout (e.g., `mo.conv` with a rank
of 4 in NHWC format). The second stage propagates those constraints through ops
that keep the input/output layout identical (the layout is pushed through users
and defining op). The inferred layouts are then assigned to tensors
(`TensorType`) - used for kernel selection and affects view fusion. `mo.graph`
is updated to account for the changed signatures when necessary. Emits errors
on conflicting constraints.

Source:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/lib/GraphCompiler/MODialect/Transforms/InferLayouts.cpp>

### CanonicalizeChainSequences

#### Description

Optimizes linear (i.e. unique input/output) chains of side-effecting
`mo.mutable` ops by breaking chains that serialize unrelated buffers.
The pass rewrites sequences where independent buffer mutations are ordered by
the same chain, splitting the chain so later operations can start from an
earlier chain value when safe, while preserving ordering for dependent ops.

Implementation lives in `CanonicalizeBufferOpSequences` rewrite pattern and the
canonicalization patterns for `ChainCreateOp` (which is currently just
flattening).

#### Dialects

Consumes and produces `mo` (GraphOp) with chain semantics (`mo.chain.create`,
`mo.mutable.*` ops implementing `MOMutableOpInterface`).

#### Performance

Reduces unnecessary serialization (e.g., splitting unnecessary intermediate
chains) of side-effecting ops on different buffers, which can improve
parallelism and scheduling freedom.

#### Additional notes

Chains are introduced at the IR construction time (e.g., by the Graph API).
Implemented as a greedy rewrite over `GraphOp` with an increased iteration cap
for long chains; may skip ops that have multiple in/out chains. Source:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/lib/GraphCompiler/MODialect/Transforms/CanonicalizeChainSequences.cpp>

### PropagateShapes

#### Description

Propagates and simplifies shape-parameter logic through the graph. It evaluates
shape computations in the parameter domain, materializes shape functions from
parameter values, and simplifies parameter bindings via substitution and
constraint solving to rebind parameter attributes/types as it walks the graph.
It also normalizes cross-graph call operands to match callee shape symbols and
replaces `mo.shape_of` with constant shape values when possible.

At a high-level these are the steps taken:

1) Topologically sort ops with SSA + parameter dependencies.
2) Worklist walk: materialize shape functions, staticize data-dependent
   (requiring reading runtime values from input tensors) shape inputs, and
   fold parameter expressions/ranges.
3) Rebind/replace parameterized types and attrs, update graph signature and
   output bindings.
4) DCE unused `kgen.param.declare` via use-def graph.
5) Normalize `mo.shape_of` and cross-graph `mo.call` operands.

#### Dialects

Consumes and produces `mo` (GraphOp) and `kgen` parameter IR, with shape support
from `mosh` and `mo.shape` ops.

#### Performance

Reduces shape-related SSA and parameter dependencies, enabling more CSE and
canonicalization, and improving compile-time simplifications of shape logic.

#### Additional notes

Used in the shape inference pipeline and in the pattern-fusion path after
`InvokeShapePackingFunctions`. The pass relies on `ParameterCollector` analysis.
It performs a parametric topological sort (via `ParametricTopologyAnalysis`) so
the worklist visits ops only after their SSA and parameter dependencies are
available, which is required for consistent rebinding and folding. It also
builds a `KGEN::ParameterUseDefGraph` to drive dead `kgen.param.declare`
elimination, and uses `KGEN::ParameterEvaluator` for substitution/constraint
solving. Shape materialization depends on the `ShapeMaterialization` (lets ops
generate the IR needed to compute their output shape parameters) and
`Staticization` (convert tensors into symbolic parameter expressions) interfaces
to evaluate shape logic. Dynamic ranks are not supported. Staticizing only works
for tensors of rank less than 2. CSE for parameter expressions doesn't cross
region boundaries. The pass is the basis for shape inferencing. The rest is
canonicalization and CSE, plus `KGEN::VerifyParameters` (TODO). Source:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/lib/GraphCompiler/MODialect/Transforms/PropagateShapesPass.cpp>

### SymbolicOptimization

#### Description

Applies domain-specific optimizations using symbolic information in the graph,
typically produced by shape inference. The pass rewrites and canonicalizes
operation sequences to simpler or more efficient forms, and can perform
fast-math style rewrites under configurable assumptions.

#### Dialects

Consumes and produces `mo` (GraphOp) with symbolic shape data from `mosh` and
parameter expressions from `kgen`.

#### Performance

Reduces redundant computation and enables later fusion by simplifying symbolic
subgraphs and normalizing shapes/reshapes/transposes. Also converts operations
to forms that have better kernel implementations.

#### Additional notes

Runs twice in the pipeline:

1. After initial shape inference and constant folding, before adding fallback
   shape functions.
2. After adding fallback shape functions, to optimize newly materialized
   operations.

The pass is organized in stages that allow for shape-driven algebraic
simplification. There are "core" patterns that roughly include the following
groups of transformations:

- data-layout "swimming" (moving shape-changing ops through other ops to either
  facilitate reuse or enable fusion/canonicalization; e.g., sink transposes
  through unary/binary ops)
- Contraction ops normalization and simplification
- split/concat canonicalization, including gather/squeeze patterns and similar
- Algebraic simplification with simple expressions folding (e.g., x+0 -> x) and
  optional arith-unstable patterns under configuration flags
  (`symbolic-ffast-math` and `symbolic-no-undefined-math`).

The pass applies these core patterns multiple times intertwining with forward
and backward mathematical canonicalizations, together with splice <-> split
conversions. Those decompose and compose certain operations/sequences (e.g.,
rsqrt <-> 1 / sqrt) to try and give more context for canonicalization patterns
and core patterns to apply.

Some limitations/requirements:

- Requires symbolic shape information (known shapes)
- Some patterns are conservative (require single-use constraints)
- Fast-math optimizations may change floating-point semantics
- Top-down traversal may miss some optimization opportunities
- Reliance on the sequence of patterns application

Source:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/lib/GraphCompiler/MODialect/Transforms/SymbolicOptimization.cpp>
Pass definition:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/include/GraphCompiler/GraphCompiler/MODialect/Transforms/MOTransforms.td>

### PatternFusion

#### Description

Fuses specific operator patterns into hand-fused kernels or more specialized
ops. It applies a set of pattern rewrites (e.g., matmul/conv const folding,
layer norm and RMS norm fusions, concat/split axis-0 rewrites, and multi-head
attention fusion) to replace multi-op subgraphs with optimized implementations
or canonical forms.

#### Dialects

Consumes and produces `mo` (GraphOp) and some `mogg` ops, and depends on `kgen`,
`mosh`, and `mogg`. Technically, the pass lives in the `mogg` transforms,
however it is run before `mo` to `mogg` lowering and on `mo` ops.

#### Performance

Enables higher-level kernel fusion and reduces dispatch overhead by replacing
common subgraphs with hand-fused kernels or more efficient ops.

#### Additional notes

Implemented as a greedy rewrite over `GraphOp`, using `PatternFusionOptions` to
toggle MHA matching and device targeting. After rewrites, the pass calls
`MO::synchronizeMetadataUpdates` to keep metadata consistent with the
transformed graph.

PatternFusion patterns use the device-aware rewrite wrappers
(`MO::DeviceAwareRewritePattern` + `MO::DeviceAwareRewriter`) so each pattern
collects device annotations from all matched ops, rejects conflicting device
sets, and stamps the chosen device onto newly created fused ops. This lets the
pattern implementations perform device-specific checks and still emit
device-consistent replacements. There can (and are) copies of very similar
passes with device specialization. Device-aware rewriter definition:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/include/GraphCompiler/GraphCompiler/MODialect/MODevices.h#L98>.

Some patterns emit `mo.custom` calls that route into the MOGG Kernel API layer
(TODO: check the lowering paths and how they affect the fusion); other patterns
rewrite to `mo` ops or `mogg` ops like `mogg.concat` and `mogg.split`
(interestingly, there are kernels for both in the kernel API, but those are
registered for `mo` and are unused in case pattern fusion is on). The MOGG
Kernel API layer is a registry of hand-fused kernels exposed via
`@compiler.register(...)` symbols, which `mo.custom` uses as the dispatch key
at lowering time.
MOGG Kernel API definitions (docs-by-source):
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/max/kernels/src/graph_compiler/builtin_kernels/attention.mojo#L231>

Most complexity is in the matching of complex patterns such as MHA. The
matching has restrictions, e.g., single-use for intermediate values.

The pass is gated by device/layout inference & validation.

Source:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/lib/GraphCompiler/MODialect/Transforms/PatternFusion.cpp>

## MOGG passes

### JitCompileKernels

#### Description

Takes Mojo source text emitted by the compiler (stored as a module resource),
parses it, runs the KGEN pipelines (LIT -> KGEN -> elaborated KGEN -> LLVM), and
replaces the MOGG::LoadCompiledKernelOp resource with the compiled binary.

#### Dialects

Uses `lit` to parse Mojo sources, and depends on `m`, `mogg`, `kgen`, `pop`,
`hlcf`, debug info, as well as target dialects (`llvm`, `nvvm`, `rocdl`,
`index`).

#### Performance

Generates compiled kernel blobs and enables later lowering stages to reference
compiled code instead of high-level kernel bodies.

#### Additional notes

JitCompileKernels is the backbone of the JIT compilation pipeline (last stage
of gc-pipeline) that converts `mogg` kernel regions into executable code. The
rest of the pipeline runs on the surrounding IR that has references to the
compiled blobs. The pass looks up the source code in the Mojo resource blob,
parses Mojo source into MLIR module with LIT parser, runs KGEN pipelines
(generate library and the elaboration pipelines), lowers to llvm, optimizes
(-O3), and runs codegen (via ObjectCompiler). The binaries are then embedded
into the MEF as resources attached to the module load path.

Source:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/lib/GraphCompiler/MOGGDialect/Transforms/JitCompileKernels.cpp>
Pass definition:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/include/GraphCompiler/GraphCompiler/MOGGDialect/MOGGPasses.td>

### FuseViews

#### Description

Fuses view-only kernels into their consumer kernels, eliminating intermediate
view kernels and avoiding unnecessary copies. Reshapes are handled as a
separate phase: first fuse non-reshape views, then fuse reshape views, and
finally raise remaining reshape kernels back to `mo.static.reshape` so they can
lower to cheap primitives later.

#### Dialects

Consumes and produces `mogg` (GraphOp) and relies on `mo` view ops (notably
`mo.static.reshape`) embedded in kernel exec bodies.

#### Performance

Reduces kernel count and data movement by folding views into consumer kernels,
while preserving reshape correctness (contiguous strides).

#### Additional notes

There's no distinct type for a view op (e.g., broadcast, reshapes, etc.). At
`mogg` level they are identified by op semantics - a view op is recognized as a
`KernelOp` that reports `isViewOp()`. View ops are cheap when they remain views,
but if they materialize into separate kernels they can cause extra copies and
break fusion opportunities. So, the pass fuses view‑only kernels into their
consumer kernels so that view transforms are applied inside the consumer,
avoiding copies and redundant kernels.

Reshapes are special because of the requirement for contiguous strides
(TODO:reason? contiguous reshapes are effectively zero-cost). These are fused
last, as a separate second transformation step, so that sequences like
broadcast -> reshape stay valid. At the time, the new view-kernel API dropped
static stride info and the fix was to thread static strides into
`view_copy_impl` where possible. Reshape was kept contiguous-only, rather than
deriving strides from input.

Guarantees that remaining reshapes are raised to `mo.static.reshape` (converted
back to their original form) and that no view is fused into a reshape kernel.
Those remaining reshapes are then lowered by `MOGGToMGP`. The
implementation basically inserts a new tensor spec reusing the input buffer, so
there's no copy.

Source:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/lib/GraphCompiler/MOGGDialect/Transforms/FuseViews.cpp>
Pass definition:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/include/GraphCompiler/GraphCompiler/MOGGDialect/MOGGPasses.td>

### FuseReshapesIntoEpilogues

#### Description

Fuses reshape operations (`SplitDim`, `MergeDim`, `AddSingletonDim`) into the
output lambda of producer kernels. This avoids materializing intermediate
tensors when the reshape can be composed into the producer's index computation.

#### Dialects

Consumes and produces `mogg` (GraphOp) and uses `mo` reshape ops plus `kgen`
helpers for kernel rewriting.

#### Performance

Reduces reshape-kernel overhead and improves fusion opportunities by composing
reshapes directly into producer kernel output lambdas.

#### Additional notes

Operates only on non-elementwise kernels (those are handled by a separate pass)
and requires single-use outputs for output fusion. That is, the output of a
kernel is simple enough that we can absorb a reshape by rewriting the
producer's output lambda without changing observable semantics. The op has to
live in the same block as its user, so no cross-control-flow fusion.

Though the pass performs fusion at `mogg` level, it handles ops in `mo` (just
three, at the time of writing: `AddSingletonDimOp`, `SplitDimOp`, and
`MergeDimOp`) and expects `mo.graph` to have `mogg.kernel`s with the bound
output lambdas (aka lambda fusion interface) - the pass performs "hard" fusion.
It works by rewriting the kernel internals (output placeholder + lambdas)
within the graph, fusing reshape ops into a producer kernel. Remaining ops are
handled by `MOGGToMGP` pass which create a new tensor spec and reuses
existing buffer.

Source:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/lib/GraphCompiler/MOGGDialect/Transforms/FuseReshapesIntoEpilogues.cpp>
Pass definition:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/include/GraphCompiler/GraphCompiler/MOGGDialect/MOGGPasses.td>

### FuseElementwise

#### Description

Fuses chains of elementwise kernels by inlining producer lambdas into consumer
kernels when the dataflow and chain dependencies permit it.

#### Dialects

Consumes and produces `mogg` (GraphOp), and relies on `mo` elementwise ops plus
`kgen` and `mogg` analysis helpers.

#### Performance

Reduces kernel launch overhead and enables more aggressive fusion by collapsing
elementwise pipelines into single kernels.

#### Additional notes

Elementwise kernels are identified as `mogg.kernel` whose terminator is a
`mogg.call.foreach`. The pass builds fusion subgraphs - sets of ops for fusion.
The fusion sets are built via `sourceDestinationSequenceFusionAnalysis`
(implementation:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/lib/GraphCompiler/MOGGDialect/Analysis/SequenceFusionAnalysis.cpp#L448>),
which fuses eligible ops downward when all uses are compatible (kernels have to
be mutable, chains are compatible, destination operands are fusible) and
optionally allows duplication within a bounded fusion set. It walks each block,
builds a topological order of eligible ops (there's a hardcoded limit on the
number of ops = 64), then forms source->destination groups where every use can
be fused; never fuses across blocks.

For each of the identified fusion subgraph, the pass fuses (hard) everything
into the last kernel by inlining the source elementwise lambda into the
destination lambda (plus updating effects and other cleanup like kernel
parameters, block arguments, etc).

Uses sequence-fusion analysis, respects mutable chain ordering, and requires
topological ordering for safe rewrites. Controlled by `dump-dot-graph`.

Source:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/lib/GraphCompiler/MOGGDialect/Transforms/FuseElementwiseKernels.cpp>
Pass definition:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/include/GraphCompiler/GraphCompiler/MOGGDialect/MOGGPasses.td>

### FusePrologues

#### Description

Fuses elementwise producer kernels directly into the input lambdas of their
consumers (prologue fusion), reducing kernel boundaries.

#### Dialects

Consumes and produces `mogg` (GraphOp), with elementwise lambdas and `mo`
kernel names used to gate specific fusion cases.

#### Performance

Reduces kernel launch overhead and improves locality by merging elementwise
prologue work into compute kernels.

#### Additional notes

An elementwise kernel is fused as a prologue if its output is the sole input
to a non-elementwise kernel that has an input lambda for that operand.

Source:
`GraphCompiler/lib/GraphCompiler/MOGGDialect/Transforms/FusePrologues.cpp`
Pass definition:
`GraphCompiler/include/GraphCompiler/GraphCompiler/MOGGDialect/MOGGPasses.td`

### FuseEpilogues

#### Description

Fuses elementwise consumer kernels directly into the output lambdas of their
producers (epilogue fusion), reducing kernel boundaries. Runs interleaved with
affine view fusion until convergence, enabling multi-hop fusion across affine
views (e.g., `matmul -> add -> merge_dim -> add` fuses into a single kernel).

#### Dialects

Consumes and produces `mogg` (GraphOp), with elementwise lambdas and `mo`
kernel names used to gate specific fusion cases.

#### Performance

Reduces kernel launch overhead and improves locality by merging elementwise
epilogue work into compute kernels.

#### Additional notes

There's no strict written-down fusion contract, but the implementation implies
the following set:

- output lambda compatibility,
- single-use output and the same block (producer's result is consumed in the
  same block),
- no dependency cycles,
- chain compatibility for mutable kernels.

Some operations like matmul/conv/concat are treated specially during epilogue
fusion and are matched by op name.

Some cases require additional care:

- If the elementwise epilogue contains a `mo.mutable_store` and shares any
  `mo.buffer` operand with the producer, fusion is blocked to avoid
  side-effects/aliasing hazards.
- Device-compatibility checks are relaxed for distributed ops (e.g.,
  `MO::AllGatherOp`). The kernel itself executes on CPU even though the
  tensors are GPU-resident.
- Even with single-use checks, fusion can introduce cycles when there are
  extra operands/chains. The pass explicitly checks that no other operand
  of the epilogue depends on the producer.

The pass internally loops over epilogue fusion and reshape-into-epilogue fusion
(shared with `FuseReshapesIntoEpilogues`) until no more fusions are possible.

Source:
`GraphCompiler/lib/GraphCompiler/MOGGDialect/Transforms/FuseEpilogues.cpp`
Pass definition:
`GraphCompiler/include/GraphCompiler/GraphCompiler/MOGGDialect/MOGGPasses.td`

### FuseMutableLoads

#### Description

Fuses `mo.mutable.load` into a consuming `mogg.kernel` by replacing the load's
value-semantic tensor with the underlying buffer, threading the load's chains
through the kernel, and updating kernel operands/results accordingly.

#### Dialects

Consumes and produces `mo`/`mogg` (GraphOp).

#### Performance

Reduces overhead from standalone mutable loads and improves buffer-level fusion
opportunities by converting tensor uses to direct buffers.

#### Additional notes

Only fuses when the load's tensor has a single use in the same block and the
consumer is a `mogg.kernel`. For multi-use loads, it first clones the load per
use (parallelization) unless a consumer kernel writes to the same buffer.
If the kernel depends on the load's out-chain, the pass can rewrite an
exclusive path of `mo.chain.create` ops to detach the dependency and avoid
cycles (fusion can create cycles if the kernel depends on the load's output
chain).

Parallelization is blocked in the kernel has write effects and shares the same
buffer with the load.

Source:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/lib/GraphCompiler/MOGGDialect/Transforms/FuseMutableLoads.cpp>
Pass definition:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/include/GraphCompiler/GraphCompiler/MOGGDialect/MOGGPasses.td>

### SimplifyKernelOperands

#### Description

Deduplicates identical operands on `mogg.kernel` and its bound lambdas, updating
block arguments to share a single operand instance.

#### Dialects

Consumes and produces `mogg` (GraphOp).

#### Performance

Reduces operand/argument counts and shrinks kernel signatures, improving IR
cleanliness and reducing downstream work.

#### Additional notes

This is a deduplication pass. It eliminates duplicate operands on both
`mogg.kernel` and each `mogg.bind` (lambda) inside the kernel. Since every
fusion pass in the pipeline (at `mogg`) can introduce duplicate operands or
lambda arguments (fusion passes splice operand lists when they inline) it is
run in between passes in the fusion "subpipeline". So, the idea is to keep
fusion passes simple at the cost of running simplification multiple times.
Walks kernel inputs and bind lambda operands in reverse order, replacing uses
of duplicate block arguments and erasing redundant operands/arguments.

Downstream passes assume operand uniqueness.

Source:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/lib/GraphCompiler/MOGGDialect/Transforms/SimplifyKernelOperands.cpp>
Pass definition:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/include/GraphCompiler/GraphCompiler/MOGGDialect/MOGGPasses.td>

### HashModuleContents

#### Description

Computes a semantic hash of module contents and attaches it as an attribute on
KGEN generators, enabling cache keys and change tracking across kernel modules.

#### Dialects

Operates on `mogg`/`kgen` modules (`mlir::ModuleOp`) with symbol references.

#### Performance

Adds hashing overhead during compilation but enables downstream cache reuse and
deduplication based on semantic equivalence.

#### Additional notes

Not a part of the default pipeline. No usage currently. Builds a callgraph that
tracks symbol references in attributes/types, hashes generator bytecode with
normalized names/locations, and writes `semantic_hash` attributes. In that sense
it is "semantic", i.e., insensitive to non-semantic differences like
names/location. Uses `SymbolTableAnalysis` and `ModuleHasher`. Source:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/lib/GraphCompiler/MOGGDialect/Transforms/HashModuleContents.cpp>
Implementation:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/lib/GraphCompiler/MOGGDialect/Support/ModuleHasher.cpp>
Pass definition:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/include/GraphCompiler/GraphCompiler/MOGGDialect/MOGGPasses.td>

### NormalizeAllNames

#### Description

Normalizes symbol and parameter names in modules to deterministic, synthetic
names to improve deduplication and stability across compiles.

#### Performance

Adds light renaming overhead but increases cache hits and reduces spurious
differences in generated kernels.

#### Additional notes

The implementation is used in the framework frontend as a cache-key
normalization step (not the pass itself). Renames symbols to `__S<index>`, then
walks operations in parallel to rename parameters to `__P<index>` and update all
symbol/parameter references. Source:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/lib/GraphCompiler/MOGGDialect/Transforms/NormalizeAllNames.cpp>
Implementation:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/KGEN/lib/KGENDialect/KGENUtils.cpp>
Pass definition:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/include/GraphCompiler/GraphCompiler/MOGGDialect/MOGGPasses.td>

### FuseConstants

#### Description

Fuses small constant inputs into kernel bodies, eliminating the need to pass
them as runtime operands.

#### Dialects

Consumes and produces `mogg` and `mo` (GraphOp).

#### Performance

Reduces kernel argument traffic and can enable additional kernel fusion by
embedding constants directly into the kernel.

#### Additional notes

This is a tiny pass that walks mogg.kernel ops and calls a helper to inline
small constant (all supported `mogg` types and small dense arrays) inputs into
the kernel body (cloning `mo.constant`). The fusion only works for host kernels

- an artifact of nvptx bringup (see
<https://github.com/modularml/modular/issues/22959>). Unfused constants go
through EICM and land into init if possible (TODO: check). Fused ones are added
to the constant resources attribute during kernel-stub lowering and are
materialized as `MGP::BufferConstantOp`

Source:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/lib/GraphCompiler/MOGGDialect/Transforms/FuseConstants.cpp>,
but the logic is in
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/lib/GraphCompiler/MOGGDialect/Support/FusionUtils.cpp#L135>
Pass definition:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/include/GraphCompiler/GraphCompiler/MOGGDialect/MOGGPasses.td>

## KGEN passes

### VerifyParameters

#### Description

Validates KGEN parameter declarations and uses, detecting mismatches,
conflicting declarations, and parameter dependency cycles. Optionally performs
trivial parameter propagation and rebinding after verification.

#### Dialects

Runs on `kgen` modules and depends on `kgen` parameter IR and the LIT symbol
table evaluation context for parameter resolution.

#### Additional notes

Builds a `ParameterUseDefGraph` per declaration region and verifies it against a
`LIT::LITSymTabEvaluationContext`, using a shared `ParameterCollector` cache and
`SymbolTableAnalysis`. If `simplify` is enabled, it propagates trivial parameter
definitions, rebinds attributes/types/locations, removes `kgen.param.declare`
ops, and peels redundant `kgen.rebind` ops.

Source:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/KGEN/lib/Transforms/VerifyParameters.cpp>
Pass definition:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/KGEN/include/KGEN/KGENPasses.td>
