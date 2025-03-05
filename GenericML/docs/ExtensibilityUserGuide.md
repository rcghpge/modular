# Kernel Extensibility API User Guide

This document covers the Extensibility APIs for two main types of users: kernel
writers who want to plug their kernels into the graph compiler and graph-api
users who want to invoke the resulting functions.

## Registering a New Kernel

An example can be found
[here](https://github.com/modularml/modular/blob/adacf509639cd7925f9d702fec652d6844e4ac16/Kernels/mojo/Mogg/MOGGKernelAPI.mojo#L3724).

```python
@compiler.register("mo.bottom_k", num_dps_outputs=2)
struct BottomK:
    @staticmethod
    fn execute[
        type: DType,
        rank: Int,
    ](
        values: ManagedTensorSlice[type=type, rank=rank],
        indices: ManagedTensorSlice[type = DType.int64, rank=rank],
        input: ManagedTensorSlice[type=type, rank=rank],
        k: Scalar,
        axis: Scalar,
        sorted: Bool,
    ):
        top_k(
            managed_tensor_slice_to_ndbuffer(input),
            k,
            axis,
            False,
            managed_tensor_slice_to_ndbuffer(values),
            managed_tensor_slice_to_ndbuffer(indices),
            sorted,
        )

    @staticmethod
    fn shape(
        input: ManagedTensorSlice,
        k: Scalar,
        axis: Scalar,
        sorted: Bool,
    ) raises -> IndexList[input.rank]:
        return top_k_shape_impl[single_thread_blocking_override=True](
            managed_tensor_slice_to_ndbuffer(input),
            k,
            axis,
        )
```

We go over the most important components and what they mean:

### Registering a Function

The key is one operation is organized under a singular struct:

```python
@compiler.register("mo.bottom_k", num_dps_outputs=2)
struct BottomK:
```

This indicates the resulting methods in the struct correspond to the
mo-operation “mo.bottom_k”. If we instead we did
`@compiler.register("my_cool_op")` , then this would would correspond to call
to a custom op with symbol “my_cool_op.”

### Required Methods

Right now we have two supported methods (plus one special thing for PT):

```python
fn execute():
fn shape() raises -> IndexList[input.rank]
```

Execute implements the kernel body, while shape is used by the compiler to
calculate the dynamic shape of outputs.

Note **shape** is only needed if the compiler cannot otherwise figure out the
parametric shapes.

E.g. for the graph-api folks if at the end of the compilation pipeline for MO
you have:

```python
mo.foo<() -> A, B, C>(...) ... -> !mo.tensor<[A, B, C], f32>
```

This needs a shape() implementation to run in order to define “A”, “B”, and
“C”. Note:

```python
mo.foo(...) ... -> !mo.tensor<[A, B, C], f32>
```

does not require a shape function as in this notation, “A”, “B”, and “C” are
defined elsewhere.

### **Operand Types Supported**

- We support **ManagedTensorSlice** as a first class citizen.
- Variadic tensors can be represented via
**StaticTuple[ManagedTensorSlice[type, rank], size]**.
  - We don’t support other variadic types besides Tensor
- Dynamic lists of tensors are supported via
**List[ManagedTensorSlice[type, rank]]** though support is
limited at this time.
  - We don’t support lists of other types besides Tensor

Some special operand structs exist which should go at the end of the operand
list:

- `DeviceContextPtr`
  - If specified, the graph compiler will give you access to the device
    context, which is needed for GPU support at the moment
  - If you don’t need this, you don’t have to include this
- `Scalar`
  - This is just syntax sugar over **ManagedTensorSlice’s** of one value
  - At the moment, this just dereferences the associated pointer, other
    implementations are open in the future (e.g. dont make a whole tensor for
    one value).
  - The tensor must be on host

**Any other structs will be done via opaque-type support** which should be as
fully featured as the old API. We support both register and memory-only types.

**Example:**

```python
@compiler.register("myOp")
struct MyOp:
    @staticmethod
    fn execute(
      # Varidic output support.
      # It is assumed the size of the tuple can be inferred from the call site
        output: StaticTuple[ManagedTensorSlice[type, rank]],

        # First class citizen
        input: ManagedTensorSlice[output.type, output.rank],

        # Handled via opaque type support
        cache: MyCacheStruct,

        # Syntax sugar over index_tensor[0]
        index: Scalar,

        # Transparently added by runtime to the call as needed
        # (Optional if you don't need it)
        ctx: DeviceContextPtr,
    ) raises:
     ...
```

### **Parameters Supported**

- We assume all included parameters can be inferred from the input arguments.
- If a parameter cannot be inferred from the input arguments solely, there are
many workarounds so please contact us.
- We do not support parameterized opaque types at the moment (both in the API
and in our stack).

We have a few “special parameters” the compiler always knows how to handle:

- **target: StringLiteral** which might be “cpu” or “gpu” and can be used for
specializing code for devices. This corresponds to the **device annotated on
the associated MO op.** `mo.add(...) {device = **gpu:0**}`
- **synchronous: Bool** if the graph compiler chooses to run this kernel
synchronously, this will be true. The heuristic meaning behind this being true
is “there is not a lot of computation.” This is often used to dispatch to a
simpler implementation of the kernel (e.g. not parallelized).
- These special parameters are optional if not needed.

Any additional parameters are specially handled by the graph compiler.

### **Compile-Time Tensor Information**

In order to access static information about tensors (e.g. shape) from the graph
compiler we use the `static_spec` parameter of the `ManagedTensorSlice`.
The `ManagedTensorSlice` provides aliases for the commonly accessed compile-time
information of each tensor.

```python
fn execute[
    target: StringLiteral
](
    output: ManagedTensorSlice,
    input: ManagedTensorSlice[output.type, output.rank],
    ctx: DeviceContextPtr,
) raises:
    alias static_shape = input._static_shape
    alias static_strides = input._static_strides
    alias alignment = input.alignment
    alias exclusive = input.exclusive
```

The `static_spec` parameter has the type `StaticTensorSpec[type, rank]`. In
addition to the information above, the `StaticTensorSpec` also contains the
fusion lambdas for input/output fusion, which should not be used directly.

In general, the methods on `ManagedTensorSlice` should be preferred when
possible. For shape and strides, `ManagedTensorSlice::shape` and
`ManagedTensorSlice::strides` will do the correct thing. The `_static_shape` and
`_static_strides` aliases should only be used when in a parameter context.

The `_fused_load` and `_fused_store` methods should be preferred over directly
accessing the fusion lambdas when loading or storing to a tensor.

### DPS Operands

```python
@compiler.register("mo.bottom_k", num_dps_outputs=2)
struct BottomK:
    @staticmethod
    fn execute[
        type: DType,
        rank: Int,
    ](
      # An "output"
        values: ManagedTensorSlice[type=type, rank=rank],
        # An "output"
        indices: ManagedTensorSlice[type = DType.int64, rank=rank],

        # Inputs below!
        input: ManagedTensorSlice[type=type, rank=rank],
        k: Scalar,
        axis: Scalar,
        sorted: Bool,
    ):
```

The number of
[destination-passing-style](https://simon.peytonjones.org/assets/pdfs/destination-passing-style.pdf)
(DPS) outputs are specified by `num_dps_outputs=2` . The first operands of
`execute` correspond to these DPS outputs.

By default `num_dps_outputs=1` if it is not included.

We can set `num_dps_outputs=0` if needed:

```python
@compiler.register("blah", **num_dps_outputs=0**)
struct Blah:
    @always_inline
    @staticmethod
    fn execute[
        type: DType, target: StringLiteral
    ](
      **# These are all "inputs"**
        blocks: ManagedTensorSlice[type, 6],
        cache_lengths: ManagedTensorSlice[DType.uint32, 1],
        lookup_table: ManagedTensorSlice[DType.uint32, 1],
        max_lengths: ManagedTensorSlice[DType.uint32, 2],
    ):
     # If this returns nothing, in the graph api we want chains to
     # reorder side-effects likely
     pass
```

### **Value Outputs**

We do not need to return things in a DPS style. We can do regular returns too
and things should work as you expect:

```python
@compiler.register(
    "continuous_batching_kv_cache_collection_h8_d128_bshd", num_dps_outputs=0
)
struct Struct_continuous_batching_kv_cache_collection_h8_d128_bshd:
    @always_inline
    @staticmethod
    fn execute[
        type: DType, target: StringLiteral
    ](
        blocks: ManagedTensorSlice[type, 6],
        cache_lengths: ManagedTensorSlice[DType.uint32, 1],
        lookup_table: ManagedTensorSlice[DType.uint32, 1],
        max_lengths: ManagedTensorSlice[DType.uint32, 2],
    **) -> ContinuousBatchingKVCacheCollection[
        type,
        kv_params_h8_d128_bshd,
    ]:**
        return generic_get_continuous_cache_kernel_api[
            kv_params=kv_params_h8_d128_bshd
        ](blocks, cache_lengths, lookup_table, max_lengths)

@compiler.register("mo.transpose")
@compiler.view_kernel
struct Transpose:
    @always_inline
    @staticmethod
    fn transpose_in_place(
        input: ManagedTensorSlice,
        permutations: ManagedTensorSlice[rank=1],
    ) -> **ManagedTensorSlice[type = input.type, rank = input.rank]:**
        var new_shape = IndexList[input.rank]()
        var new_stride = IndexList[input.rank]()

        @parameter
        for i in range(input.rank):
            var dim = Int(permutations[i])
            new_shape[i] = input.dim_size(dim)
            new_stride[i] = input.stride_length(dim)

        return ManagedTensorSlice[type = input.type, rank = input.rank](
            input._ptr, new_shape, new_stride
        )
```

Note, one thing we are lacking is the ability to create **ManagedTensorSlice**
which point to **newly allocated memory**. **You can do it, but it won’t be
properly freed at this time and leak** (probably, i haven’t tried). Support for
this will come later if needed.

Note that you usually want `num_dps_outputs=0` for these types of kernels. If
you want a mixture of DPS and value return outputs, this is supported though is
not as well tested.

### Special Decorators

There are some special decorators like `@compiler.view_kernel` and
`@compiler.elementwise` which are used to control optimizations by the graph
compiler. These are undocumented for now, though if you think you need one of
these, let us know.

### Fusion

You can access the input/output lambdas via `specsof`.

### Converting Between Tensor Types

Plenty of examples can be found in invoking this
[method](https://github.com/modularml/modular/blob/adacf509639cd7925f9d702fec652d6844e4ac16/Kernels/mojo/Mogg/MOGGKernelAPI.mojo#L527).

## Calling a New Kernel in the Graph API

### Mapping MO Calls To The Registered Kernel

The new API kernels can be called via custom ops. E.g.

```python
@compiler.register("MyOp")
struct MyOp:
 pass
```

Can be called via `mo.custom {symbol = "MyOp"}` .

The remaining useful knowledge is how to properly invoke this custom op to
match with the registered kernel. Here an example is prudent:

```python
# 3 SSA outputs, 3 SSA Inputs
%dps0, %dps1, %out_value = mo.custom {symbol = "MyOp"}(%in1, %in2, %in3) : ... -> ...
```

This maps to:

```python
@compiler.register("MyOp", num_dps_outputs=2)
struct MyOp:
    @staticmethod
    fn execute(
     dps0: ...,
     dps1: ...,
     in1: ...,
     in2: ...,
     in3: ...
   ) -> OutputTy:
    ...
   var out_value = ...
   return out_value
```

AKA: the dps outputs come “first”, followed by the return by value outputs.

To see support for variadic returns (of which we only support
ManagedTensorSlice): see “mo.split” and “mo.concat” implementations.

### Ordering And Side Effects

Chains should be used in the graph api level to order side-effects in places
where existing data dependencies are not sufficient (e.g. an op with no
“outputs” or it actually mutates something).

These do not correspond to any concept in the registered kernel but should be
properly respected by the rest of the graph compiler.

In terms of mapping them to the kernel registration, just pretend they don’t
exist. Chains should go at the end of the operand/result lists in the graph api
however.

E.g. in our previous example we can just do:

```python
%chain_in: !mo.chain
%chain_out: !mo.chain
%dps0, %dps1, %out_value, **%chain_out** =
 mo.custom {symbol = "MyOp"}(%in1, %in2, %in3, **%chain_in**) : ... -> ...
```

To enforce an ordering of execution via chains.
