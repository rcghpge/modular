# Mojo unreleased changelog

This is a list of UNRELEASED changes for the Mojo language and tools.

When we cut a release, these notes move to `changelog-released.md` and that's
what we publish.

[//]: # Here's the template to use when starting a new batch of notes:
[//]: ## UNRELEASED
[//]: ### ✨ Highlights
[//]: ### Language enhancements
[//]: ### Language changes
[//]: ### Library changes
[//]: ### Tooling changes
[//]: ### ❌ Removed
[//]: ### 🛠️ Fixed

## UNRELEASED

### ✨ Highlights

### Language enhancements {#25-7-language-enhancements}

- [Issue #3925](https://github.com/modular/modular/issues/3925): Mojo now allows
  methods to be overloaded based on "owned" vs "by-ref" argument conventions,
  selecting the owned overload when given an owned value, and selecting the
  by-ref version otherwise.  This allows somewhat more efficient algorithms,
  e.g. consuming vs borrowing iterators:

  ```mojo
  struct MyCollection:
    fn __iter__(var self) -> Self.ConsumingIterator: ...
    fn __iter__(self) -> Self.BorrowingIterator: ...
  ```

- Literals now have a default type. For example, you can now bind `[1,2,3]` to
  `T` in a call to a function defined as `fn zip[T: Iterable](impl:T)` because
  it will default to the standard library's List type.

- Mojo now has a `__functions_in_module` experimental intrinsic that allows
  reflection over the functions declared in the module where it is called. For
  example:

  ```mojo
  fn foo(): pass

  def bar(x: Int): pass

  def main():
    alias funcs = __functions_in_module()
    # equivalent to:
    alias same_funcs = Tuple(foo, bar)
  ```

  The intrinsic is currently limited for use from within `main`.

- The `@implicit` decorator now accepts an optional `deprecated` keyword
  argument. This can be used to phase out implicit conversions instead of just
  removing the decorator (which can result in another, unintended implicit
  conversion path). For example, the compiler now warns about the following:

  ```mojo
  struct MyStuff:
    @implicit(deprecated=True)
    fn __init__(out self, value: Int):
      pass

  fn deprecated_implicit_conversion():
    # warning: deprecated implicit conversion from 'IntLiteral[1]' to 'MyStuff'
    _: MyStuff = 1

    _ = MyStuff(1)  # this is okay, because the conversion is already explicit.
  ```

- The `@deprecated` decorator can now take a target symbol with the `use` keyword
  argument. This is mutually exclusive with the existing positional string
  argument. A deprecation warning will be automatically generated.

  ```mojo
  @deprecated(use=new)
  fn old():
    pass

  fn new():
    pass

  fn main():
    old() # 'old' is deprecated, use 'new' instead
  ```

### Language changes {#25-7-language-changes}

- Expressions like `(Int, Float)` is no longer a syntax sugar for
  `Tuple[Int, Float]`. It instead creates a tuple instance of two type values,
  i.e., `(Int, Float) : Tuple[__typeof(Int), __typeof(Float)]`.

- The `__type_of` magic function has been been renamed to `type_of`. Using the
  old spelling will yield a deprecation warning. Similarly, `__origin_of` has
  been deprecated in favor of the new `origin_of`.

### Library changes {#25-7-library-changes}

#### Libraries

- Added `os.isatty()` function to check whether a file descriptor refers to a
  terminal. This function accepts an `Int` file descriptor. If you have a
  `FileDescriptor` object, use its `isatty()` method instead.

- The `Hasher` trait's `_update_with_bytes` method now takes `Span[Byte]`
  instead of `UnsafePointer[UInt8]` and a separate length parameter. This
  change applies to all hasher implementations including `AHasher` and `Fnv1a`.

- Added `unsafe_get`, `unsafe_swap_elements` and `unsafe_subspan` to `Span`.

- The deprecated `DType.index` is now removed in favor of the `DType.int`.

- `math.isqrt` has been renamed to `rsqrt` since it performs reciprocal square
  root functionality.

- Added `swap_pointees` function to `UnsafePointer` as an alternative to `swap`
  when the pointers may potentially alias each other.

- `Span` and `StringSlice` constructors now accept `Int` for length parameters
  instead of `UInt`. This change makes these types more ergonomic to use with
  integer literals and other `Int`-based APIs.

- `memcpy` and `parallel_memcpy` without keyword arguments are deprecated.

- The `math` package now has a mojo native implementation of `acos`, `asin`,
  `cbrt`, and `erfc`.

- Added support for NVIDIA GeForce GTX 970.

- Added support for NVIDIA Jetson Thor.

- Added support for NVIDIA DGX Spark.

- `Optional` now conforms to `Iterable` and `Iterator` acting as a collection of
  size 1 or 0.

- `origin_cast` for `LayoutTensor`, `NDBuffer` and `UnsafePointer` has been
  deprecated and removed. `LayoutTensor` and `NDBuffer` now supports a safer
  `as_any_origin()` origin casting. `UnsafePointer` has the same
  safe alternative and in addition, it has an additional safe `as_immutable`
  casting function and explicitly unsafe `unsafe_mut_cast` and
  `unsafe_origin_cast` casting function.

- Implicit conversions between `Int` and `UInt` are now deprecated.

  The `@implicit` decorator on `Int.__init__(UInt)` and `UInt.__init__(Int)`
  will be removed in a future version of Mojo. Code that currently performs
  implicit conversions between `Int` and `UInt` will issue a deprecation warning,
  and should be updated to explicitly read `Int(uint_val)` or `UInt(int_val)`
  respectively.

- `assert_equal` now displays colored character-by-character diffs when string
  comparisons fail, making it easier to spot differences. Differing characters
  are highlighted in red for the left string and green for the right string.

- Added `sys.compile.SanitizeAddress` providing a way for mojo code to detect
  `--sanitize address` at compile time.

- Error messages now preserve symbolic calls to `always_inline("builtin")`
  functions rather than inlining them into the error message.

- `SIMD` now implements the `DivModable` trait.

- Mojo now uses system allocators in programs built with `mojo build --sanitize address`.
  This means asan can see mojo heap allocations and should now be able to
  detect many more heap memory errors.

- `TestSuite` now can generate test reports with `.generate_report()`. Also
  a `TestReport` and `TestSuiteReport` structs were added.

- `TestSuite` now allows explicitly skipping registered tests using the
  `TestSuite.skip` API.

- `TestSuite` now allows basic control from CLI arguments. Tests can be skipped
  from the CLI by passing test function names after a `--skip` flag, e.g.

  ```console
  mojo run test_my_stuff.mojo --skip test_currently_failing test_also_failing
  ```

  Similarly, the `--only` flag enables the specification of an allowlist, e.g.
  the following will skip any other registered test cases:

  ```console
  mojo run test_my_stuff.mojo --only test_only_this test_this_as_well
  ```

  The `--skip-all` flag will skip all registered test cases in the suite. Note
  that `--only` respects skipped tests, i.e. it does not run tests that are
  skipped using `TestSuite.skip`.

- `Codepoint` now conforms to `Comparable` adding `__le__`, `__lt__`, `__ge__`,
  and `__gt__` implementations.

- Several standard library APIs have been updated to use `Int` instead of `UInt`
  for improved ergonomics, eliminating the need for explicit casts when using
  `Int` values (the default type for integer literals and loop indices):
  - `BitSet[size: Int]` - Changed parameter from `UInt` to `Int`
  - `BitSet.set(idx: Int)`, `BitSet.clear(idx: Int)`, `BitSet.toggle(idx: Int)`,
    `BitSet.test(idx: Int)` - Changed from `UInt` to `Int`
  - `String(unsafe_uninit_length: Int)` - Changed from `UInt` to `Int`
  - `String.capacity() -> Int` - Changed return type from `UInt` to `Int`
  - `String.reserve(new_capacity: Int)` - Changed from `UInt` to `Int`
  - `List(length: Int, fill: T)` - Changed from `UInt` to `Int`
  - `Codepoint.unsafe_write_utf8() -> Int` - Changed return type from `UInt` to `Int`
  - `Codepoint.utf8_byte_length() -> Int` - Changed return type from `UInt` to `Int`

- Added `repeat()` function to the `itertools` module that creates an iterator
  which repeats an element a specified number of times. Unlike Python's
  `itertools.repeat()`, infinite iteration is not currently supported - the
  `times` parameter is required. Example usage:

  ```mojo
  from itertools import repeat

  for val in repeat(42, times=3):
      print(val)  # Prints: 42, 42, 42
  ```

- `gpu.sync.syncwarp()` now supports Apple GPUs via `SIMDGROUP` barrier
  implementation. On Apple GPUs, this provides execution synchronization for
  all active lanes using a `SIMDGROUP` barrier with no memory fence. For
  threadgroup memory ordering, use `barrier()` instead. Note that lane masks
  are not supported on Apple GPUs, so the mask argument is ignored.

- The `gpu` package has been reorganized into logical subdirectories for better
  code organization:
  - `gpu/primitives/` - Low-level GPU execution primitives (warp, block,
    cluster, id, grid_controls)
  - `gpu/memory/` - Memory operations (async_copy, TMA, address spaces)
  - `gpu/sync/` - Synchronization primitives (barriers, semaphores)
  - `gpu/compute/` - Compute operations (mma, tensor cores, tcgen05)

  **Backward compatibility**: All existing imports continue to work unchanged.
  Deprecated import paths (`gpu.id`, `gpu.mma`, `gpu.cluster`,
  `gpu.grid_controls`, `gpu.warp`, `gpu.semaphore`, `gpu.mma_sm100`,
  `gpu.tcgen05`, `gpu.mma_util`, `gpu.mma_operand_descriptor`,
  `gpu.tensor_ops`) are preserved as re-export wrappers with deprecation
  notices. Users can migrate to the new recommended import patterns at their
  own pace:

  ```mojo
  # Old (deprecated but still works):
  from gpu.id import block_idx, thread_idx
  from gpu.mma import mma
  from gpu.mma_sm100 import UMMAKind
  from gpu.tcgen05 import tcgen05_alloc
  from gpu.semaphore import Semaphore
  from gpu.cluster import cluster_sync

  # New (recommended):
  from gpu import block_idx, thread_idx, cluster_sync
  from gpu.compute.mma import mma
  from gpu.compute.mma_sm100 import UMMAKind
  from gpu.compute.tcgen05 import tcgen05_alloc
  from gpu.sync.semaphore import Semaphore
  ```

- The `_GPUAddressSpace` type has been removed and consolidated into
  `AddressSpace`. GPU-specific address space constants (GLOBAL, SHARED,
  CONSTANT, LOCAL, SHARED_CLUSTER) are now available as aliases on the unified
  `AddressSpace` type. The `GPUAddressSpace` alias has also been removed in
  favor of using `AddressSpace` directly. Since `AddressSpace` is part of the
  prelude, it no longer needs to be explicitly imported in most code.

- TMA (Tensor Memory Accelerator) types have been moved to a dedicated module.
  The types `TMADescriptor`, `TensorMapSwizzle`, `TensorMapDataType`,
  `TensorMapInterleave`, `TensorMapL2Promotion`, `TensorMapFloatOOBFill`, and
  functions `create_tma_descriptor` and `prefetch_tma_descriptor` are now
  available from `gpu.host.nvidia.tma` instead of `gpu.host._nvidia_cuda`.

- The `empty` origin has been renamed to `external`.

### Tooling changes {#25-7-tooling-changes}

- `mojo test` has [been deprecated](https://forum.modular.com/t/proposal-deprecating-mojo-test/2371)
  and will be removed in a future release.

- Elaboration error now reports the full call instantiation failure path.
  For this mojo file:

  ```mojo
  fn fn1[T: ImplicitlyCopyable, //] (a: T):
      constrained[False]()
  fn fn2[T: ImplicitlyCopyable, //] (a: T):
      return fn1(a)
  fn main():
      fn2(1)
  ```

  now the error prints the path of `main -> fn2 -> fn1 -> constrained[False]`
  instead of just `constrained[False]`.

- Elaboration error now prints out trivial parameter values with call expansion failures.
 For this simple mojo program:

  ```mojo
  fn fn1[a: Int, b: Int]():
      constrained[a < b]()

  fn fn2[a: Int, b: Int]():
      fn1[a, b]()

  fn main():
      fn2[4, 2]()
  ```

  now the error message shows `parameter value(s): ("a": 4, "b": 2)`.
  Only string value and numerical values are printed out by default now,
   other values are shown as `...`.
  use `--elaboration-error-verbose` to show all parameter values.

  ```mojo
  test.mojo:6:14: note: call expansion failed with parameter value(s): ("a": 4, "b": 2)
      fn1[a, b]()

  ```

- `--elaboration-error-limit` option is added to `mojo run` and `mojo build`.
  This option sets a limit to number of elaboration errors that get printed.
  The default value is 20.
  This limit can be changed by `--elaboration-error-limit=n` to `n` where
   `0` means unlimited.

- `--help-hidden` option is added to all mojo tools to show hidden options.

### ❌ Removed {#25-7-removed}

- `LayoutTensorBuild` type has been removed.  Use `LayoutTensor` with parameters
  directly instead.

- Elaboration error message prelude is removed by default.
 Preludes are call expansion locations in `stdlib/builtin/_startup.mojo`
  which persists in all call expansion path but rarely is where error happens.
   Remove these to de-clutter elaboration errors.
 Use `--elaboration-error-include-prelude` to include prelude.
  - By default (without prelude)

    ```mojo
      test.mojo:43:4: error: function instantiation failed
      fn main():
         ^
      test.mojo:45:12: note: call expansion failed
          my_func()
      ...
    ```

  - with prelude

    ```mojo
    open-source/max/mojo/stdlib/stdlib/builtin/_startup.mojo:119:4: error: function instantiation failed
    fn __mojo_main_prototype(
       ^
    open-source/max/mojo/stdlib/stdlib/builtin/_startup.mojo:119:4: note: call expansion failed with parameter value(s): (...)
    open-source/max/mojo/stdlib/stdlib/builtin/_startup.mojo:42:4: note: function instantiation failed
    fn __wrap_and_execute_main[
       ^
    open-source/max/mojo/stdlib/stdlib/builtin/_startup.mojo:68:14: note: call expansion failed
        main_func()
                 ^
    test.mojo:43:4: note: function instantiation failed
    fn main():
       ^
    test:45:12: note: call expansion failed
        my_func()
    ...
    ```

### 🛠️ Fixed {#25-7-fixed}

- The `math.cos` and `math.sin` function can now be evaluated at compile time
  (fixes #5111).

- Fixed `IntTuple.value(i)` method returning incorrect values when elements are
  stored as nested single-element tuples. Previously, calling
  `Layout.row_major(M, N).stride.value(i)` would return negative offset values
  (e.g., -65536, -65537) instead of the actual stride values. This affected any
  code that accessed layout stride or shape values using the `value()` method.

- Fixed `LayoutTensor.shape[idx]()` method returning incorrect values for nested
  layouts. The bug occurred when accessing shape dimensions of tensors with
  nested layouts like `((32, 2), (32, 4))`, where the method would return
  garbage values instead of the correct product (e.g., 64).

  - Fixed `LayoutTensor` element-wise arithmetic operations (`+`, `-`, `*`, `/`)
  between tensors with different memory layouts. Previously, operations like
  `a.transpose() - b` would produce incorrect results when the operands had
  different layouts, because the same layout index was incorrectly used for both
  operands. This now correctly computes separate indices for each tensor based
  on its layout.

- Fixed `LayoutTensor.shape[idx]()` method returning incorrect values for nested
  layouts. The bug occurred when accessing shape dimensions of tensors with
  nested layouts like `((32, 2), (32, 4))`, where the method would return
  garbage values instead of the correct product (e.g., 64).

- Fixed `arange()` function in `layout._fillers` to properly handle nested
  layout structures. Previously, the function would fail when filling
  tensors with nested layouts like
  `Layout(IntTuple(IntTuple(16, 8), IntTuple(32, 2)), ...)` because it
  attempted to extract shape values from nested tuples incorrectly.

- Fixed [PR5479](https://github.com/modular/modular/issues/5479): mojo crashes
  when compiling standalone `__del__` function without struct context.
