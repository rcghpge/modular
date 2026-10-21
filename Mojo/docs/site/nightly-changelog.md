---
title: Mojo nightly
---

This version is still a work in progress.

## Highlights

## Documentation

## Language enhancements

- `with` statements now no longer copy when passed a reference to a context
  manager like in `with foo: pass`.

- `__exit__(self, error: E)` now additionally supports `var self` and
  `deinit self`, similar to the non-error `__exit__`.

## Language changes

- Precompiled packages now no longer leak names and provide the same surface as
  the same packages compiled from source. This may break your code (especially
  from `std`):

  ```mojo
  from std.os import rmdir

  def main():
    std.os.rmdir(...) # This no longer works - use rmdir, or import std.os.
  ```

## Library stabilizations

- `List.insert`
- `List.clear`

## Library performance improvements

## Library changes

- `Tuple` element types no longer need to be `Movable`. A `Tuple` is now
  `Movable` only when all of its elements are.
  `Tuple` has a new `init_with` initializer, which constructs each element in
  place from a closure parameterized on the element index:

  ```mojo
  comptime T = Tuple[Pinned, Pinned]  # `Pinned` is not `Movable`.
  var t = T(init_with=lambda [i: Int]() -> T.Ts[i]: Pinned(i * 10))
  ```

- `Layout` now names its element count `length`, matching `Span` and `List`.
  Construct one with `Layout[T](length=n)` and read the count back with
  `len(layout)`: `Layout` and `ManagedAllocation` now conform to `Sized`. The
  `Layout(count=)` initializer and the `Layout.count()` method are deprecated.

  ```mojo
  var allocation = alloc(Layout[Int32](length=4))
  for i in range(len(allocation)):
      allocation.unsafe_ptr().unsafe_offset(i).unsafe_write(Int32(i))
  dealloc(allocation^)
  ```

- Added `FileDescriptor` to `std.io`, an owning wrapper around
  `RawFileDescriptor`. It is a linear type: close it explicitly with
  `fd^.close()`, which raises if the close fails. It is movable but not
  copyable, and `Optional[FileDescriptor]` is the size of a single
  descriptor. `RawFileDescriptor` gains a raising `close()` method.

## GPU programming

- Added the `apple-m6` and `apple-m6-metal4` accelerator targets, and
  `CompilationTarget.is_apple_m6()` to `std.sys`. Apple M6 GPUs are now
  detected automatically, so you no longer need to pass
  `--target-accelerator=apple-m5` on M6 hardware.

## Tooling changes

## Removed

- Legacy closures are removed. The `capturing` function effect and the
  `@__parameter` (and `@parameter`), `@__copy_capture`, and
  `@__move_capture` decorators are now errors. Declare a nested function's
  captures in a capture list instead, and pass the closure as a value:

  ```mojo
  def apply[F: def(Int) -> Int](f: F, x: Int) -> Int:
      return f(x)

  def main():
      var offset = 10
      def add(x: Int) {var offset} -> Int:  # was @__copy_capture(offset)
          return x + offset
      print(apply(add, 1))
  ```

  A function passed as a compile-time parameter must now be `thin`, for
  example `[f: def(Int) thin -> Int]`.

- `CompiledFunctionInfo` no longer has `num_captures`, `capture_sizes`, or
  `populate` members. Kernels can no longer capture implicitly, so these
  were always empty.

## Fixed

- `SIMD.slice()` no longer passes `llvm.vector.extract` an offset that is not
  a multiple of the output width, which LLVM rejects. A wide slice at such an
  offset, like `slice[16, offset=8]()` on a 32-element vector, now takes the
  element-wise path, and aligned slices narrower than the target's native
  vector width, such as the halves `split()` returns for small vectors, now
  use the intrinsic.
