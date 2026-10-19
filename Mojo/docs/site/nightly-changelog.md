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

## Library stabilizations

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

## GPU programming

## Tooling changes

## Removed

## Fixed
