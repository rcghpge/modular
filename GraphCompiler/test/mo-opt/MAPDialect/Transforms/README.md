# MAP dialect transform tests

Tests that validate the `map` dialect passes and transformations, such as
`mo-to-map` (lowering MO ops into `map.iter` / `imap` kernels) and
`map-fuse-kernels` (fusing producers into consumers).

Add `.mlir` files here; they are picked up automatically by the `**/*.mlir`
glob in `GraphCompiler/test/mo-opt/BUILD.bazel`. The usual pattern is:

```text
// RUN: mo-opt %s --map-fuse-kernels | FileCheck %s
```

Both passes are currently no-ops gated on the `MAX_GC_USE_ADV_FUSION`
environment variable, so there is no behavior to check yet. Populate this
directory as the passes gain functionality.
