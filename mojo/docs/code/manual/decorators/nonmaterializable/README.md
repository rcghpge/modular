This directory contains code examples for the
[@nonmaterializable decorator](../../../../../docs/manual/decorators/nonmaterializable.md)
reference page.

Contents:

- Each `.mojo` file is a standalone Mojo application.
- The `BUILD.bazel` file defines:
  - A `mojo_binary` target for each `.mojo` file (using the file name without
    extension).
  - A `modular_run_binary_test` target for each binary (with a `_test` suffix).
