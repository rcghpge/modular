# MLRT: Modular Low-level Runtime

## Introduction

The Modular Low-level Runtime (MLRT) is a collection of low-level C++ runtime
components which service many of the high-level Max and Mojo components within
the Modular stack.

MLRT contains the following libraries:

- AsyncRT - Parallel C++ runtime with threadpools
- Core - Core C++ runtime utilities
- GraphRT - Graph runtime infrastructure

## [AsyncRT](./AsyncRT/README.md)

AsyncRT contains low level support for domain independent parallel CPU
computation, including a thread pool implementation.

## Core

Core contains C++ runtime utilities, including IO, Tensor and MojoValue.

### IO

Provides direct I/O capabilities between storage and GPU memory using NVIDIA's
GPUDirect Storage (GDS) technology. This allows GPUs to read and write data
directly from NVMe SSDs without passing through the CPU or system memory,
reducing I/O latency for data-intensive GPU workloads.

Note: this library is experimental and not yet integrated into the rest of the
codebase.

### Tensor

Provides tensor and buffer management utilities for ML workloads, including
memory-efficient buffer references with async lifetime tracking and stateful
execution contexts for model inference.

### MojoValue

Provides a C++ wrapper for runtime Mojo objects, enabling safe transfer of Mojo
values between the C++ and Mojo worlds. MojoValue manages ownership and lifetime
of Mojo objects in C++ code, automatically invoking the appropriate Mojo
destructor when the value is destroyed.

## [GraphRT](./GraphRT/README.md)

Namespace: `M::`. Depends on AsyncRT and MLIR (but not on Kernels or KGEN)

GraphRT is a low-level execution layer for precompiled dataflow graphs.
"Primitives" are registered as C/C++ functions and looked up by name, allowing
the bottom level of the runtime stack to drive arbitrary CPU computation and
interface with existing foreign runtimes.

## WIP Status

The `MLRT` directory will also contain the `Async RT` and `GraphRT` directories,
this change is WIP and is described in
<https://www.notion.so/modularai/Runtime-Folder-Restructuring-2a11044d37bb80bfb665d92acb950201>.
