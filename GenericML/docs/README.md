# GenericML

## Introduction

This module contains high-quality helper classes and routines for building
machine learning frameworks - but implemented in a framework-independent way.
These routines help provide common currency types in the runtime, and aid
implementation of glue code.

These libraries are framework independent, so they should not be very
opinionated, and they should not depend on any specific framework modules.

## GenericML/Support

This library provides common types for reasoning about Tensors with
low-dependencies, suitable for use by both compiler and runtime clients. This is
allowed to depend on LLVM Support, the Modular Support library, GraphRT, and
AsyncRT. It is not allowed to depend on any specific framework's libraries (for
example TFLite, Faux, PyTorch etc). Further, this cannot depend on MLIR because
it is intended to be usable at runtime.
