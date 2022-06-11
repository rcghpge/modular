# GenericML

[TOC]

## Introduction

This module contains high-quality helper classes and routines for building
machine learning frameworks - but implemented in a framework-independent way.
These routines help provide common currency types in the runtime, and aid
implementation of glue code.

These libraries are framework independent, so they should not be very
opinionated, and they should not depend on any specific framework modules.

## GenericML/Support

This library provides common types for reasoning about Tensors with
low-dependencies, suitable for use by both compiler and runtime clients.  This
is allowed to depend on LLVM Support, the Modular Support library, and LLCL.  It
is not allowed to depend on any specific framework's libraries (e.g. TFLite,
Faux, PyTorch etc).  Further, this cannot depend on MLIR because it is intended
to be usable at runtime.

## GenericML/Kernels

This library provides "production quality" kernels that are shippable as part of
our product, which are built to be largely framework independent.

This is meant to be well considered and actively curated, not a playground with
lots of experimental toys (for that, use the Faux framework) and not specific
to any one framework (put these inside the framework).

## GenericML/tools/model

The model tool provides a framework-agnostic way for working with models.
It is meant to be a wrapper around existing tools within the modular code base
(e.g. TFLC, StockTFLite etc). It also provides a way to download models, perform
competitive analysis, and more. Usage of the tool is meant to be as simple as
possible. Usage information is available in the [Model documentation](Model.md)
and developer information is available in the [Model developer documentation](ModelDeveloperGuide.md).
