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
LLCL. It is not allowed to depend on any specific framework's libraries (for
example TFLite, Faux, PyTorch etc). Further, this cannot depend on MLIR because
it is intended to be usable at runtime.

## GenericML/Kernels

This library provides "production quality" kernels that are shippable as part of
our product, which are built to be largely framework independent.

This is meant to be well considered and actively curated, not a playground with
lots of experimental toys (for that, use the Faux framework) and not specific to
any one framework (put these inside the framework).

## GenericML/tools/model

The model tool provides a framework-agnostic way for working with models. It is
meant to be a wrapper around existing tools within the modular code base (for
example TFL, StockTFLite etc). It also provides a way to download models, perform
competitive analysis, and more. Usage of the tool is meant to be as simple as
possible. Usage and installation information is available in the [Model
documentation](Model.md).

The `model` tool is designed to be:

- **User Friendly**: The model tool designed to be user-friendly and easy to use.
The model tool will download the model for you, convert it, and plot the results.

- **Single source of truth**: When someone mentions benchmarking
`TFLite/MobileNet-V2` then that means that they ran the model tool against the
model specified in `Model/MobileNet-V2.yaml` and not some random model file.

- **Usable across teams**: Some teams might be interested in benchmarking models,
others might be interested in comparing accuracy, etc. These are different uses
cases that different teams are interested in. The model tool should contain
enough tools to service the different usages across teams as well as provide the
results in a digestable way.

- **Model, framework, and platform agnostic**: The same model command should be
able to run different frameworks models across different hardware.
