---
title: MAX nightly
---

This version is still a work in progress.

## Highlights

## Documentation

## MAX models

## MAX framework

### Inference server

### Server metrics

### `max` CLI

### Python API

- Added the `ops.CoordinateTransformMode` and `ops.RoundMode` enums.
  `ops.resize_linear`, `ops.resize_nearest`, and their `max.experimental`
  functional counterparts take them for `coordinate_transform_mode` and
  `round_mode` in place of the integer codes 0-3. The integer codes still
  work.

### C API

## Kernels and GPU programming

## Breaking changes

## Fixes

## Mojo language

For all the updates to the Mojo language, standard library, and tools,
see the [Mojo release notes](https://mojolang.org/releases/).
