---
title: MAX nightly
---

This version is still a work in progress.

## Highlights

## Documentation

## MAX models

- GLM-5.1, GLM-5.2, and GLM-5.3 now default `clear_thinking` to `False`, which
  keeps the `reasoning_content` a client sends back for earlier turns in the
  prompt. GLM-5.3 previously defaulted it to `True`, and GLM-5.1's and
  GLM-5.2's templates clear earlier reasoning when it is unset. This matches
  Z.ai's Coding Plan endpoint, and
  [Z.ai's thinking-mode guide](https://docs.z.ai/guides/capabilities/thinking-mode)
  recommends it for coding and agent workloads. Pass `clear_thinking=True` in
  the chat template options to drop earlier reasoning.

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

- Added `max.driver.float32_to_bfloat16_as_uint16()`,
  `max.driver.bfloat16_as_uint16_to_float32()`, and
  `max.driver.float32_array_to_buffer()` for moving bfloat16 data between
  NumPy and a `Buffer`. `max.pipelines.lib` still re-exports
  `float32_to_bfloat16_as_uint16()` and `float32_array_to_buffer()`.

### C API

## Kernels and GPU programming

## Breaking changes

- The deprecated `LayoutTensor` and `LayoutTensorIter` types have been removed
  from the `layout` package, along with the `to_layout_tensor()` and
  `lt_to_tt()` conversions. Use `TileTensor` and `TileTensorIter` instead.
  `ThreadScope` now lives in `layout.tile_io`.

- Removed the `max.pipelines.lib.bfloat16_utils` module. Import its functions
  from `max.driver` instead.

- Removed `GraphPipelineModel`, `ModuleV3PipelineModel` and
  `MultiGraphPipelineModelWithKVCache` from `max.pipelines.lib`. Subclass
  `PipelineModel`, return the module from `graph_specs()` under
  `GraphRole.ENCODER`, and load it with `load_graphs()` instead.

## Fixes

## Mojo language

For all the updates to the Mojo language, standard library, and tools,
see the [Mojo release notes](https://mojolang.org/releases/).
