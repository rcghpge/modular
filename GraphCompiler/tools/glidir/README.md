# GlidIR

GlidIR is an interactive MLIR visualization tool built with ImGui and OpenGL.

## Features

- Side-by-side IR text viewer and tree explorer
- Bookmarks: double-click an operation in the IR to pin it for quick navigation
- Regex search with next/previous navigation
- Tree view shows ops, regions, blocks, types, attributes, operands, and results
- Inline stats on each tree node (operand/result counts, block/op counts)
- Font size adjustment with Cmd/Ctrl +/-

## Usage

```text
glidir path/to/file.mlir
```
