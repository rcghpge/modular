# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# UNSUPPORTED: windows
# RUN: mojo %s | FileCheck %s

import _mlir

from max.graph import _c


fn test_empty_graph() raises:
    with _mlir.Context() as ctx:
        ctx.load_modular_dialects()
        ctx.load_all_available_dialects()
        let unknown = _mlir.Location.unknown(ctx)
        var m = _mlir.Module(unknown)

        let signature = _mlir.builtin_types.FunctionType(ctx)

        _ = _c.graph_new(m, unknown, "empty_graph", signature)

        # CHECK: builtin.module
        # CHECK: mo.graph
        # CHECK: sym_name = "empty_graph"
        print(m)


fn test_identity_graph() raises:
    with _mlir.Context() as ctx:
        ctx.load_modular_dialects()
        ctx.load_all_available_dialects()
        let unknown = _mlir.Location.unknown(ctx)
        var m = _mlir.Module(unknown)

        let intScalar = _mlir.Type.parse(ctx, "!mo.scalar<si32>")

        var ins = DynamicVector[_mlir.Type]()
        ins.append(intScalar)
        var outs = DynamicVector[_mlir.Type]()
        outs.append(intScalar)

        let signature = _mlir.builtin_types.FunctionType(ins, outs)
        let g = _c.graph_new(m, unknown, "identity_graph", signature)

        let block = g.region(0).first_block()
        let arg0 = block.argument(0)
        var operands = DynamicVector[_mlir.Value]()
        operands.append(arg0)

        block.append(_mlir.Operation("mo.output", unknown, operands=operands))

        # CHECK: mo.graph @identity_graph
        # CHECK: mo.output
        print(m)


def main():
    test_empty_graph()
    test_identity_graph()
