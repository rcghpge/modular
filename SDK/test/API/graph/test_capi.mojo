# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# UNSUPPORTED: windows
# RUN: mojo %s | FileCheck %s

import mlir
from max.graph import capi


fn test_empty_graph() raises:
    with mlir.Context() as ctx:
        ctx.load_modular_dialects()
        ctx.load_all_available_dialects()
        let unknown = mlir.Location.unknown(ctx)
        var m = mlir.Module(unknown)

        let signature = mlir.builtin_types.FunctionType(ctx)

        _ = capi.graph_new(m, unknown, "empty_graph", signature)

        # CHECK: builtin.module
        # CHECK: mo.graph
        # CHECK: sym_name = "empty_graph"
        print(m)


fn test_identity_graph() raises:
    with mlir.Context() as ctx:
        ctx.load_modular_dialects()
        ctx.load_all_available_dialects()
        let unknown = mlir.Location.unknown(ctx)
        var m = mlir.Module(unknown)

        let intScalar = mlir.Type.parse(ctx, "!mo.scalar<si32>")

        var ins = DynamicVector[mlir.Type]()
        ins.append(intScalar)
        var outs = DynamicVector[mlir.Type]()
        outs.append(intScalar)

        # var outs = capi.arity_new()
        # capi.arity_append_type(outs, intScalar)

        let signature = mlir.builtin_types.FunctionType(ins, outs)
        let g = capi.graph_new(m, unknown, "identity_graph", signature)

        let block = g.region(0).first_block()
        let arg0 = block.argument(0)
        var operands = DynamicVector[mlir.Value]()
        operands.append(arg0)

        block.append(mlir.Operation("mo.output", unknown, operands=operands))

        # CHECK: mo.graph @identity_graph
        # CHECK: mo.output
        print(m)


def main():
    test_empty_graph()
    test_identity_graph()
