//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MODIALECT_MOTRANSFORMS_TD
#define GENERICML_GRAPHCOMPILER_MODIALECT_MOTRANSFORMS_TD

include "mlir/Pass/PassBase.td"

def ShapeInferenceMaterialization :
    Pass<"materialize-shape-funcs", "ModuleOp"> {
  let summary = "Materializes the shape inference functions.";
  let description = [{
    This pass materializes shape inference functions for parameter-declaring ops
    that have them, and add invocations to the graph, so that the parameters
    will be declared and bound to the results of shape inference function
    invocations, instead of declared by the ops.

    The ops need to be fully parametric (i.e. unknown shapes or dimensions in
    the arguments and results are not allowed), and the shape inference
    functions are not evaluated.
  }];
}

def InlineShapeFunctions : Pass<"inline-shape-funcs", "ModuleOp"> {
  let summary = "Inlines shape functions into the graph.";
  let description = [{
    This pass inlines (already materialized) shape functions invocations into
    the graph as parameter computations.
  }];
}

def PropagateShapes : Pass<"propagate-shapes", "GraphOp"> {
  let summary = "Propagates shape parameters through the graph.";
  let description = [{
    This pass evaluates shape parameter logic in the graph, simplifying and
    reducing shape logic as much as it can.
  }];
}

def ResolveUnknownParameters : Pass<"resolve-unknown-parameters", "ModuleOp"> {
  let summary = "Resolves unknown tensor shape and dimension parameters.";
  let description = [{
    This pass parameterizes `mo` ops that can be parameterized. This involves 2
    things:
    1. replacing unknown shape and dimension parameters (i.e. `?`) in `!mo.tensor`
    instances with parameter references.
    2. letting newly parameterized ops declare their new parameters that appear
    in their results.
  }];
}

def StaticizeShapes : Pass<"staticize-shapes", "ModuleOp"> {
  let summary = "Lower shape-like tensors to `!mosh.ape`.";
  let description = [{
    This pass attempts to lower shape-like MO tensors (i.e. rank-1, signed 32 or
    64 bit integer) to the special `!mosh.ape` type, and lower computations on
    them to parameter computations or static operations (as opposed to
    operations that take in dynamic values, like reshape or broadcast). This
    essentially "lifts" the shape logic encoded as tensor computations into the
    symbolic shape inference realm.

    The pass currently only resolves shapes that originate from `mo.shape_of`
    ops or constant shape-like tensors. In particular shape-like (or dim-like)
    arguments provided to the graph are not supported for now.
  }];
}

def DeparameterizeMoOps : Pass<"deparameterize-mo-ops", "GraphOp"> {
  let summary = "Lower parameter declarations in MO ops into rebinds.";
  let description = [{
    This pass lowers all parameter declarations in ops that could not be
    resolved using shape inference materialization. The pass fails if any of the
    ops don't support deparameterization and are still declaring parameters.
  }];
}

def SymbolicOptimization : Pass<"symbolic-optimize", "GraphOp"> {
  let summary = "Optimize the graph using symbolic reasoning.";
  let description = [{
    This pass leverages symbolic information present in the graph (usually as a
    result of shape inference) to optimize computations. It includes
    transformations such as:
    - moving certain operations within the graph
    - rewriting a series of operations into more efficient ones
    - canonicalizing certain operations into simpler form to enable
      downstream optimizations
  }];
}

#endif // GENERICML_GRAPHCOMPILER_MODIALECT_MOTRANSFORMS_TD
