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
    Pass to materialize shape inference functions for ops that have them, and
    add invocations to the graph. The ops need to be fully parametric (i.e.
    unknown shapes or dimensions in the arguments and results are not allowed).
    The shape inference functions are not evaluated.
  }];
}

def InlineShapeFunctions : Pass<"inline-shape-funcs", "GraphOp"> {
  let summary = "Inlines shape functions into the graph.";
  let description = [{
    Inlines (already materialized) shape functions into the graph as parameter
    computations.
  }];
}

def PropagateShapes : Pass<"propagate-shapes", "GraphOp"> {
  let summary = "Propagates shape parameters through the graph.";
  let description = [{
    Evaluates shape parameter logic in the graph, simplifying and reducing shape
    logic as much as it can.
  }];
}

def ResolveUnknownParameters : Pass<"resolve-unknown-parameters", "ModuleOp"> {
  let summary = "Resolves unknown tensor shape and dimension parameters.";
  let description = [{
    Pass to parameterize `mo` ops. That is, replaces unknown shape and dimension
    parameters (i.e. `?`) in `!mo.tensor` instances with parameter references.
  }];
}

def StaticizeShapes : Pass<"staticize-shapes", "GraphOp"> {
  let summary = "Lower shape-like tensors to `!mo.shape`.";
  let description = [{
    This pass attempts to lower shape-like MO tensors (i.e. rank-1, signed 32 or
    64 bit integer) to the special `!mo.shape` type, and lower computations on
    them to static operations. The pass currently only resolves shapes that
    originate from `mo.shape_of` ops. In particular shape-like (or dim-like)
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

#endif // GENERICML_GRAPHCOMPILER_MODIALECT_MOTRANSFORMS_TD
