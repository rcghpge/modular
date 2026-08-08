//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Interfaces for MAP ops.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MAPDIALECT_MAPOPINTERFACES_TD
#define GENERICML_GRAPHCOMPILER_MAPDIALECT_MAPOPINTERFACES_TD

include "mlir/IR/OpBase.td"

def MAP_IndexTransformOpInterface : OpInterface<"IndexTransformOpInterface"> {
  let description = [{
    Common interface for ops that project a structured loop's own
    (untransformed) index into a different index addressing a view of one of
    the loop's inputs (today: `map.index.slice`; future view kinds -- e.g.
    broadcast, transpose -- would implement it too), generic over which
    concrete transform kind is involved.
  }];
  let cppNamespace = "::M::MAP";

  let methods = [
    InterfaceMethod<
      "Returns the index operand this op transforms.",
      "::mlir::Value", "getIndex", (ins)
    >,
    InterfaceMethod<
      "Materializes this transform's hoisted strided-view form: a standalone "
      "tensor value, of this transform's own output shape, that a load can "
      "address directly with the loop's own untransformed index instead of "
      "re-running the transform every iteration. `tensor` is the value this "
      "transform reads from -- either the original load target, or an outer "
      "hop's own hoisted view when chained.",
      "::mlir::Value", "createStridedView",
      (ins "::mlir::OpBuilder &":$builder, "::mlir::Value":$tensor)
    >,
  ];
}

#endif // GENERICML_GRAPHCOMPILER_MAPDIALECT_MAPOPINTERFACES_TD
