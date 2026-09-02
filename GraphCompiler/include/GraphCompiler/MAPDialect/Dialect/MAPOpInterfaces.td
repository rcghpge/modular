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

def MAP_CapturingFusionClosure : OpInterface<"CapturingFusionClosure"> {
  let description = [{
    Common interface for the three closure kinds a `map.iter.opaque` (or a
    to-be-outlined `map.iter.foreach`) holds: `map.iter.foreach.body`,
    `map.iter.prologue`, and `map.iter.epilogue`. Each is a single-block,
    isolated-from-above region reaching every outer value through a
    `$captures` operand, shadowed as the block's trailing arguments.
    `LoopInvariantViewMotion` and `ViewFuser` both dispatch on this
    interface generically instead of on each concrete closure type.
  }];

  let cppNamespace = "::M::MAP";

  let methods = [
    InterfaceMethod<
      "Returns the closure's single-block body region.",
      "::mlir::Region &", "getBody", (ins),
      "return $_op.getBody();"
    >,
    InterfaceMethod<
      "Returns the outer-scope values this closure captures.",
      "::mlir::ValueRange", "getCaptures", (ins),
      "return $_op.getCaptures();"
    >,
    InterfaceMethod<
      "Returns the body block arguments shadowing getCaptures(), one to "
      "one, in the same order -- always the trailing block arguments.",
      "::mlir::Block::BlockArgListType", "getCaptureBlockArgs", (ins),
      [{
        return $_op.getBody().front().getArguments().take_back(
            $_op.getCaptures().size());
      }]
    >,
    InterfaceMethod<
      "Appends `capture` as a new capture and returns its new shadowing "
      "block argument. May create a duplicate of an existing capture -- "
      "callers don't dedup; the shared capturing-closure canonicalizer does.",
      "::mlir::BlockArgument", "addCapture", (ins "::mlir::Value":$capture),
      [{
        ::mlir::BlockArgument arg = $_op.getBody().front().addArgument(
            capture.getType(), capture.getLoc());
        $_op.getCapturesMutable().append(capture);
        return arg;
      }]
    >,
    InterfaceMethod<
      "Erases the capture at `idx` along with its shadowing block argument.",
      "void", "eraseCaptureAt", (ins "unsigned":$idx),
      [{
        ::mlir::Block &body = $_op.getBody().front();
        unsigned offset = body.getNumArguments() - $_op.getCaptures().size();
        body.eraseArgument(offset + idx);
        $_op.getCapturesMutable().erase(idx);
      }]
    >,
  ];
}

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
      "hop's own hoisted view when chained. This op still lives inside the "
      "isolated closure being hoisted out of, so any of its OWN operands "
      "beyond `tensor` (e.g. a slice's starts/stops/steps) may themselves be "
      "capture-shadowing block arguments that don't dominate the hoisted "
      "op's insertion point outside the closure -- look every such operand "
      "up in `outerValues` (mapping each capture block argument to the "
      "outer-scope value it shadows) before using it.",
      "::mlir::Value", "createStridedView",
      (ins "::mlir::OpBuilder &":$builder, "::mlir::Value":$tensor,
           "const ::mlir::IRMapping &":$outerValues)
    >,
  ];
}

#endif // GENERICML_GRAPHCOMPILER_MAPDIALECT_MAPOPINTERFACES_TD
