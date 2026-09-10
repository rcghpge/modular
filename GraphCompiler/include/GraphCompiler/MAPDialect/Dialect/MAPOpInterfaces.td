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
      "Returns the outer-scope value `arg` shadows, or null when `arg` is not "
      "one of this closure's capture block args -- a leading callback "
      "parameter (the index, or an epilogue's stored value) shadows nothing "
      "outside the closure, and neither does an argument of some other op. "
      "The inverse of getCaptureBlockArgs(), matched by identity rather than "
      "by recomputing where the captures start, so a caller walking outward "
      "from a block argument never restates that offset.",
      "::mlir::Value", "getCaptureFor", (ins "::mlir::BlockArgument":$arg),
      [{
        // Reached through the interface, not `$_op`: a default body is emitted
        // per concrete op, and `getCaptureBlockArgs` is itself an interface
        // default rather than a member of `IterForeachBodyOp` & co.
        auto closure =
            ::mlir::cast<::M::MAP::CapturingFusionClosure>($_op.getOperation());
        ::mlir::Block::BlockArgListType captureArgs =
            closure.getCaptureBlockArgs();
        for (unsigned i = 0, e = captureArgs.size(); i < e; ++i)
          if (captureArgs[i] == arg)
            return closure.getCaptures()[i];
        return {};
      }]
    >,
    InterfaceMethod<
      "Appends `capture` to the capture operands WITHOUT adding a shadowing "
      "block argument. Segment-aware, so it stays correct for a closure whose "
      "operands are split into segments (e.g. the epilogue's trailing "
      "`$intermediateTensor`). Use when the block argument is added separately -- "
      "e.g. a fuser adding it to a pre-clone body block; `addCapture` is the "
      "usual entry point that also adds the block argument.",
      "void", "appendCaptureOperand", (ins "::mlir::Value":$capture),
      [{
        $_op.getCapturesMutable().append(capture);
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
        // Same segment-aware append as `appendCaptureOperand`; inlined here
        // because a default method cannot call another via `$_op` (a
        // default-only method is not a member of the concrete op).
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

def MAP_StrideViewOpInterface : OpInterface<"StrideViewOpInterface"> {
  let description = [{
    Common interface for the hoisted stride views -- `map.index.slice.stride`,
    `map.index.broadcast.stride`, `map.index.reshape.stride`,
    `map.index.transpose.stride` -- the form
    `IndexTransformOpInterface::createStridedView` materializes. Each produces
    a tensor of its view's OUTPUT shape over storage it does not own, so a
    consumer that wants the tensor being re-addressed rather than the view
    itself walks through it here instead of listing the concrete view kinds.
  }];
  let cppNamespace = "::M::MAP";

  let methods = [
    InterfaceMethod<
      "Returns the tensor this view re-addresses. Nothing is copied, so the "
      "result holds the same elements as the view, only differently indexed.",
      "::mlir::Value", "getViewedTensor", (ins),
      "return $_op.getInput();"
    >,
  ];
}

#endif // GENERICML_GRAPHCOMPILER_MAPDIALECT_MAPOPINTERFACES_TD
