//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/MODialect/MOAttrs.h"
#include "GenericML/GraphCompiler/MODialect/MOOps.h"
#include "GenericML/GraphCompiler/Transforms/MODialect/MOTransforms.h"

namespace M::MO {

#define GEN_PASS_DEF_INFERLAYOUTS
#include "GenericML/GraphCompiler/Transforms/MODialect/MOTransforms.h.inc"

//===----------------------------------------------------------------------===//
// InferLayouts pass.
//===----------------------------------------------------------------------===//

struct InferLayoutPass : public impl::InferLayoutsBase<InferLayoutPass> {

  //===----------------------------------------------------------------------===//
  // Helper functions
  //===----------------------------------------------------------------------===//

  /// Add the mapping `(value : layout)` to `valueToInferredLayout`. Assume
  /// `op` is the source of the constraint, and it's used to emit error if
  /// `value` is already mapped to a different layout.
  ///
  /// NOTE if `value`'s type already contains a layout attribute, this becomes
  /// a no-op, so that the pass is idempotent and only infer missing layouts.
  ///
  /// Return failure if any error is emitted.
  LogicalResult addLayoutConstraint(Operation *op, TensorValue value,
                                    LayoutAttr layout) {
    if (value.getType().getLayoutAttr())
      return success();

    auto [iter, inserted] = valueToInferredLayout.insert({value, layout});
    if (!inserted && iter->getSecond() != layout) {
      return op->emitError()
             << "attempting to constrain value to layout: " << layout
             << " when it's already constrained to layout: "
             << iter->getSecond();
    }

    return success();
  };

  /// Walk through operations in `graphOp`, identify ops that impose direct
  /// layout constraints on their inputs/outputs (e.g., `mo.conv` requires
  /// `NHWC` layout for its input), and collect those in
  /// `valueToInferredLayout`.
  ///
  /// Return failure if we encountered conflicting constraints (e.g., a value is
  /// constrained to have both `NHWC` and `NCHW` layouts.
  LogicalResult addSeedingConstraints() {
    // Declare some canonical target-independent layouts required by MO ops.
    LayoutAttr convInputLayout = LayoutAttr::get(ctx, "NHWC");
    LayoutAttr convFilterLayout = LayoutAttr::get(ctx, "RSCF");
    LayoutAttr matmulWeightsLayout = LayoutAttr::get(ctx, "KN");

    // Infer layouts by walking over constraining operations and updating the
    // `valueToInferredLayout` map.
    WalkResult walkResult = graphOp.walk([&](Operation *op) {
      if (auto convOp = dyn_cast<ConvOp>(op)) {
        TensorValue input = convOp.getInput();
        TensorValue filter = convOp.getFilter();
        if (failed(addLayoutConstraint(op, input, convInputLayout)) ||
            failed(addLayoutConstraint(op, filter, convFilterLayout)))
          return WalkResult::interrupt();

      } else if (auto matmulOp = dyn_cast<MatmulOp>(op)) {
        TensorValue weights = matmulOp.getTensorB();
        if (failed(addLayoutConstraint(op, weights, matmulWeightsLayout)))
          return WalkResult::interrupt();
      }

      return WalkResult::advance();
    });
    return failure(walkResult.wasInterrupted());
  }

  /// Propagate layout through ops (e.g., from input to output or vice-versa for
  /// elementwise ops, which requires same layouts for all inputs and outputs).
  LogicalResult propagateLayoutThroughOps() {
    SmallVector<TensorValue> worklist;
    DenseSet<TensorValue> worklistHistory;

    auto tryPushOntoWorklist = [&](TensorValue value) {
      if (auto [_, inserted] = worklistHistory.insert(value); inserted)
        worklist.push_back(value);
    };
    // Intialize the worklist
    for (auto [value, _] : valueToInferredLayout)
      tryPushOntoWorklist(value);

    // Process the worklist
    while (!worklist.empty()) {
      TensorValue value = worklist.pop_back_val();
      LayoutAttr layout = valueToInferredLayout.at(value);

      SmallVector<Operation *> affectedOps(value.getUsers());
      if (Operation *definingOp = value.getDefiningOp(); definingOp)
        affectedOps.push_back(definingOp);

      for (Operation *op : affectedOps) {
        // For now we only look at ops with this trait, but we might propagate
        // layout through more ops like transpose.
        if (op->hasTrait<mlir::OpTrait::SameOperandsAndResultType>()) {
          // For some reason `llvm::concat` produces null values.
          SmallVector<Value> inputAndOutputs(op->getOperands());
          llvm::append_range(inputAndOutputs, op->getResults());
          // We must propagate through both inputs and outputs for generality.
          for (Value val : inputAndOutputs) {
            if (auto tensorVal = dyn_cast<TensorValue>(val)) {
              // Short-circuit on any error during layout inference
              if (failed(addLayoutConstraint(op, tensorVal, layout)))
                return failure();
              tryPushOntoWorklist(tensorVal);
            }
          }
        }
      }
    }
    return success();
  }

  /// For each `(value, layout)` pair in `valueToInferredLayout`, update the
  /// `value`'s type to contain `layout`, and update the `mo.graph` function
  /// type for consistency.
  void assignInferredLayoutsToTypes() {
    // Update the layout attributes of tensor types in the body of the
    // `mo.graph`, based on the `valueToInferredLayout` map.
    for (auto [value, layout] : valueToInferredLayout)
      value.setType(value.getType().cloneWith(layout));

    // Update/synchronize the `mo.graph` function type because the input or
    // output tensor values might have been labelled with new layout.
    synchronizeMetadataUpdates(graphOp);
  }

  //===----------------------------------------------------------------------===//
  // Pass entry point
  //===----------------------------------------------------------------------===//

  void runOnOperation() override {
    // Initialize the struct fields
    this->ctx = &getContext();
    this->graphOp = getOperation();

    if (failed(addSeedingConstraints()))
      return signalPassFailure();

    // This is needed to tame the MO type system.
    if (failed(propagateLayoutThroughOps()))
      return signalPassFailure();

    assignInferredLayoutsToTypes();

    // Clean up struct fields
    valueToInferredLayout.clear();
    this->graphOp = {};
  }

  //===----------------------------------------------------------------------===//
  // Struct fields.
  //===----------------------------------------------------------------------===//

  // The MLIR context of this pass
  MLIRContext *ctx;
  // The graph op this pass is operating upon.
  GraphOp graphOp;
  // A map from tensor value to their inferred layout. Note that not all tensor
  // values in the `graphOp` will be in this map, because the inference works on
  // a subset of ops.
  DenseMap<TensorValue, LayoutAttr> valueToInferredLayout;
};

} // namespace M::MO
