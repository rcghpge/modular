//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinOps.h"

#include "GenericML/GraphCompiler/MODialect/MOOps.h"
#include "GenericML/GraphCompiler/MODialect/MOTypes.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGPasses.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGTypes.h"
#include "GenericML/GraphCompiler/MOGGDialect/Support/FusionUtils.h"
#include "GenericML/GraphCompiler/MOGGDialect/Support/KGENUtils.h"
#include "GenericML/GraphCompiler/MOGGDialect/Support/MojoTargetFunctions.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "Runtime/Tensor/Tensor.h"
#include "Support/AssertStream.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/IR/IRMapping.h"

using namespace M;

namespace M::MOGG {

#define GEN_PASS_DEF_FUSEAFFINEVIEWS
#include "GenericML/GraphCompiler/MOGGDialect/MOGGPasses.h.inc"

namespace {

struct FuseAffineViewsPass
    : public impl::FuseAffineViewsBase<FuseAffineViewsPass> {
  using FuseAffineViewsBase::FuseAffineViewsBase;

  static inline bool isFusibleNonElementwise(Operation *op) {
    auto kernel = dyn_cast<KernelOp>(op);
    if (!kernel)
      return false;

    // Cannot be elementwise kernel, that is fused differently.
    auto isElemwise = kernel.getFusionBodyLambda();
    if (isElemwise)
      return false;

    std::optional<ArrayAttr> outputLambdas = kernel.getOutputFusionInterface();
    if (!outputLambdas.has_value())
      return false;

    // Output can only have one user.
    Value result = kernel.getResult(0);
    if (!result.hasOneUse())
      return false;

    // Avoid KGEN kernels for now.
    if (kernel.getIsKGEN())
      return false;

    return true;
  }

  void runOnOperation() override {
    auto graph = getOperation();
    MLIRContext *ctx = graph.getContext();

    for (MOGG::KernelOp kernel : graph.getOps<MOGG::KernelOp>()) {

      if (!isFusibleNonElementwise(kernel))
        continue;

      Value result = kernel.getResult(0);
      Operation *user = *result.getUsers().begin();

      // 1. Fuse the view operation as a view on the output buffer. Clone the
      // kernel to operate on the new shape.
      if (auto addDim = dyn_cast_or_null<MO::AddSingletonDimOp>(user)) {
        MO::TensorType newType = addDim.getResult().getType();

        OutputPlaceholdersOp outPh = kernel.getOutputPlaceholder();
        Value outTensor = outPh.getResults()[0];
        auto oldType = cast<MO::TensorType>(outTensor.getType());

        // Update the internal placeholder op.
        outTensor.setType(newType);

        // Insert a view to the old shape and replace everything EXCEPT the
        // output return and the internal simd store with this view.
        OpBuilder builder{outPh};
        builder.setInsertionPointAfter(outPh);

        auto shapeTy = MOGG::ShapeType::get(ctx, oldType.getTypedShapeAttr());
        Value shape =
            builder.create<MOGG::ParamShapeToValueOp>(outPh.getLoc(), shapeTy);

        auto call = builder.create<MOGG::CallOp>(
            outPh.getLoc(), TypeRange{oldType}, "reshape_contiguous_buffer",
            ValueRange{outTensor, shape}, mlir::ArrayAttr{},
            /*kgenAttrs=*/mlir::DictionaryAttr{});
        Value reshapedOutput = call.getResult(0);

        outTensor.replaceUsesWithIf(reshapedOutput, [&](OpOperand &operand) {
          Operation *usr = operand.getOwner();

          // Don't replace the reshape we just added, ofc.
          if (usr == call)
            return false;

          // Output should be returning the same shape as the graph level
          // expects.
          if (isa<MOGG::OutputOp>(usr))
            return false;

          // The store should be operating on the tensor with the same
          // dimension as the output expects.
          if (auto callOp = dyn_cast<MOGG::CallOp>(usr)) {
            if (callOp.getMojoKernelTarget() == Utils::MOGGSimdStoreTarget &&
                callOp.getOperand(0) == outTensor) {
              return false;
            }
          }
          return true;
        });

        // 2.Transform the indices of the output lambda by fusing in the view
        // operation.
        StringAttr lambdaName = kernel.getOutputFusionInterfaceAt(0);

        kernel.walk([&](MOGG::LambdaOp lambda) {
          // Insert call to inverse of the view op to invert the indices.
          if (lambda.getSymName() == lambdaName) {

            mlir::Block &block = lambda.getBodyRegion().front();
            Value indices = block.getArgument(0);
            builder.setInsertionPoint(&block, block.begin());

            mlir::NamedAttrList attrs;
            attrs.set("rank", builder.getIndexAttr(oldType.getRank()));
            attrs.set("axis", addDim.getAxis());
            attrs.set("value", builder.getIndexAttr(0));

            auto call = builder.create<MOGG::CallOp>(
                outPh.getLoc(), TypeRange{indices.getType()}, "insert_index",
                ValueRange{indices}, mlir::ArrayAttr{},
                /*kgenAttrs=*/builder.getDictionaryAttr(attrs));

            indices.replaceUsesWithIf(call->getResult(0),
                                      [&](OpOperand &operand) {
                                        if (operand.getOwner() == call)
                                          return false;
                                        return true;
                                      });
          }
        });

        // Change the graph level type of the mogg kernel op and replace the
        // old view.
        result.setType(newType);
        addDim.getResult().replaceAllUsesWith(result);
      }
    }
  }
};

} // namespace

} // namespace M::MOGG
