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
#include "GenericML/GraphCompiler/MOGGDialect/Support/KGENStubBuilderHelpers.h"
#include "GenericML/GraphCompiler/MOGGDialect/Support/KGENUtils.h"
#include "GenericML/GraphCompiler/MOGGDialect/Support/MojoTargetFunctions.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "Runtime/Tensor/Tensor.h"
#include "Support/AssertStream.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"

using namespace M;

namespace M::MOGG {

#define GEN_PASS_DEF_FUSEAFFINEVIEWS
#include "GenericML/GraphCompiler/MOGGDialect/MOGGPasses.h.inc"

namespace {

// TODO: Better to rename the pass, by no means a gather operation (which is
// handled in the pass) is "affine".
struct FuseAffineViewsPass
    : public impl::FuseAffineViewsBase<FuseAffineViewsPass> {
  using FuseAffineViewsBase::FuseAffineViewsBase;

  static inline bool isFusibleNonElementwise(KernelOp kernel) {
    // Cannot be elementwise kernel, that is fused differently.
    auto isElemwise = kernel.getFusionBodyLambda();
    if (isElemwise)
      return false;

    // Avoid KGEN kernels for now.
    if (kernel.getIsKGEN())
      return false;

    return true;
  }

  static inline bool isOutputFusibleNonElementwise(KernelOp kernel) {
    if (!isFusibleNonElementwise(kernel))
      return false;

    if (!kernel.getOutputFusionInterface().has_value())
      return false;

    // Output can only have one user.
    Value result = kernel.getResult(0);
    if (!result.hasOneUse())
      return false;
    Operation *user = *result.getUsers().begin();

    // Only fuse within the same block and not across, i.e, control flow.
    if (kernel->getBlock() != user->getBlock()) {
      return false;
    }

    return true;
  }

  static inline bool isInputFusibleNonElementwise(KernelOp kernel,
                                                  size_t inputNum,
                                                  KernelOp inputKernel) {
    if (!isFusibleNonElementwise(kernel))
      return false;

    // Only fuse within the same block and not across, i.e, control flow.
    if (kernel->getBlock() != inputKernel->getBlock())
      return false;

    return !kernel.getInputFusionInterfaceAt(inputNum).strref().empty();
  }

  void addSingletonDimTransform(Value indices, MO::TensorType outputTensorType,
                                MO::AddSingletonDimOp addDim,
                                OpBuilder &builder) {
    mlir::NamedAttrList attrs;
    attrs.set("rank", builder.getIndexAttr(outputTensorType.getRank()));
    attrs.set("axis", addDim.getAxis());
    attrs.set("value", builder.getIndexAttr(0));

    auto call = builder.create<MOGG::CallOp>(
        addDim.getLoc(), TypeRange{indices.getType()}, "insert_index",
        ValueRange{indices}, mlir::ArrayAttr{},
        /*kgenAttrs=*/builder.getDictionaryAttr(attrs));

    indices.replaceUsesWithIf(call->getResult(0), [&](OpOperand &operand) {
      if (operand.getOwner() == call)
        return false;
      return true;
    });
  }

  void addSplitDim(Value indices, MO::TensorType outputTensorType,
                   MO::SplitDimOp splitDim, Operation *insertPt,
                   OpBuilder &builder) {
    mlir::MLIRContext *ctx = builder.getContext();

    mlir::NamedAttrList attrs;
    attrs.set("rank", builder.getIndexAttr(outputTensorType.getRank()));
    attrs.set("axis", splitDim.getAxis());

    int64_t i = splitDim.getAxis().getValue().getLimitedValue();

    // Insert the shape calculation outside of the lambda so the index is
    // captured by the lambda rather than the pointer to the shape.
    OpBuilder shapeCalcBuilder{ctx};
    shapeCalcBuilder.setInsertionPointAfter(insertPt);

    // Get the the dimension we are splitting by. For insance for [N] -> [n1,
    // n2], this value refers to `n2`.
    Value getDim = MOGG::lowerDimToKGEN(splitDim.getLoc(), shapeCalcBuilder,
                                        splitDim.getType().getDims()[i + 1]);

    // Add the actual indices transfrom into the lambda expression.
    auto call = builder.create<MOGG::CallOp>(
        splitDim.getLoc(), TypeRange{indices.getType()}, "split_dim_indices",
        ValueRange{indices, getDim}, mlir::ArrayAttr{},
        /*kgenAttrs=*/builder.getDictionaryAttr(attrs));

    indices.replaceUsesWithIf(call->getResult(0), [&](OpOperand &operand) {
      if (operand.getOwner() == call)
        return false;
      return true;
    });
  }

  Value addGatherIndicesTranslation(Location loc, OpBuilder &builder,
                                    Value indicesAfterGather,
                                    Value indicesBuffer, Value gatherAxis,
                                    Value inputShape) {
    // The parameters to the call are bound with NDBuffer and shape, which graph
    // compiler knows how to deduce from.

    // Translate gathered indices back to the original indices.
    auto call = builder.create<MOGG::CallOp>(
        loc, TypeRange{indicesAfterGather.getType()},
        "translate_gather_indices",
        ValueRange{indicesBuffer, gatherAxis, indicesAfterGather, inputShape},
        mlir::ArrayAttr{},
        /*kgenAttrs=*/builder.getDictionaryAttr({}));

    return call.getResult(0);
  }

  template <typename ViewOp>
  void fuseViewOpDim(MOGG::KernelOp kernel, ViewOp viewOp, Value result,
                     Operation *user) {
    MO::TensorType newType = viewOp.getResult().getType();

    OutputPlaceholdersOp outPh = kernel.getOutputPlaceholder();
    Value outTensor = outPh.getResults()[0];
    auto oldType = cast<MO::TensorType>(outTensor.getType());

    // Update the internal placeholder op.
    outTensor.setType(newType);

    // Insert a view to the old shape and replace everything EXCEPT the
    // output return and the internal simd store with this view.
    OpBuilder builder{outPh};
    builder.setInsertionPointAfter(outPh);

    auto shapeTy =
        MOGG::ShapeType::get(kernel.getContext(), oldType.getTypedShapeAttr());
    Value shapeInKGEN =
        MOGG::lowerShapeToKGEN(outPh->getLoc(), builder, oldType.getDims());
    Value shape = builder.create<MOGG::ToMOGGOp>(
        outPh->getLoc(), TypeRange{shapeTy}, ValueRange{shapeInKGEN});

    auto reshape = builder.create<MOGG::CallOp>(
        outPh.getLoc(), TypeRange{oldType}, "reshape_contiguous_buffer",
        ValueRange{outTensor, shape}, mlir::ArrayAttr{},
        /*kgenAttrs=*/mlir::DictionaryAttr{});
    Value reshapedOutput = reshape.getResult(0);

    outTensor.replaceUsesWithIf(reshapedOutput, [&](OpOperand &operand) {
      Operation *usr = operand.getOwner();

      // Don't replace the reshape we just added, ofc.
      if (usr == reshape)
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

        if constexpr (std::is_same_v<ViewOp, MO::AddSingletonDimOp>)
          addSingletonDimTransform(indices, oldType, viewOp, builder);
        else if (std::is_same_v<ViewOp, MO::SplitDimOp>)
          addSplitDim(indices, oldType, viewOp, reshape, builder);
      }
    });

    // Change the graph level type of the mogg kernel op and replace the
    // old view.
    result.setType(newType);
    viewOp.getResult().replaceAllUsesWith(result);
  }

  void fuseGatherInputKernel(KernelOp kernel, KernelOp gatherOp,
                             unsigned toFuseIdx) {
    OpBuilder builder(kernel->getContext());

    SmallVector<Value> gatherOperands = gatherOp.getOperands();
    MO::GatherOpAdaptor adaptor(gatherOperands);

    if (auto inputTp = cast<MO::TensorType>(adaptor.getInput().getType());
        inputTp.getDeviceRef() &&
        inputTp.getDeviceRef().getLabel() != kCPULabel) {
      builder.setInsertionPoint(kernel);
      Value deviceAxis = builder.create<MO::TransferOp>(
          gatherOp->getLoc(),
          cast<MO::TensorType>(adaptor.getAxis().getType())
              .cloneWith(inputTp.getDeviceRef()),
          adaptor.getAxis());
      gatherOperands[/*AxisOpNum=*/2] = deviceAxis;
    }

    // Update the input.
    kernel->eraseOperand(toFuseIdx);
    kernel->insertOperands(toFuseIdx, gatherOperands);

    StringAttr lambdaName = kernel.getInputFusionInterfaceAt(toFuseIdx);
    ArrayRef interfaces = kernel.getInputFusionInterface()->getValue();
    ArrayRef unfusedHead = interfaces.take_front(toFuseIdx);
    SmallVector<Attribute> fusedInterfaces(unfusedHead.begin(),
                                           unfusedHead.end());

    // Update the block argument.
    Value unFusedInput = kernel.getBody().getArgument(toFuseIdx);
    for (auto [opNum, val] : llvm::enumerate(gatherOperands)) {
      // Insert right after the current value being fused to avoid messing up
      // the loop iteration, in which the function is called.
      unsigned idx = toFuseIdx + 1 + opNum;
      Value fusedVal =
          kernel.getBody().insertArgument(idx, val.getType(), val.getLoc());
      // If this is the input, replace the use to the fused input.
      if (val == adaptor.getInput()) {
        unFusedInput.replaceAllUsesWith(fusedVal);
        fusedInterfaces.emplace_back(lambdaName);
      } else {
        fusedInterfaces.emplace_back(builder.getStringAttr(""));
      }
    }
    kernel.getBody().eraseArgument(toFuseIdx);
    ArrayRef unfusedTail = interfaces.drop_front(toFuseIdx + 1);
    fusedInterfaces.append(unfusedTail.begin(), unfusedTail.end());

    // Generate fused body.
    builder.setInsertionPointToStart(&kernel.getBody().front());
    MO::GatherOpAdaptor bodyAdaptor(kernel.getBody().getArguments().slice(
        toFuseIdx, gatherOp->getNumOperands()));

    // 1st, Extract the shape from the fused input.
    Value fusedInput = bodyAdaptor.getInput();
    auto resultTy = cast<MO::TensorType>(fusedInput.getType());
    auto shape = MOGG::ShapeType::get(builder.getContext(),
                                      resultTy.getTypedShapeAttr());

    Value shapeInKGEN =
        MOGG::lowerShapeToKGEN(gatherOp->getLoc(), builder, resultTy.getDims());
    Value fusedInputShape = builder.create<MOGG::ToMOGGOp>(
        gatherOp->getLoc(), TypeRange{shape}, ValueRange{shapeInKGEN});

    // 2st, Transform the input lambda body.
    kernel.walk([&](MOGG::LambdaOp lambda) {
      // Insert call to inverse of the view op to invert the indices.
      if (lambda.getSymName() == lambdaName) {
        mlir::Block &block = lambda.getBodyRegion().front();
        Value indicesAfterGather = block.getArgument(0);
        builder.setInsertionPoint(&block, block.begin());

        auto indicesBeforeGather = addGatherIndicesTranslation(
            gatherOp->getLoc(), builder, indicesAfterGather,
            bodyAdaptor.getIndices(), bodyAdaptor.getAxis(), fusedInputShape);

        indicesAfterGather.replaceAllUsesExcept(
            indicesBeforeGather, indicesBeforeGather.getDefiningOp());
        return WalkResult::interrupt();
      }

      return WalkResult::advance();
    });

    kernel.setInputFusionInterfaceAttr(builder.getArrayAttr(fusedInterfaces));
    kernel.setSummaryAttr(
        getSummaryString({gatherOp, kernel}, FusionType::HARD));
    // Trivially dead.
    if (gatherOp->getUsers().empty())
      gatherOp->erase();
  }

  void runOnOperation() override {
    auto graph = getOperation();

    graph.walk([&](MOGG::KernelOp kernel) {
      if (isOutputFusibleNonElementwise(kernel)) {
        Value result = kernel.getResult(0);
        Operation *user = *result.getUsers().begin();

        // 1. Fuse the view operation as a view on the output buffer. Clone the
        // kernel to operate on the new shape.
        if (auto addDim = dyn_cast_or_null<MO::AddSingletonDimOp>(user))
          fuseViewOpDim(kernel, addDim, result, user);
        else if (auto split = dyn_cast_or_null<MO::SplitDimOp>(user))
          fuseViewOpDim(kernel, split, result, user);
      }

      for (int idx = kernel.getNumOperands() - 1; idx >= 0; idx--) {
        OpOperand &operand = kernel->getOpOperand(idx);
        if (auto gatherKernel = operand.get().getDefiningOp<MOGG::KernelOp>()) {
          // Pattern match based on summary.
          //
          // TODO: This is not ideal, it would be much better if mo.gather can
          // be directly lowered to a elementwise-like kernel at the first
          // place. But that would require some extension on elementwise kernel
          // to support indices transformation.
          if (!gatherKernel.getSummary().has_value() ||
              gatherKernel.getSummary()->rsplit(':').first != "mo.gather" ||
              !isInputFusibleNonElementwise(kernel, operand.getOperandNumber(),
                                            gatherKernel))
            return;

          M::IntArrayElementsAttr attr;

          MO::GatherOpAdaptor adaptor(gatherKernel.getOperands());
          if (mlir::matchPattern(adaptor.getAxis(), mlir::m_Constant(&attr))) {
            int64_t axis = (*attr.begin()).getSExtValue();
            unsigned inputRank =
                cast<MO::TensorType>(adaptor.getInput().getType()).getRank();
            // Try doing input fusion on gather only when we can statically
            // verify that the gather axis is not the last dim, otherwise, it
            // might break the `simd_load`.
            if (axis != -1 && axis != inputRank - 1)
              fuseGatherInputKernel(kernel, gatherKernel,
                                    operand.getOperandNumber());
          }
        }
      }
    });
  }
};

} // namespace
} // namespace M::MOGG
