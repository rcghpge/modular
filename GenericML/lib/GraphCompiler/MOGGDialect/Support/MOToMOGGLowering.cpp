//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/MOGGDialect/Support/MOToMOGGLowering.h"
#include "GenericML/GraphCompiler/MODialect/MOOps.h"
#include "GenericML/GraphCompiler/MOGGDialect/Analysis/MojoLibraryAnalysis.h"
#include "GenericML/GraphCompiler/MOGGDialect/Analysis/Support.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GenericML/GraphCompiler/MOGGDialect/Support/FusionUtils.h"
#include "GenericML/GraphCompiler/MOGGDialect/Support/KGENStubBuilder.h"
#include "GenericML/GraphCompiler/MOGGDialect/Support/KGENUtils.h"
#include "GenericML/GraphCompiler/MOGGDialect/Support/KernelSlicing.h"

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/LITDialect/LITUtils.h"
#include "KGEN/MOGGPreElab/MOGGPreElabDecorators.h"
#include "KGEN/MOGGPreElab/MOGGPreElabHelpers.h"
#include "KGEN/POPDialect/POPOps.h"
#include "Support/AssertStream.h"

#include "mlir/IR/Verifier.h"
#include "llvm/ADT/STLExtras.h"

using namespace M;
using namespace MOGG;

MOGGExpKernelBuilder::MOGGExpKernelBuilder(
    Operation *op, OpBuilder &insertPoint, bool forceSync,
    MOGG::MojoLibraryAnalysis &mojoLibraryAnalysis)
    : moOp(op), graphOp(op->getParentOfType<MO::GraphOp>()),
      builder(insertPoint), ctx(op->getContext()), loc(op->getLoc()),
      mojoLibraryAnalysis(mojoLibraryAnalysis),
      typeManager(
          MOGG::TypeManager::get(ctx, mojoLibraryAnalysis).takeValue()) {

  isAsync = (getDevice() == getHostDevice()) && !forceSync &&
            !isa<MO::LayoutTransformOp>(op);

  // Get ready for building ops.
  builder.setInsertionPoint(op);
}

template <typename Func>
void forEachFusedArgument(
    ArrayRef<KGEN::MOGGPreElab::MojoKernelOperandAdaptor> operands,
    Func &&func) {
  for (const auto &[i, arg] : llvm::enumerate(operands)) {
    if (!arg.isFusedTensorType())
      continue;

    func(i);
  }
}

void MOGGExpKernelBuilder::lowerToElementwiseKernel(
    ExpKernelOp kernelOp, const KernelAdaptor &adaptor) {
  auto execBody = &kernelOp.getExecBody().front();
  SmallVector<Value> inputs(execBody->getArguments().begin(),
                            execBody->getArguments().end());

  auto outTensorTp = cast<MO::TensorType>(kernelOp->getResultTypes()[0]);
  auto resultValTp = outTensorTp.getElementType();

  // Values
  auto inputValTps = llvm::map_to_vector(moOp->getOperandTypes(), [](Type t) {
    // Only handles tensor inputs for now.
    return cast<MO::TensorType>(t).getElementType();
  });

  SmallVector<Value> fusibleTensors;
  forEachFusedArgument(
      adaptor.executeFunction->inputArguments,
      [&](uint64_t idx) { fusibleTensors.push_back(inputs[idx]); });
  auto bindOp = builder.create<BindLambdaOp>(loc, outTensorTp, fusibleTensors,
                                             BindLambdaKind::ElemwiseLambda);

  Block *elemwiseBody = &bindOp.getLambda().front();
  builder.setInsertionPointToStart(&bindOp.getLambda().front());

  unsigned fusedArgIdx = 0;
  SmallVector<Value> outlinedArgs;
  for (auto [idx, v] : llvm::enumerate(inputs)) {
    // If this is a fusible input, use the fused value, otherwise we
    // captured the infusible inputs.
    if (adaptor.executeFunction->inputArguments[idx].isFusedTensorType())
      outlinedArgs.push_back(elemwiseBody->getArgument(fusedArgIdx++));
    else
      outlinedArgs.push_back(v);
  }

  builder.setInsertionPointToStart(elemwiseBody);
  auto callOutlined = builder.create<CallOutlinedOp>(
      loc, resultValTp, this->moOp->getName().getStringRef(), outlinedArgs);
  builder.create<OutputOp>(loc, callOutlined->getResults());
  builder.setInsertionPointAfter(bindOp);
  builder.create<CallForeachOp>(loc, bindOp->getResult(0),
                                kernelOp.getOutputPlaceholder()->getResult(0));
}

void MOGGExpKernelBuilder::lowerToNormalKernel(ExpKernelOp kernelOp,
                                               const KernelAdaptor &adaptor) {
  auto execBody = &kernelOp.getExecBody().front();
  auto outputPlaceholder = kernelOp.getOutputPlaceholder();
  SmallVector<Value> outputs(outputPlaceholder->getResults());
  SmallVector<Value> inputs(execBody->getArguments().begin(),
                            execBody->getArguments().end());

  forEachFusedArgument(
      adaptor.executeFunction->inputArguments, [&](uint64_t idx) {
        Value concreteInput = inputs[idx];
        Value bound = builder.create<BindLambdaOp>(loc, concreteInput.getType(),
                                                   concreteInput,
                                                   BindLambdaKind::InputLambda);
        inputs[idx] = bound;
      });

  forEachFusedArgument(adaptor.executeFunction->outputArguments,
                       [&](uint64_t idx) {
                         Value concreteOutput = outputs[idx];
                         Value bound = builder.create<BindLambdaOp>(
                             loc, concreteOutput.getType(), concreteOutput,
                             BindLambdaKind::OutputLambda);
                         outputs[idx] = bound;
                       });
  builder.create<CallExecuteOp>(loc, moOp->getName().getStringRef(), inputs,
                                outputs);
}

ErrorOr<ExpKernelOp> MOGGExpKernelBuilder::lowerToKernel() {
  // Verify the MO op with the associated kernel to simplify lowering logic (no
  // errors will be reported, just assertions).
  auto errorOr = mojoLibraryAnalysis.verifyMOOpMatchesKernel(moOp);
  if (errorOr.isError())
    return errorOr.takeError();

  auto kernelOp =
      builder.create<ExpKernelOp>(loc,
                                  /*results=*/moOp->getResultTypes(),
                                  /*operands=*/moOp->getOperands(),
                                  /*summary=*/getSummaryString(moOp),
                                  /*isAsync=*/builder.getBoolAttr(isAsync),
                                  /*device=*/getDevice());
  builder.setInsertionPointToStart(&kernelOp.getExecBody().front());
  builder.create<OutputPlaceholdersOp>(loc, kernelOp->getResultTypes());

  auto mojoKernel = *getMojoKernelTarget(mojoLibraryAnalysis, moOp);
  KernelAdaptor kernelAdaptor("", mojoKernel, nullptr);
  if (kernelAdaptor.executeFunction->isElementwise())
    lowerToElementwiseKernel(kernelOp, kernelAdaptor);
  else
    lowerToNormalKernel(kernelOp, kernelAdaptor);

  return kernelOp;
}
