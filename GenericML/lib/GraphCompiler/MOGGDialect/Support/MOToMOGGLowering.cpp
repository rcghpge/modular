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
#include "GenericML/GraphCompiler/MOGGDialect/Support/MOToKGENLowering.h"
#include "GenericML/GraphCompiler/MOGGDialect/Support/ParameterMapper.h"

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/LITDialect/LITUtils.h"
#include "KGEN/MOGGPreElab/MOGGPreElabDecorators.h"
#include "KGEN/MOGGPreElab/MOGGPreElabHelpers.h"
#include "KGEN/POPDialect/POPOps.h"
#include "KGEN/include/KGEN/MOGGPreElab/MojoKernelAdaptor.h"
#include "Support/AssertStream.h"

#include "mlir/IR/Verifier.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"

using namespace M;
using namespace MOGG;

using KGENKernelAdaptor =
    KGEN::MOGGPreElab::MojoKernelFunctionAdaptor<KGEN::GeneratorOp>;

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

struct KernelDesc {
  llvm::BitVector fusibleInputs;
  llvm::BitVector fusibleOutputs;
  // Currently, fuseBody == true iff isa<elementwise>(kernel), we might want to
  // relax the constraints later.
  bool isElemwise;

  // Extracts the kernel descriptor from a give mojo code after
  // pre-elaboration.
  KernelDesc(KGEN::GeneratorOp mojoCode) {
    KGENKernelAdaptor kernelAdaptor(mojoCode);
    ASSERT_STREAM(!kernelAdaptor.outputResult, "NOT IMPLEMENTED");

    fusibleInputs.resize(kernelAdaptor.inputArguments.size());
    fusibleOutputs.resize(kernelAdaptor.outputArguments.size());

    auto operandFusible =
        [&](KGEN::MOGGPreElab::MojoKernelOperandAdaptor operand) {
          auto tensorAdaptor =
              std::get_if<KGEN::MOGGPreElab::TensorOperandAdaptor>(
                  &operand.underlyingType);
          return tensorAdaptor && tensorAdaptor->fused;
        };

    // TODO: many corner cases not handled, but the case is true for elementwise
    // kernel.
    for (auto [i, out] : llvm::enumerate(kernelAdaptor.outputArguments))
      fusibleOutputs[i] = operandFusible(out);
    for (auto [i, in] : llvm::enumerate(kernelAdaptor.inputArguments))
      fusibleInputs[i] = operandFusible(in);

    isElemwise = kernelAdaptor.isElementwiseFunction();
  }
};

ErrorOr<ExpKernelOp> MOGGExpKernelBuilder::lowerToKernel() {
  auto hasChain = [](TypeRange types) {
    return llvm::any_of(types, [](Type ty) { return isa<MO::ChainType>(ty); });
  };

  if (hasChain(moOp->getOperandTypes()) || hasChain(moOp->getResultTypes()))
    llvm_unreachable("NOT IMPLEMENTED");

  // Currently, we call this function to do some basic checks in the kernel.
  // Some of these checks may happen again later in the function.
  // As we implement more and more of the check logic in MojoLibraryAnalysis,
  // All this code duplication will be removed from this file.
  auto errorOr = mojoLibraryAnalysis.verifyMOOpMatchesKernel(moOp);
  if (errorOr.isError())
    return errorOr.takeError();

  auto maybeKernel = getMojoKernelTarget(mojoLibraryAnalysis, moOp);
  if (maybeKernel.isError())
    return maybeKernel.takeError();

  auto mojoKernel = maybeKernel.takeValue();

  const KernelDesc desc(mojoKernel);

  // 1. input/output fusiable or not
  // 2. whether this is an elementwise kernel (which has fuse body).
  auto kernelOp =
      builder.create<ExpKernelOp>(loc,
                                  /*results=*/moOp->getResultTypes(),
                                  /*operands=*/moOp->getOperands(),
                                  /*summary=*/getSummaryString(moOp),
                                  /*isAsync=*/builder.getBoolAttr(isAsync),
                                  /*device=*/getDevice());

  StringRef kernelName = moOp->getName().getStringRef();
  Block *execBody = &kernelOp.getExecBody().front();

  builder.setInsertionPointToStart(execBody);
  auto outputPlaceHolder =
      builder.create<OutputPlaceholdersOp>(loc, kernelOp->getResultTypes());

  if (desc.isElemwise) {
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
    for (unsigned fusibleInput : desc.fusibleInputs.set_bits()) {
      Value fusibleTensor = inputs[fusibleInput];
      fusibleTensors.push_back(fusibleTensor);
    }

    auto bindOp = builder.create<BindLambdaOp>(loc, outTensorTp, fusibleTensors,
                                               BindLambdaKind::ElemwiseLambda);

    Block *elemwiseBody = &bindOp.getLambda().front();
    builder.setInsertionPointToStart(&bindOp.getLambda().front());

    unsigned fusedArgIdx = 0;
    SmallVector<Value> outlinedArgs;
    for (auto [idx, v] : llvm::enumerate(inputs)) {
      // If this is a fusible input, use the fused value, otherwise we
      // captured the infusible inputs.
      if (desc.fusibleInputs.test(idx))
        outlinedArgs.push_back(elemwiseBody->getArgument(fusedArgIdx++));
      else
        outlinedArgs.push_back(v);
    }

    builder.setInsertionPointToStart(elemwiseBody);
    auto callOutlined = builder.create<CallOutlinedOp>(
        loc, resultValTp, kernelName, outlinedArgs);
    builder.create<OutputOp>(loc, callOutlined->getResults());
    builder.setInsertionPointAfter(bindOp);
    builder.create<CallForeachOp>(loc, bindOp->getResult(0),
                                  outputPlaceHolder->getResult(0));
  } else {
    SmallVector<Value> outputs(outputPlaceHolder->getResults());
    SmallVector<Value> inputs(execBody->getArguments().begin(),
                              execBody->getArguments().end());

    for (unsigned fusibleInput : desc.fusibleInputs.set_bits()) {
      Value concreteInput = inputs[fusibleInput];
      Value bound = builder.create<BindLambdaOp>(loc, concreteInput.getType(),
                                                 concreteInput,
                                                 BindLambdaKind::InputLambda);
      inputs[fusibleInput] = bound;
    }

    for (unsigned fusibleOutput : desc.fusibleOutputs.set_bits()) {
      Value concreteOutput = outputs[fusibleOutput];
      Value bound = builder.create<BindLambdaOp>(loc, concreteOutput.getType(),
                                                 concreteOutput,
                                                 BindLambdaKind::OutputLambda);
      outputs[fusibleOutput] = bound;
    }
    builder.create<CallExecuteOp>(loc, kernelName, inputs, outputs);
  }

  return kernelOp;
}
