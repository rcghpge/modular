//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements the MO dialect operations.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GenericML/GraphCompiler/Conversion/MOToPrimitives.h"
#include "GenericML/GraphCompiler/MOGGDialect/Utils.h"
#include "GenericML/GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "GenericML/GraphCompiler/MOSHDialect/MOSHTypes.h"
#include "GenericML/MGPDialect/MGPOps.h"
#include "GenericML/MGPDialect/MGPUtils.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/IR/FunctionImplementation.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/StringMap.h"

using namespace M;
using namespace MOGG;

/*
 * Lowering helpers.
 */
namespace {

static SmallVector<MO::TensorType> castToMoTensorTypes(TypeRange types) {
  return llvm::map_to_vector(types,
                             [](Type t) { return t.cast<MO::TensorType>(); });
}

// A MO tensor which has been lowered into a buffer and a spec.
struct LoweredTensor {
  /// The buffer for this tensor which is not immutable as it can be the output
  /// buffer.
  TypedValue<MGP::CpuBufferType> buffer;

  /// The mgp.cpu.tensor_spec describing the tensor the buffer
  /// represents.
  TypedValue<MGP::CpuTensorSpecType> spec;

  LoweredTensor() : buffer({}), spec({}) {}

  LoweredTensor(TypedValue<MGP::CpuBufferType> buf,
                TypedValue<MGP::CpuTensorSpecType> spec)
      : buffer(buf), spec(spec) {}
};

/// This class wraps some of the common functionality of the lowering process
/// into a shared class.
class LowerHelper {
public:
  LowerHelper(MGP::MOGraphLoweringState &state, mlir::Operation *op)
      : loweringState(state), opBeingLowered(op), loc(op->getLoc()) {}

  // Extract the buffer and spec from the MGP tensor each input value lowered
  // to, and return each pair as a LoweredTensor.
  SmallVector<LoweredTensor> getLoweredInputs();

  // Extract the buffer and spec from the MGP tensor that `val` lowered to, and
  // return them as a LoweredTensor.
  LoweredTensor getLoweredTensorForValue(Value val);

  // Lower the output buffer which will get passed into the input.
  ErrorOr<SmallVector<LoweredTensor>>
  lowerOutputBuffer(llvm::StringRef opName, Value inChain,
                    const SmallVector<LoweredTensor> &inputTensors);

  // Emit instructions to invoke the legacy primitive shape function that has
  // the given name, input values, and returns tensor spec of types
  // corresponding to `resultMoTypes`.
  SmallVector<TypedValue<MGP::CpuTensorSpecType>>
  invokeLegacyShapeFunc(StringRef shapeFuncName, ArrayRef<Value> inputs,
                        ArrayRef<MO::TensorType> resultMoTypes);

  // Emit instructions to invoke the legacy primitive shape function that has
  // the given name, input values, and returns tensor spec of given types.
  SmallVector<TypedValue<MGP::CpuTensorSpecType>>
  invokeLegacyShapeFunc(StringRef shapeFuncName, ArrayRef<Value> inputs,
                        ArrayRef<MGP::CpuTensorSpecType> resultSpecTypes);

  // Convert the tensors to a list of buffer/spec value pairs.
  SmallVector<Value> getOperands(SmallVector<LoweredTensor> &in,
                                 SmallVector<LoweredTensor> &outs);

  // Allocate tensors for all outputs given their buffers and specs. Tensor
  // creation is gated on inChain.
  void allocateOutputTensor(Value inChain,
                            SmallVector<LoweredTensor> &outputTensors);

private:
  MGP::MOGraphLoweringState &loweringState;
  mlir::Operation *opBeingLowered;
  mlir::Location loc;

  ErrorOr<LoweredTensor>
  lowerDynamicOutput(MGP::CpuTensorSpecType outSpecType, MO::TensorType moType,
                     llvm::StringRef opName, Value inChain,
                     const SmallVector<LoweredTensor> &inputTensors);

  /// Lower outputs whose shape function needs to return multiple output specs
  /// in one invokation.
  ErrorOr<SmallVector<LoweredTensor>>
  lowerMultiOutputShapeFunc(StringRef shapeFuncName, Value inChain,
                            ArrayRef<LoweredTensor> inputTensors,
                            ArrayRef<MO::TensorType> resultMoTypes);
};

SmallVector<LoweredTensor> LowerHelper::getLoweredInputs() {
  return llvm::map_to_vector(opBeingLowered->getOperands(), [&](Value operand) {
    return getLoweredTensorForValue(operand);
  });
}

LoweredTensor LowerHelper::getLoweredTensorForValue(Value val) {
  mlir::Value mgpTensor = loweringState.lookupExecuteLoweredValue(val);
  assert(mgpTensor != nullptr && "referenced SSA value has not been lowered");
  auto extract =
      loweringState.getExecuteOpBuilder().create<MGP::CpuTensorExtractOp>(
          loc, mgpTensor);

  // Add the expanded buffer and spec as arguments to the operation.
  return LoweredTensor(extract.getBufferResult(),
                       extract.getTensorSpecResult());
}

void LowerHelper::allocateOutputTensor(
    Value inChain, SmallVector<LoweredTensor> &outputTensors) {
  size_t outputIndex = 0;

  for (LoweredTensor &tensor : outputTensors) {
    // Get the type of the spec and infer the type of the tensor from that.
    auto ty = cast<MGP::CpuTensorSpecType>(tensor.spec.getType());
    auto tensorType = MGP::CpuTensorType::get(
        loweringState.getExecuteOpBuilder().getContext(), ty);

    // Create an immutable tensor
    auto tensorCreateOp =
        loweringState.getExecuteOpBuilder().create<MGP::CpuTensorCreateOp>(
            loc,
            /*result=*/tensorType,
            /*inChain=*/inChain,
            /*buffer=*/tensor.buffer,
            /*tensorSpec=*/tensor.spec);

    // Update the MO tensor to now refer to this mgp tensor.
    loweringState.emplaceExecuteLoweredValue(
        opBeingLowered->getResult(outputIndex), tensorCreateOp.getResult());
    outputIndex++;
  }
}

ErrorOr<LoweredTensor> LowerHelper::lowerDynamicOutput(
    MGP::CpuTensorSpecType outSpecType, MO::TensorType moType,
    llvm::StringRef opName, Value inChain,
    const SmallVector<LoweredTensor> &inputTensors) {
  // Dynamic Case
  // TODO: This is hacky, hacky, hacky, remove as soon as we can lower
  // parametric shapes to the runtime
  // TODO: If we decide we want this longer term (say, for MLAS kernels)
  // We should refactor it out as a Trait maybe? and make this a method
  // and data member of the op class.

  OpBuilder exeBuilder = loweringState.getExecuteOpBuilder();
  auto getOutputSpec = [&](StringRef opName, ArrayRef<Value> inputs) {
    return invokeLegacyShapeFunc(opName, inputs, {outSpecType}).front();
  };

  LoweredTensor output;

  // Hackery! Pure Hackery! I'd start setting up classes for this, but
  // again, I don't expect this to survive E
  if (!moType.hasUnknownShape() && !moType.hasStaticShape()) {
    // We get the values corresponding to dynamic dimensions.
    UNWRAP_ERROR(runtimeDims, loweringState.lowerShapeAttrDynDims(
                                  moType.getTypedShapeAttr(), loc));
    output.spec =
        loweringState.getExecuteOpBuilder()
            .create<MGP::CpuTensorSpecCreateOp>(loc, outSpecType, runtimeDims)
            .getResult();
  } else if (opName == "mop.kernels.conv") {
    Value stride = loweringState.lookupExecuteLoweredValue(
        opBeingLowered->getOperands()[2]);
    Value dilation = loweringState.lookupExecuteLoweredValue(
        opBeingLowered->getOperands()[3]);
    output.spec = getOutputSpec(
        "mop.shape.conv",
        {inputTensors[0].spec, inputTensors[1].spec, stride, dilation});
  } else if (opName == "mop.kernels.pool.max" ||
             opName == "mop.kernels.pool.avg") {
    Value kernel = loweringState.lookupExecuteLoweredValue(
        opBeingLowered->getOperands()[1]);
    Value stride = loweringState.lookupExecuteLoweredValue(
        opBeingLowered->getOperands()[2]);
    Value dilation = loweringState.lookupExecuteLoweredValue(
        opBeingLowered->getOperands()[3]);
    output.spec = getOutputSpec(
        "mop.shape.pool", {inputTensors[0].spec, kernel, stride, dilation});
  } else {
    llvm::Twine dynErr = opName + " requires output to have static shape";
    opBeingLowered->emitOpError(dynErr);
    return Error(dynErr);
  }
  output.buffer = exeBuilder
                      .create<MGP::CpuBufferAllocSpecOp>(
                          loc,
                          /*inChain=*/inChain,
                          /*runtimeSpec=*/output.spec,
                          /*rawAlign=*/loweringState.getMinimumCpuAlignment())
                      .getOutBuffer();

  return output;
}

SmallVector<Value> LowerHelper::getOperands(SmallVector<LoweredTensor> &ins,
                                            SmallVector<LoweredTensor> &outs) {
  // Pass the operands as a normal value list to MLIR.
  SmallVector<Value> operands;
  operands.reserve((ins.size() + outs.size()) * 2);
  for (LoweredTensor &tensor : ins) {
    operands.push_back(tensor.buffer);
    operands.push_back(tensor.spec);
  }

  for (LoweredTensor &tensor : outs) {
    operands.push_back(tensor.buffer);
    operands.push_back(tensor.spec);
  }

  return operands;
}

ErrorOr<SmallVector<LoweredTensor>>
LowerHelper::lowerMultiOutputShapeFunc(StringRef shapeFuncName, Value inChain,
                                       ArrayRef<LoweredTensor> inputTensors,
                                       ArrayRef<MO::TensorType> resultMoTypes) {
  SmallVector<LoweredTensor> outputBuffersAndSpecs;
  SmallVector<TypedValue<MGP::CpuTensorSpecType>> outputSpecs;
  if (shapeFuncName == "mop.shape.split") {
    // Get inputs for "mop.shape.split"(%inputSpec, %splitSizes, %axis)
    Value inputSpec = inputTensors[0].spec;
    Value splitSizes = loweringState.lookupExecuteLoweredValue(
        opBeingLowered->getOperands()[1]);
    Value axis = loweringState.lookupExecuteLoweredValue(
        opBeingLowered->getOperands()[2]);
    outputSpecs = invokeLegacyShapeFunc(
        shapeFuncName, {inputSpec, splitSizes, axis}, resultMoTypes);
  } else {
    opBeingLowered->emitError("multi-output shape func not supported");
    return Error("multi-output shape func not supported");
  }

  // Create output buffers from specs.
  for (const auto &outSpec : outputSpecs) {
    auto buffer = loweringState.getExecuteOpBuilder()
                      .create<MGP::CpuBufferAllocSpecOp>(
                          loc,
                          /*inChain=*/inChain,
                          /*runtimeSpec=*/outSpec,
                          /*rawAlign=*/loweringState.getMinimumCpuAlignment())
                      .getOutBuffer();
    outputBuffersAndSpecs.push_back({buffer, outSpec});
  }

  return outputBuffersAndSpecs;
}

ErrorOr<SmallVector<LoweredTensor>>
LowerHelper::lowerOutputBuffer(llvm::StringRef opName, Value inChain,
                               const SmallVector<LoweredTensor> &inputTensors) {
  // Handle ops with shape functions that return multiple outputs.
  // For example, mop.shape.split returns a variable number of outputs, so it
  // takes a slightly different lowering path.
  static const llvm::StringMap<StringRef> multiOutputShapeFuncs{
      {"mop.kernels.split", "mop.shape.split"},
  };

  SmallVector<MO::TensorType> resultMoTypes =
      castToMoTensorTypes(opBeingLowered->getResultTypes());

  if (multiOutputShapeFuncs.contains(opName) &&
      llvm::any_of(resultMoTypes, [](MO::TensorType resTy) {
        return resTy.hasUnknownShape();
      })) {
    return lowerMultiOutputShapeFunc(multiOutputShapeFuncs.at(opName), inChain,
                                     inputTensors, resultMoTypes);
  }

  // Proceed with the normal case when there is only one output, or all the
  // output shapes are known statically.
  SmallVector<LoweredTensor> outputBufferAndSpec;

  // Allocate the output buffer for each result and add as argument.
  for (MO::TensorType ty : resultMoTypes) {
    if (!ty.hasRank())
      return Error("dynamic output rank not supported");
    MGP::CpuTensorSpecType outSpecType = MGP::createCpuTensorSpec(ty);
    std::optional<uint64_t> bufferSize = outSpecType.getSizeInBytes();
    LoweredTensor output;
    if (!bufferSize) {
      UNWRAP_ERROR(out, lowerDynamicOutput(outSpecType, ty, opName, inChain,
                                           inputTensors));
      output = out;
    } else {
      // Static Case
      // Allocate the buffer for the output.
      output.buffer =
          loweringState.getExecuteOpBuilder()
              .create<MGP::CpuBufferAllocStaticOp>(
                  loc,
                  /*inChain=*/inChain,
                  /*aSize=*/*bufferSize,
                  /*bRawAlign=*/loweringState.getMinimumCpuAlignment())
              .getOutBuffer();

      // Create an MGP tensor to match the MO output tensor.
      output.spec = loweringState.getExecuteOpBuilder()
                        .create<MGP::CpuTensorSpecCreateOp>(loc, outSpecType)
                        .getResult();
    }

    // Track the buffer and spec so we can create the immutable tensor after the
    // operation.
    outputBufferAndSpec.push_back(output);
  }

  return outputBufferAndSpec;
}

SmallVector<TypedValue<MGP::CpuTensorSpecType>>
LowerHelper::invokeLegacyShapeFunc(StringRef shapeFuncName,
                                   ArrayRef<Value> inputs,
                                   ArrayRef<MO::TensorType> resultMoTypes) {
  SmallVector<MGP::CpuTensorSpecType> resultSpecTypes =
      llvm::map_to_vector(resultMoTypes, [](MO::TensorType t) {
        return MGP::createCpuTensorSpec(t);
      });
  return invokeLegacyShapeFunc(shapeFuncName, inputs, resultSpecTypes);
};

SmallVector<TypedValue<MGP::CpuTensorSpecType>>
LowerHelper::invokeLegacyShapeFunc(
    StringRef shapeFuncName, ArrayRef<Value> inputs,
    ArrayRef<MGP::CpuTensorSpecType> resultSpecTypes) {
  OperationState opState(loc, shapeFuncName);
  opState.addOperands(inputs);
  opState.addTypes(resultSpecTypes);

  Operation *invokeOp = loweringState.getExecuteOpBuilder().create(opState);

  // Return the output specs.
  return llvm::map_to_vector(invokeOp->getResults(), [](Value v) {
    return cast<TypedValue<MGP::CpuTensorSpecType>>(v);
  });
};

} // namespace

LogicalResult ExtOp::lower(MGP::MOGraphLoweringState &loweringState) {
  ExtOp op = *this;
  LowerHelper helper{loweringState, op};

  // Operands to pass to the operation. Input tensors are expanded to buffers
  // and specs in IR those are passed in.
  auto inputTensors = helper.getLoweredInputs();

  // Ext ops carry the lowered operation name around with them,
  // others might do something else.
  StringRef opName = op.getOpName();

  // Lower the outputs to buffers which can then be passed back in as
  // parameters.
  //
  // At this stage we gate the allocation only on the overall 'execute' block's
  // chain. The DeferAllocsPass will change this to use the chain which signals
  // when all of the operation's input buffers have been written to, thus
  // ensuring we allocate buffers just-in-time for their writers. Jit compiled
  // kernels have no shape function and just use the shape infered values
  // directly and are guarded to only run on static or shape infered ops.
  auto errOr = helper.lowerOutputBuffer(opName, loweringState.getExecuteChain(),
                                        inputTensors);
  if (errOr.isError())
    return failure();
  SmallVector<LoweredTensor> outputBufferAndSpec = errOr.takeValue();

  size_t numOutputBuffers = outputBufferAndSpec.size();
  SmallVector<Value> operands =
      helper.getOperands(inputTensors, outputBufferAndSpec);

  // Create a call to a custom invoke function we can put in the MEF.
  mlir::OperationState opState{op->getLoc(), op.getOpName()};

  // Add the execute chain as the first argument.
  operands.insert(operands.begin(), loweringState.getExecuteChain());
  opState.addOperands(operands);

  // The return type is just the chain, other outputs are passed as argument by
  // their buffer pointers.
  opState.addTypes({loweringState.getExecuteChain().getType()});

  // Add it to the MLIR.
  mlir::Operation *invokeOp =
      loweringState.getExecuteOpBuilder().create(opState);
  Value outChain = invokeOp->getResult(0);
  // Downstream passes may need to distinguish input from output buffers.
  MGP::setNumBackOutputBuffers(invokeOp, numOutputBuffers);

  // Create the output tensors once we are sure the operation has finished
  // writing to them.
  helper.allocateOutputTensor(outChain, outputBufferAndSpec);

  return success();
}

// ConcatOp

LogicalResult ConcatOp::lower(MGP::MOGraphLoweringState &loweringState) {
  ConcatOp op = *this;
  LowerHelper helper{loweringState, op};
  Value execChain = loweringState.getExecuteChain();

  // Operands to pass to the operation. Input tensors are expanded to buffers
  // and specs in IR those are passed in.
  auto inputTensors = helper.getLoweredInputs();

  // Lower the outputs to buffers which can then be passed back in as
  // parameters.
  auto errOr = helper.lowerOutputBuffer("None", execChain, inputTensors);
  if (errOr.isError())
    return failure();
  SmallVector<LoweredTensor> outputBufferAndSpec = errOr.takeValue();

  SmallVector<Value> operands;
  for (auto tensor : inputTensors)
    operands.push_back(tensor.buffer);

  auto concatOp =
      loweringState.getExecuteOpBuilder().create<MGP::CpuBufferConcatOp>(
          op.getLoc(),
          /*inChain=*/loweringState.getExecuteChain(),
          /*outBuffer=*/outputBufferAndSpec[0].buffer,
          /*inBuffers=*/operands);

  // Make sure downstream passes recognise the first buffer arg as an output.
  // CAUTION: This is different from the mop output argument convention used
  // elsewhere here.
  MGP::setNumFrontOutputBuffers(concatOp, 1);

  helper.allocateOutputTensor(concatOp.getOutChain(), outputBufferAndSpec);
  return success();
}

// SplitOp

LogicalResult SplitOp::lower(MGP::MOGraphLoweringState &loweringState) {
  SplitOp op = *this;
  LowerHelper helper{loweringState, op};
  Value execChain = loweringState.getExecuteChain();
  OpBuilder execBuilder = loweringState.getExecuteOpBuilder();

  // Get lowered input tensor's buffer and spec
  LoweredTensor loweredInputTensor =
      helper.getLoweredTensorForValue(op.getInput());
  Value inputSpec = loweredInputTensor.spec;
  Value inputBuf = loweredInputTensor.buffer;

  // TODO Enable this.
  if (inputBuf.getDefiningOp<MGP::CpuBufferConstantOp>())
    return op.emitError("constant input buffer is unsupported");

  // Get lowered split size tensor value
  Value splitSizes =
      loweringState.lookupExecuteLoweredValue(op.getSplitSizes());

  // Get output tensor specs
  SmallVector<MO::TensorType> resultMoTypes =
      castToMoTensorTypes(op->getResultTypes());
  SmallVector<TypedValue<MGP::CpuTensorSpecType>> outSpecs =
      helper.invokeLegacyShapeFunc("mop.shape.split_axis_0",
                                   {inputSpec, splitSizes}, resultMoTypes);

  // Build/collect inputs to the MGP buffer split op
  Location loc = op.getLoc();
  MLIRContext *ctx = execBuilder.getContext();
  SmallVector<Type> outBufTypes;
  SmallVector<Value> runtimeSizes;
  SmallVector<uint64_t> rawSizes;
  for (TypedValue<MGP::CpuTensorSpecType> outSpec : outSpecs) {
    MGP::CpuTensorSpecType outSpecTy = outSpec.getType();
    std::optional<uint64_t> bufferSize = outSpecTy.getSizeInBytes();

    outBufTypes.emplace_back(MGP::CpuBufferType::get(ctx, bufferSize));
    rawSizes.emplace_back(optSizeToRawSize(bufferSize));
    if (!bufferSize.has_value()) {
      runtimeSizes.emplace_back(
          execBuilder.create<MGP::CpuTensorSpecSizeOp>(loc, outSpec));
    }
  }

  // Build MGP buffer split op
  auto mgpBufferSplitOp = execBuilder.create<MGP::CpuBufferSplitBorrowOp>(
      loc,
      /*outBuffers=*/outBufTypes,
      /*inChain=*/execChain,
      /*inBuffer=*/inputBuf,
      /*runtimeSizes=*/runtimeSizes,
      /*outBufferRawSizes=*/rawSizes);

  // Associate each result MO tensor with a new MGP tensor
  for (auto [outIdx, outSpec, outBuf] :
       llvm::enumerate(outSpecs, mgpBufferSplitOp.getResults())) {
    // Get the type of the spec and infer the type of the tensor from that.
    auto outSpecTy = cast<MGP::CpuTensorSpecType>(outSpec.getType());
    auto outTensorTy = MGP::CpuTensorType::get(ctx, outSpecTy);

    // Associate the result MO tensor with a new MGP tensor
    auto tensorCreateOp =
        execBuilder.create<MGP::CpuTensorCreateOp>(op.getLoc(),
                                                   /*result=*/outTensorTy,
                                                   /*inChain=*/execChain,
                                                   /*buffer=*/outBuf,
                                                   /*tensorSpec=*/outSpec);
    loweringState.emplaceExecuteLoweredValue(op.getResult(outIdx),
                                             tensorCreateOp.getResult());
  }

  return success();
}

LogicalResult LoadJitKernelOp::lower(MGP::MOGraphLoweringState &loweringState) {
  MOGG::LoadJitKernelOp kernel = *this;

  SmallVector<Value> operands{
      loweringState.getInitChain(),
      loweringState.getInitMgpContext(),
  };

  // MEF can't currently work with arrays of strings. To circumvent this
  // restriction we pass in the combined symbol as a string with a ";". This is
  // the setup time load so performance should be less of an issue (though at
  // runtime it is just string ref operations anyway).
  auto syms = kernel.getSymbols();
  std::string combined;

  size_t index = 0;
  // Combine into a list with a terminator character.
  for (StringRef symbol : syms.getAsValueRange<mlir::StringAttr>()) {
    combined += symbol.str() + ";";

    loweringState.trackJitKernel(symbol.str(), index);
    index++;
  }

  // Drop the last ";".
  combined.pop_back();
  auto newSymAttrs = loweringState.getInitOpBuilder().getStringAttr(combined);

  // Create a call to a custom invoke function we can put in the MEF.
  mlir::OperationState opState{kernel.getLoc(), "mop.load_jit_archive"};
  opState.addOperands(operands);

  // This will load and store the kernel at the given index into memory. The
  // address will be tracked in the symbol table and it can then be accessed by
  // refering to the index in the table. IMPORTANT! Order in MEF is alphabetical
  // by attribute name.
  opState.addAttribute("archiveData", kernel.getValue());
  opState.addAttribute("jitCtxSlot",
                       loweringState.getInitOpBuilder().getUI32IntegerAttr(
                           loweringState.getJitCtxSlot()));
  opState.addAttribute("kernelName", newSymAttrs);

  opState.addTypes({loweringState.getInitChain().getType()});

  // Kernel will be loaded by the context inside the JIT.
  mlir::Operation *kernelInGraph =
      loweringState.getInitOpBuilder().create(opState);
  loweringState.setInitChain(kernelInGraph->getResult(0));

  return success();
}

LogicalResult
ExecuteCompiledOp::lower(MGP::MOGraphLoweringState &loweringState) {
  ExecuteCompiledOp op = *this;
  LowerHelper helper{loweringState, op};

  // Operands to pass to the operation. Input tensors are expanded to buffers
  // and specs in IR those are passed in.
  auto inputTensors = helper.getLoweredInputs();

  // Allocate buffers to hold all the operation's result tensors.
  // Jit compiled kernels have no shape function and just use the shape infered
  // values directly and are guarded to only run on static or shape infered ops.
  auto errOr = helper.lowerOutputBuffer(
      "jit_compiled", loweringState.getExecuteChain(), inputTensors);
  if (errOr.isError())
    return failure();
  SmallVector<LoweredTensor> outputBufferAndSpec = errOr.takeValue();

  size_t numOutputBuffers = outputBufferAndSpec.size();
  size_t numBuffers = inputTensors.size() + outputBufferAndSpec.size();

  // We only take the input buffers, we don't need the specs as we just read the
  // used parameters directly.
  SmallVector<Value> operands;
  operands.reserve(numBuffers);

  // Collect the buffers from the input/output tensors.
  for (LoweredTensor &tensor : inputTensors)
    operands.push_back(tensor.buffer);
  for (LoweredTensor &tensor : outputBufferAndSpec)
    operands.push_back(tensor.buffer);

  // Add the used kernel parameters.
  ArrayAttr usedParams = op.getUsedParamsAttr();
  for (auto param : usedParams.getAsRange<KGEN::ParamDeclRefAttr>()) {
    auto symbolicDimOr = loweringState.lowerExpression(op->getLoc(), param);
    if (symbolicDimOr.isError())
      return failure();

    operands.push_back(symbolicDimOr.takeValue());
  }

  SmallVector<Type> returnTypes;
  returnTypes.push_back(loweringState.getExecuteChain().getType());

  // Add each output parameter as an index return type.
  for (size_t i = 0, e = getOutputParamDecls().size(); i < e; ++i)
    returnTypes.push_back(loweringState.getExecuteOpBuilder().getIndexType());

  // Create a call to a custom invoke function we can put in the MEF.
  mlir::OperationState opState{op->getLoc(), "mop.kgen.execute_kernel"};

  uint32_t id = loweringState.getJitKernelId(op.getKernelNameAttr().str());

  // For compiled ops add the context as argument and all the metadata needed to
  // extract a compiled kernel from the context.
  operands.insert(operands.begin(), loweringState.getExecuteMgpContext());
  opState.addAttribute("jitCtxSlot",
                       loweringState.getExecuteOpBuilder().getUI32IntegerAttr(
                           loweringState.getJitCtxSlot()));
  opState.addAttribute(
      "kernelIndex",
      loweringState.getExecuteOpBuilder().getUI32IntegerAttr(id));
  opState.addAttribute("kernelName", op.getKernelNameAttr());

  // Tell the primitive how many of the arguments are buffers, and how many
  // of those are output buffers, so it knows how to interpret its operands.
  opState.addAttribute(
      "numBufferArgs",
      loweringState.getExecuteOpBuilder().getUI32IntegerAttr(numBuffers));
  opState.addAttribute(
      "numOutputBufferArgs",
      loweringState.getExecuteOpBuilder().getUI32IntegerAttr(numOutputBuffers));

  // The operation's execution will be gated by a chain. As for output
  // allocation above, at this stage use the 'execute' block's chain.
  // The PeepholeTensorExtract will change this.
  operands.insert(operands.begin(), loweringState.getExecuteChain());
  opState.addOperands(operands);

  // The return type is the chain and any output parameters.
  opState.addTypes(returnTypes);

  // Add it to the MLIR.
  mlir::Operation *invokeOp =
      loweringState.getExecuteOpBuilder().create(opState);
  Value outChain = invokeOp->getResult(0);
  // Downstream passes may need to distinguish input from output buffers.
  MGP::setNumBackOutputBuffers(invokeOp, numOutputBuffers);

  KGEN::ParamDeclArrayAttr params = op.getOutputParamDeclsAttr();

  // Each of the index output now map onto a concrete parameter value.
  for (size_t i = 1, e = invokeOp->getNumResults(); i < e; ++i) {
    if (failed(loweringState.lowerParamDecl(params[i - 1],
                                            invokeOp->getResult(i))))
      return failure();
  }

  // Create the output tensors once we are sure the operation has finished
  // writing to them.
  helper.allocateOutputTensor(outChain, outputBufferAndSpec);

  return success();
}

//===----------------------------------------------------------------------===//
// Parameter printing/parsing support in form: <P1, P2, P3>
//===----------------------------------------------------------------------===//

static ParseResult parseParamDecls(OpAsmParser &p,
                                   KGEN::ParamDeclArrayAttr &paramDecls) {
  if (succeeded(p.parseOptionalLess()) && failed(p.parseOptionalGreater())) {
    if (KGEN::parseParamDecls(p, paramDecls) || p.parseGreater())
      return failure();
  } else {
    paramDecls = KGEN::ParamDeclArrayAttr::get(p.getContext(), {});
  }
  return success();
}

static void printParamDecls(OpAsmPrinter &p, Operation *,
                            KGEN::ParamDeclArrayAttr paramDecls) {
  if (!paramDecls.empty()) {
    p << "<";
    printParamDecls(p, paramDecls);
    p << '>';
  }
}

static ParseResult parseLambda(OpAsmParser &p, StringAttr &name, TypeAttr &func,
                               Region &region) {
  SmallVector<OpAsmParser::Argument> args;
  llvm::SMLoc loc;
  KGEN::SignatureType sigType;
  FunctionType functionType;
  KGEN::ParamDeclArrayAttr inputParams, resultParams;

  llvm::SMLoc sigLoc;
  if (p.parseSymbolName(name) || p.parseEqual() ||
      p.getCurrentLocation(&sigLoc) ||
      KGEN::parseFunctionSignature(p, args, inputParams, resultParams,
                                   functionType, sigType) ||
      p.getCurrentLocation(&loc) || p.parseRegion(region, args))
    return failure();
  if (!inputParams.empty() || !resultParams.empty())
    return p.emitError(sigLoc, "unexpected input or result parameters");

  func = TypeAttr::get(functionType);
  return success();
}

static void printLambda(OpAsmPrinter &p, mlir::Operation *lambda,
                        StringAttr name, TypeAttr sig, Region &region) {
  p.printSymbolName(name.strref());
  p << " = ";

  auto func = cast<FunctionType>(sig.getValue());

  auto sigTy = KGEN::SignatureType::get(lambda->getContext(), func.getInputs(),
                                        func.getResults());
  KGEN::printFunctionSignature(p, region, {}, {}, func, sigTy);
  p.printRegion(region, /*printEntryBlockArgs=*/false);
}

//===----------------------------------------------------------------------===//
// MOGG CallOp
//===----------------------------------------------------------------------===//

bool CallOp::setKGENParamToValueIfPresent(StringRef name, Attribute value) {
  // Exit out if we don't have the parameter already (we can't add new ones as
  // the params come from mojo).
  std::optional<mlir::DictionaryAttr> dict = getKgenParams();
  if (!dict)
    return false;

  mlir::NamedAttrList attributes(*dict);
  Attribute oldAttr = attributes.set(name, value);

  // Update the dict if it doesn't already have that value.
  if (oldAttr != value)
    setKgenParamsAttr(attributes.getDictionary(getContext()));

  return true;
}

//===----------------------------------------------------------------------===//
// MOGG KernelOp
//===----------------------------------------------------------------------===//

OutputPlaceholdersOp KernelOp::getOutputPlaceholder() {
  OutputPlaceholdersOp out = nullptr;
  for (auto ph : this->getOps<MOGG::OutputPlaceholdersOp>()) {
    assert(!out && "Only one placeholder expected");
    out = ph;
  }
  return out;
}

bool KernelOp::inlineLambdas() {
  KernelOp kernel = *this;
  bool changed = false;
  mlir::SymbolTable symtable{kernel};

  // Inline lambdas.
  kernel.walk([&](MOGG::LambdaInvokeOp invoke) {
    auto name = invoke.getLambda().getLeafReference().strref();
    auto lambda = symtable.lookup<MOGG::LambdaOp>(name);

    if (lambda) {
      inlineInvoke(invoke, lambda);
      invoke.erase();
      changed = true;
    }
  });

  // Remove any now unused lambdas.
  kernel.walk([&](MOGG::LambdaOp lambda) {
    if (mlir::SymbolTable::symbolKnownUseEmpty(lambda, kernel)) {
      lambda.erase();
      changed = true;
    }
  });

  // If we touched any of the invokes we must disable fusion on the inputs &
  // outputs as we will have removed the input/output lambda.
  if (changed) {
    kernel.removeInputFusionInterfaceAttr();
    kernel.removeOutputFusionInterfaceAttr();
  }

  return changed;
}

/// Disables future fusion on this input.
void KernelOp::disableInputFusionOnInput(size_t index) {
  std::optional<ArrayAttr> inputLambdas = getInputFusionInterfaceAttr();

  if (inputLambdas.has_value()) {
    SmallVector<Attribute> lambdaNames;

    // Add all the old attributes to the new one but remove the attribute at the
    // disabled index.
    size_t i = 0;
    for (StringAttr attr : inputLambdas->getAsRange<StringAttr>()) {
      if (i == index) {
        lambdaNames.push_back(StringAttr::get(getContext(), ""));
      } else {
        lambdaNames.push_back(attr);
      }

      i++;
    }

    // Update the attribute.
    setInputFusionInterfaceAttr(ArrayAttr::get(getContext(), lambdaNames));
  }
}

/// Removes an operand at the given index and disables input fusion on that
/// position if this kernel otherwise supports it.
void KernelOp::removeOperandAtIndex(size_t index) {
  (*this)->eraseOperand(index);
  getBody().front().eraseArgument(index);

  std::optional<ArrayAttr> inputInterfaces = getInputFusionInterface();

  // Update the input fusion interfaces if we have them to remove fusion on this
  // attr.
  if (inputInterfaces.has_value()) {
    SmallVector<Attribute> lambdaNames;
    size_t i = 0;
    // Add each of the current input hooks to the list, skipping the input we
    // just removed.
    for (StringAttr name : inputInterfaces->getAsRange<StringAttr>()) {
      if (i != index)
        lambdaNames.push_back(name);
      i++;
    }
    setInputFusionInterfaceAttr(ArrayAttr::get(getContext(), lambdaNames));
  }
}

/// Add an operand to the kernel and adds an argument to the kernel region.
/// Fixes up the input fusion interfaces to disallow input fusion on this
/// position. Returns the added kernel argument.
Value KernelOp::addOperand(Value operand) {
  (*this)->insertOperands(getNumOperands(), ValueRange{operand});
  Value blockArg =
      getBody().front().addArgument(operand.getType(), operand.getLoc());

  std::optional<ArrayAttr> inputInterfaces = getInputFusionInterface();

  // Update the input fusion interfaces if we have them to remove fusion on this
  // attr.
  if (inputInterfaces.has_value()) {
    SmallVector<Attribute> lambdaNames;
    for (StringAttr name : inputInterfaces->getAsRange<StringAttr>())
      lambdaNames.push_back(name);
    // Add an empty lambda name for the operand we just added.
    lambdaNames.push_back(StringAttr::get(getContext(), ""));
    setInputFusionInterfaceAttr(ArrayAttr::get(getContext(), lambdaNames));
  }

  return blockArg;
}

void KernelOp::setInputFusionInterface(size_t index, StringRef lambdaName) {
  std::optional<ArrayAttr> inputInterfaces = getInputFusionInterface();

  // Update the input fusion interfaces if we have them to remove fusion on this
  // attr.
  if (inputInterfaces.has_value()) {
    SmallVector<Attribute> lambdaNames;
    for (StringAttr name : inputInterfaces->getAsRange<StringAttr>()) {
      if (lambdaNames.size() == index)
        lambdaNames.push_back(StringAttr::get(getContext(), lambdaName));
      else
        lambdaNames.push_back(name);
    }
    setInputFusionInterfaceAttr(ArrayAttr::get(getContext(), lambdaNames));
  }
}

// Fuse the body of this kernel into another.
void KernelOp::fuseBodyIntoNew(
    OpBuilder &builder, mlir::SymbolTable &symTable, IRMapping &remapper,
    llvm::StringMap<StringRef> &oldLambdaNamesToNew) {
  auto neverSkip = [](Operation *) { return false; };

  fuseBodyIntoNewWithSkip(builder, symTable, remapper, oldLambdaNamesToNew,
                          neverSkip);
}

// Fuse the body of this kernel into another.
void KernelOp::fuseBodyIntoNewWithSkip(
    OpBuilder &builder, mlir::SymbolTable &symTable, IRMapping &remapper,
    llvm::StringMap<StringRef> &oldLambdaNamesToNew,
    std::function<bool(Operation *)> shouldSkip) {

  // Clone the incoming kernel into this kernel.
  for (Operation &op : getBody().front()) {
    if (shouldSkip(&op))
      continue;

    if (isa<MOGG::OutputPlaceholdersOp>(op) || isa<MOGG::OutputOp>(op)) {
      // Skip placeholders & output, already remapped.
    } else if (auto lambda = dyn_cast<LambdaOp>(op)) {
      // Add lambdas seperately so we rename them on insertion. (The
      // symbol table does this for us).
      auto newLambda = cast<LambdaOp>(builder.clone(op, remapper));
      symTable.insert(newLambda, builder.getInsertionPoint());
      builder.setInsertionPointAfter(newLambda);

      oldLambdaNamesToNew[lambda.getSymName()] = newLambda.getSymName();
    } else if (auto call = dyn_cast<CallOp>(op)) {
      // Lambdas will have been inlined. The only place we can have
      // a lambda reference is in a call.
      auto newCall = dyn_cast<CallOp>(builder.clone(op, remapper));

      SmallVector<Attribute> newAttrs;

      // Remap any lambda references to point to the new lambda name.
      std::optional<ArrayAttr> params = newCall.getParams();
      if (params) {
        for (SymbolRefAttr sym : params->getAsRange<SymbolRefAttr>()) {
          StringRef name = sym.getLeafReference().strref();
          auto itr = oldLambdaNamesToNew.find(name);
          assert(itr != oldLambdaNamesToNew.end() &&
                 "First use of lambda inserted before the lambda itself");
          newAttrs.push_back(FlatSymbolRefAttr::get(getContext(), itr->second));
        }
        newCall.setParamsAttr(builder.getArrayAttr(newAttrs));
      }
    } else {
      builder.clone(op, remapper);
    }
  }
}

LogicalResult KernelOp::verify() {
  auto inputInterfaces = getInputFusionInterface();
  auto outputInterfaces = getOutputFusionInterface();

  if (inputInterfaces.has_value() &&
      getNumOperands() != inputInterfaces->size()) {
    return emitError("Number of input fusion interfaces does not match number "
                     "of inputs")
           << " (" << inputInterfaces->size() << " vs " << getNumOperands()
           << ")";
  }

  if (outputInterfaces.has_value() &&
      getNumResults() != outputInterfaces->size()) {
    return emitError("Number of output fusion interfaces does not match number "
                     "of results")
           << " (" << outputInterfaces->size() << " vs " << getNumResults()
           << ")";
  }

  if (getBody().empty()) {
    return emitError("Region cannot be empty");
  }

  uint64_t numArguments =
      std::distance(getBody().args_begin(), getBody().args_end());
  if (getNumOperands() != numArguments) {
    return emitError(
               "Number of operands does not match number of body arguments ")
           << " (" << getNumOperands() << " vs " << numArguments << ")";
  }

  return success();
}

static inline bool allNullFusionInterface(ArrayAttr arr) {
  auto strArr = arr.getAsRange<StringAttr>();
  return std::all_of(strArr.begin(), strArr.end(), [](StringAttr strAttr) {
    return strAttr.strref().empty();
  });
}

LogicalResult KernelOp::canonicalize(KernelOp op, PatternRewriter &) {
  if (op.getInputFusionInterface().has_value() &&
      allNullFusionInterface(op.getInputFusionInterfaceAttr())) {
    op.removeInputFusionInterfaceAttr();
  }

  if (op.getOutputFusionInterface().has_value() &&
      allNullFusionInterface(op.getOutputFusionInterfaceAttr())) {
    op.removeOutputFusionInterfaceAttr();
  }

  return success();
}

static ParseResult parseKernel(
    OpAsmParser &p, SmallVectorImpl<OpAsmParser::UnresolvedOperand> &operands,
    SmallVectorImpl<Type> &operandTypes, SmallVectorImpl<Type> &resultTypes,
    Region &region, KGEN::ParamDeclArrayAttr &resultParams) {
  SmallVector<OpAsmParser::Argument> args;
  llvm::SMLoc loc;
  KGEN::SignatureType sigType;
  FunctionType functionType;
  KGEN::ParamDeclArrayAttr inputParams, dummyRes;

  llvm::SMLoc sigLoc;
  if (parseParamDecls(p, resultParams) ||
      p.parseOperandList(operands, AsmParser::Delimiter::Paren) ||
      p.parseColon() || p.getCurrentLocation(&sigLoc) ||
      KGEN::parseFunctionSignature(p, args, inputParams, dummyRes, functionType,
                                   sigType) ||
      p.getCurrentLocation(&loc) || p.parseRegion(region, args))
    return failure();
  if (!inputParams.empty() || !dummyRes.empty())
    return p.emitError(sigLoc, "unexpected input or result parameters");

  for (const OpAsmParser::Argument &arg : args)
    operandTypes.push_back(arg.type);

  for (Type res : functionType.getResults())
    resultTypes.push_back(res);

  return success();
}

static void printKernel(OpAsmPrinter &p, mlir::Operation *kernel,
                        ValueRange operands, TypeRange operandTypes,
                        TypeRange resultTypes, Region &region,
                        KGEN::ParamDeclArrayAttr resultParams) {
  printParamDecls(p, kernel, resultParams);
  p << '(';
  p.printOperands(operands);
  p << ") : ";

  auto functionType =
      FunctionType::get(kernel->getContext(), operandTypes, resultTypes);

  KGEN::printFunctionSignature(p, region, {}, {}, functionType,
                               SignatureType::get(functionType));
  p.printRegion(region, /*printEntryBlockArgs=*/false);
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

void MOGGDialect::registerOperations() {
  addOperations<
#define GET_OP_LIST
#include "GenericML/GraphCompiler/MOGGDialect/MOGG.cpp.inc"
      >();
}

#define GET_OP_CLASSES
#include "GenericML/GraphCompiler/MOGGDialect/MOGG.cpp.inc"
