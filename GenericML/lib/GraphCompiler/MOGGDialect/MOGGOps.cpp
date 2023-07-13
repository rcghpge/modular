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

  // Elementwise shapes are guaranteed to match.
  static const llvm::SmallDenseSet<StringRef, 8> elementwise{
      "mop.kernels.mul",     "mop.kernels.linalg.band_part",
      "mop.kernels.sigmoid", "mop.kernels.softmax",
      "mop.kernels.sqrt",    "mop.kernels.sub",
      "mop.kernels.tanh",
  };

  // (arg0Spec, arg1Spec, ..., argNSpec) -> outSpec
  static const llvm::StringMap<std::string> specNaryShapeFuncs{
      {"mop.kernels.matmul", "mop.shape.matmul"},
      {"mop.kernels.matmul_transpb", "mop.shape.matmul_transpb"},
      {"mop.kernels.pack_matmul_b", "mop.shape.pack_matmul_b"},
      {"mop.kernels.pack_matmul_b.transp_b",
       "mop.shape.pack_matmul_b.transp_b"},
      {"mop.kernels.fully_connected", "mop.shape.fully_connected"},
      {"mop.kernels.batch_matmul", "mop.shape.batch_matmul"},
      {"mop.kernels.batch_matmul_adjb", "mop.shape.batch_matmul_adjb"},
      {"mop.kernels.concat_axis_0", "mop.shape.concat_axis_0"},
  };

  // (arg0Spec, arg1) -> outSpec
  static const llvm::StringMap<std::string> specAndTensorBinaryShapeFuncs{
      {"mop.kernels.arg_max", "mop.shape.arg_reduce"},
      {"mop.kernels.broadcast_to", "mop.shape.broadcast_to"},
      {"mop.kernels.transpose", "mop.shape.transpose"},
      {"mop.kernels.pad", "mop.shape.pad"},
      {"mop.kernels.mean", "mop.shape.reduce"},
      {"mop.kernels.reduce.add", "mop.shape.reduce"},
      {"mop.kernels.reduce.mul", "mop.shape.reduce"},
      {"mop.kernels.reduce.min", "mop.shape.reduce"},
      {"mop.kernels.reduce.max", "mop.shape.reduce"},
  };

  OpBuilder exeBuilder = loweringState.getExecuteOpBuilder();
  auto getOutputSpec = [&](StringRef opName, ArrayRef<Value> inputs) {
    return invokeLegacyShapeFunc(opName, inputs, {outSpecType}).front();
  };

  LoweredTensor output;

  // Hackery! Pure Hackery! I'd start setting up classes for this, but
  // again, I don't expect this to survive E
  if (elementwise.contains(opName)) {
    // All elementwise kernels are guaranteed to have at least one argument
    // which is guaranteed to match the output.
    output.spec = inputTensors[0].spec;
  } else if (!moType.hasUnknownShape() && !moType.hasStaticShape()) {
    // We get the values corresponding to dynamic dimensions.
    UNWRAP_ERROR(runtimeDims, loweringState.lowerShapeAttrDynDims(
                                  moType.getTypedShapeAttr(), loc));
    output.spec =
        loweringState.getExecuteOpBuilder()
            .create<MGP::CpuTensorSpecCreateOp>(loc, outSpecType, runtimeDims)
            .getResult();
  } else if (specNaryShapeFuncs.count(opName)) {
    // Get just the specs for tensors.
    SmallVector<Value> inputSpecs = llvm::map_to_vector(
        inputTensors, [](const LoweredTensor &t) -> Value { return t.spec; });

    // Create a call to a custom invoke function we can put in the MEF.
    output.spec = getOutputSpec(specNaryShapeFuncs.lookup(opName), inputSpecs);
  } else if (specAndTensorBinaryShapeFuncs.count(opName)) {
    TypedValue<MGP::CpuTensorSpecType> arg0SpecVal = inputTensors[0].spec;
    Value arg1 = opBeingLowered->getOperand(1);
    Value arg1Val = loweringState.lookupExecuteLoweredValue(arg1);
    output.spec = getOutputSpec(specAndTensorBinaryShapeFuncs.lookup(opName),
                                {arg0SpecVal, arg1Val});

  } else if (opName == "mop.kernels.gather") {
    Value target_axis = loweringState.lookupExecuteLoweredValue(
        opBeingLowered->getOperands()[2]);
    output.spec =
        getOutputSpec("mop.shape.gather", {inputTensors[0].spec,
                                           inputTensors[1].spec, target_axis});
  } else if (opName == "mop.kernels.concat") {
    // Ignore axis tensor at the end
    SmallVector<Value> shapeFuncArgs;
    shapeFuncArgs.reserve(inputTensors.size() - 1);
    for (size_t i = 1, e = inputTensors.size(); i < e; ++i)
      shapeFuncArgs.push_back(inputTensors[i].spec);

    // Put the axis last.
    Value axis = loweringState.lookupExecuteLoweredValue(
        opBeingLowered->getOperands().front());
    shapeFuncArgs.push_back(axis);
    output.spec = getOutputSpec("mop.shape.concat", shapeFuncArgs);
  } else if (opName == "mop.kernels.gather_sum") {
    // Hackery! Pure Hackery! I'd start setting up classes for this, but
    // again, I don't expect this to survive E
    Value gather_axis = loweringState.lookupExecuteLoweredValue(
        opBeingLowered->getOperands()[2]);
    Value reduce_axis = loweringState.lookupExecuteLoweredValue(
        opBeingLowered->getOperands()[3]);
    output.spec = getOutputSpec(
        "mop.shape.gather_sum",
        {inputTensors[0].spec, inputTensors[1].spec, gather_axis, reduce_axis});
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
  } else if (opName == "mop.kernels.range") {
    Value start = loweringState.lookupExecuteLoweredValue(
        opBeingLowered->getOperands()[0]);
    Value limit = loweringState.lookupExecuteLoweredValue(
        opBeingLowered->getOperands()[1]);
    Value step = loweringState.lookupExecuteLoweredValue(
        opBeingLowered->getOperands()[2]);
    output.spec = getOutputSpec("mop.shape.range", {start, limit, step});
  } else if (opName == "mop.kernels.where") {
    Value input = loweringState.lookupExecuteLoweredValue(
        opBeingLowered->getOperands()[0]);
    output.spec = getOutputSpec("mop.shape.where", {input});
  } else if (opName == "mop.kernels.slice") {
    Value input = inputTensors[0].spec;
    Value start = loweringState.lookupExecuteLoweredValue(
        opBeingLowered->getOperands()[1]);
    Value stop = loweringState.lookupExecuteLoweredValue(
        opBeingLowered->getOperands()[2]);
    Value step = loweringState.lookupExecuteLoweredValue(
        opBeingLowered->getOperands()[3]);

    output.spec =
        getOutputSpec("mop.shape.slice", {loweringState.getExecuteChain(),
                                          input, start, stop, step});
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
  if (multiOutputShapeFuncs.contains(opName) && !resultMoTypes.empty() &&
      !resultMoTypes.front().hasStaticShape()) {
    return lowerMultiOutputShapeFunc(multiOutputShapeFuncs.lookup(opName),
                                     inChain, inputTensors, resultMoTypes);
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
  auto errOr = helper.lowerOutputBuffer("mop.kernels.concat_axis_0", execChain,
                                        inputTensors);
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
  opState.addAttribute(
      "kernelIndex",
      loweringState.getExecuteOpBuilder().getUI32IntegerAttr(id));
  opState.addAttribute("kernelName", op.getKernelNameAttr());
  opState.addAttribute("jitCtxSlot",
                       loweringState.getExecuteOpBuilder().getUI32IntegerAttr(
                           loweringState.getJitCtxSlot()));

  // Tell the primitive how many of the arguments are buffers so it knows when
  // to stop reading.
  opState.addAttribute(
      "numBufferArgs",
      loweringState.getExecuteOpBuilder().getUI32IntegerAttr(numBuffers));

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
