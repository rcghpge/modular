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
#include "GenericML/MGPDialect/MGPOps.h"
#include "GenericML/MGPDialect/MGPUtils.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "Support/STLExtras.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/IR/FunctionImplementation.h"
#include "llvm/ADT/StringMap.h"
#include <llvm/ADT/STLExtras.h>

using namespace M;
using namespace MOGG;

/*
 * Lowering helpers.
 */
namespace {

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

  // Lower each of the inputs in the operation to mgp buffers & specs.
  SmallVector<LoweredTensor> lowerInputs();

  // Lower the output buffer which will get passed into the input.
  ErrorOr<SmallVector<LoweredTensor>> lowerOutputBuffer(
      llvm::StringRef opName, Value inChain,
      const SmallVector<LoweredTensor> &inputTensors,
      std::optional<SmallVector<int32_t, 2>> bcastIndices = std::nullopt,
      std::optional<IntArrayElementsAttr> matmulOriginalBDims = std::nullopt);

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
                     const SmallVector<LoweredTensor> &inputTensors,
                     std::optional<SmallVector<int32_t, 2>> bcastIndices,
                     std::optional<IntArrayElementsAttr> matmulOriginalBDims);
};

SmallVector<LoweredTensor> LowerHelper::lowerInputs() {
  SmallVector<LoweredTensor> inputs;
  // Allocate space for the buffers + specs per tensor.
  inputs.reserve(opBeingLowered->getNumOperands());

  for (Value operand : opBeingLowered->getOperands()) {
    // Extract the buffer and spec from the MGP tensor.
    mlir::Value mgpTensor = loweringState.lookupExecuteLoweredValue(operand);

    auto extract =
        loweringState.getExecuteOpBuilder().create<MGP::CpuTensorExtractOp>(
            loc, mgpTensor);

    // Add the expanded buffer and spec as arguments to the operation.
    inputs.push_back(
        {extract.getBufferResult(), extract.getTensorSpecResult()});
  }

  return inputs;
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
    const SmallVector<LoweredTensor> &inputTensors,
    std::optional<SmallVector<int32_t, 2>> bcastIndices,
    std::optional<IntArrayElementsAttr> matmulOriginalBDims) {
  // Dynamic Case
  // TODO: This is hacky, hacky, hacky, remove as soon as we can lower
  // parametric shapes to the runtime
  // TODO: If we decide we want this longer term (say, for MLAS kernels)
  // We should refactor it out as a Trait maybe? and make this a method
  // and data member of the op class.

  // Elementwise shapes are guaranteed to match.
  static const llvm::SmallDenseSet<StringRef, 32> elementwise{
      "mop.kernels.abs",     "mop.kernels.add",
      "mop.kernels.div",     "mop.kernels.erf",
      "mop.kernels.exp",     "mop.kernels.gelu",
      "mop.kernels.max",     "mop.kernels.min",
      "mop.kernels.mul",     "mop.kernels.matrix_band_part",
      "mop.kernels.pow",     "mop.kernels.relu",
      "mop.kernels.rsqrt",   "mop.kernels.sigmoid",
      "mop.kernels.softmax", "mop.kernels.sqrt",
      "mop.kernels.sub",     "mop.kernels.tanh",
  };

  static const llvm::SmallDenseSet<StringRef, 8> elementwise_conditional{
      "mop.kernels.equal",
      "mop.kernels.greater",
      "mop.kernels.greater_equal",
      "mop.kernels.not_equal",
  };

  // (arg0Spec, arg1Spec, ..., argNSpec) -> outSpec
  static const llvm::StringMap<std::string> specNaryShapeFuncs{
      {"mop.kernels.matmul", "mop.shape.matmul"},
      {"mop.kernels.matmul_transpb", "mop.shape.matmul_transpb"},
      {"mop.kernels.pack_matmul_b", "mop.shape.pack_matmul_b"},
      {"mop.kernels.fully_connected", "mop.shape.fully_connected"},
      {"mop.kernels.batch_matmul", "mop.shape.batch_matmul"},
      {"mop.kernels.batch_matmul_adjb", "mop.shape.batch_matmul_adjb"},
      {"mop.kernels.select", "mop.shape.select"},
      {"mop.kernels.in_place_concat", "mop.shape.in_place_concat"},
  };

  // (arg0Spec, arg1) -> outSpec
  static const llvm::StringMap<std::string> specAndTensorBinaryShapeFuncs{
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
    mlir::OperationState opState{loc, opName};
    // inputs are input specs
    opState.addOperands(inputs);
    // The return is the new spec
    opState.addTypes(cast<mlir::Type>(outSpecType));
    // TODO(#11821) remove once shape inference is on by default
    if (matmulOriginalBDims.has_value()) {
      mlir::OperationName extOpName{ExtOp::getOperationName(),
                                    opState.getContext()};
      opState.addAttribute(ExtOp::getOriginalBDimsAttrName(extOpName),
                           matmulOriginalBDims.value());
    }

    // Add it to the MLIR.
    mlir::Operation *invokeOp = exeBuilder.create(opState);
    return cast<TypedValue<MGP::CpuTensorSpecType>>(invokeOp->getResult(0));
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
  } else if (elementwise_conditional.contains(opName)) {
    // Get the shape of the first Tensor to compute the output spec
    output.spec =
        getOutputSpec("mop.shape.conditional", {inputTensors[0].spec});
  } else if (specNaryShapeFuncs.count(opName)) {
    // Get just the specs for tensors.
    SmallVector<Value> inputSpecs = map_to_vector(
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
  } else if (opName == "mop.kernels.cast") {
    auto outType = opBeingLowered->getResultTypes()[0].cast<MO::TensorType>();
    // Create a fake spec value to use with the shape function. We only need the
    // dtype, so make it a scalar.
    // TODO: Use a unit const instead? Are shape functions compatible with it?
    auto castScalarType = MO::TensorType::get(
        MO::ShapeAttr::get(SmallVector<int64_t>{},
                           MO::ShapeType::get(exeBuilder.getContext())),
        outType.getElementType());
    auto castScalarSpec = exeBuilder.create<MGP::CpuTensorSpecCreateOp>(
        opBeingLowered->getLoc(), MGP::createCpuTensorSpec(castScalarType),
        SmallVector<Value>{});
    output.spec =
        getOutputSpec("mop.shape.cast", {inputTensors[0].spec, castScalarSpec});
  } else if (opName == "mop.kernels.concat") {
    // Ignore axis tensor at the end
    SmallVector<Value> shapeFuncArgs = map_to_vector(
        llvm::drop_end(inputTensors),
        [](LoweredTensor tensor) -> Value { return tensor.spec; });
    Value axis = loweringState.lookupExecuteLoweredValue(
        opBeingLowered->getOperands().back());
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
  } else if (opName == "mop.kernels.broadcast_to_tensor") {
    assert(bcastIndices && "Broadcast kernel doesn't have indices");

    // Broadcast kernels or operations which have fused a broadcast are supposed
    // to provide an indices attribute which we can then use to tell the shape
    // function which two inputs should be broadcasted together to produce the
    // final shape.
    SmallVector<Value> inputSpecs{
        inputTensors[(*bcastIndices)[0]].spec,
        inputTensors[(*bcastIndices)[1]].spec,
    };

    output.spec = getOutputSpec("mop.shape.broadcast_to_tensor", inputSpecs);
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
  } else if (opName == "mop.kernels.matmul_packb" ||
             opName.startswith("mop.kernels.fully_connected.packb")) {
    // Get just the specs for tensors.
    Value lhsSpec = inputTensors.front().spec;

    // Create a call to a custom invoke function we can put in the MEF.
    output.spec = getOutputSpec("mop.shape.matmul_packb", {lhsSpec});
  } else {
    opBeingLowered->emitError("op requires output to have static shape");
    return Error("op requires output to have static shape");
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

ErrorOr<SmallVector<LoweredTensor>> LowerHelper::lowerOutputBuffer(
    llvm::StringRef opName, Value inChain,
    const SmallVector<LoweredTensor> &inputTensors,
    std::optional<SmallVector<int32_t, 2>> bcastIndices,
    std::optional<IntArrayElementsAttr> matmulOriginalBDims) {

  SmallVector<LoweredTensor> outputBufferAndSpec;

  // Allocate the output buffer for each result and add as argument.
  for (Type t : opBeingLowered->getResultTypes()) {
    auto ty = t.cast<MO::TensorType>();
    if (!ty.hasRank())
      return Error("dynamic output rank not supported");
    MGP::CpuTensorSpecType outSpecType = MGP::createCpuTensorSpec(ty);
    Optional<uint64_t> bufferSize = outSpecType.getSizeInBytes();
    LoweredTensor output;
    if (!bufferSize) {
      UNWRAP_ERROR(out, lowerDynamicOutput(outSpecType, ty, opName, inChain,
                                           inputTensors, bcastIndices,
                                           matmulOriginalBDims));
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

} // namespace

void ExtOp::build(OpBuilder &builder, OperationState &state,
                  TypeRange resultTypes, StringRef opName, ValueRange operands,
                  ArrayRef<uint64_t> originalBDims) {
  MLIRContext *ctx = builder.getContext();
  mlir::OperationName extOpName{ExtOp::getOperationName(), ctx};
  auto originalBDimsAttr = IntArrayElementsAttr::get(
      ctx, originalBDims, IntegerType::SignednessSemantics::Unsigned);
  build(builder, state, resultTypes, opName, operands, originalBDimsAttr);
}

void ExtOp::build(OpBuilder &builder, OperationState &state,
                  TypeRange resultTypes, StringRef opName,
                  ValueRange operands) {
  IntArrayElementsAttr nullAttr;
  build(builder, state, resultTypes, opName, operands, nullAttr);
}

LogicalResult ExtOp::lower(MGP::MOGraphLoweringState &loweringState) {
  ExtOp op = *this;
  LowerHelper helper{loweringState, op};

  // Operands to pass to the operation. Input tensors are expanded to buffers
  // and specs in IR those are passed in.
  auto inputTensors = helper.lowerInputs();

  // Ext ops carry the lowered operation name around with them,
  // others might do something else.
  StringRef opName = op.getOpName();

  // TODO(#11821) remove once shape inference is on by default
  std::optional<IntArrayElementsAttr> matmulOriginalBDims = std::nullopt;
  if (auto originalBDimsAttr = getOriginalBDimsAttr())
    matmulOriginalBDims = originalBDimsAttr;

  // Lower the outputs to buffers which can then be passed back in as
  // parameters.
  auto errOr = helper.lowerOutputBuffer(
      opName, loweringState.getExecuteChain(), inputTensors,
      /*bcastIndices=*/std::nullopt, matmulOriginalBDims);
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

// InPlaceConcatOp

static void printParamDecls(OpAsmPrinter &p, Operation *,
                            KGEN::ParamDeclArrayAttr paramDecls) {
  if (!paramDecls.empty()) {
    p << "<() -> ";
    printParamDecls(p, paramDecls);
    p << '>';
  }
}

LogicalResult InPlaceConcatOp::lower(MGP::MOGraphLoweringState &loweringState) {
  InPlaceConcatOp op = *this;
  LowerHelper helper{loweringState, op};

  // Operands to pass to the operation. Input tensors are expanded to buffers
  // and specs in IR those are passed in.
  auto inputTensors = helper.lowerInputs();

  // Lower the outputs to buffers which can then be passed back in as
  // parameters.
  auto errOr = helper.lowerOutputBuffer(
      "mop.kernels.in_place_concat", loweringState.getExecuteChain(),
      inputTensors,
      /*bcastIndices=*/std::nullopt, /*matmulOriginalBDims=*/std::nullopt);
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

  // Combine into a list with a terminator character.
  for (StringRef symbol : syms.getAsValueRange<mlir::StringAttr>())
    combined += symbol.str() + ";";

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
  auto inputTensors = helper.lowerInputs();

  // TODO: We carry around the original operation name just to trick the current
  // dynamic shape logic.
  std::string originalName = "mop.kernels." + op.getOriginalOpName().str();

  std::optional<SmallVector<int32_t, 2>> bcastIndices = std::nullopt;

  // If the operation is a broadcast or is a fused kernel containing a broadcast
  // the indices of the inputs to that broadcast will be listed. These are
  // needed to tell the shape function how to compute the output shape.
  // Specifically the tell the shape inference which of the inputs represent the
  // broadcast inputs in a multi input fused kernel. If there are many
  // broadcasts we only look at one because we know that they are compatable
  // shapes as per fusion rules (we fuse on elementwise kernels which only
  // consume comptable shapes and only fuse the immediate operands so we can
  // guarantee shape compatibility).
  if (op.getBcastInputIndices()) {
    mlir::ArrayAttr indices = *op.getBcastInputIndices();

    SmallVector<int32_t, 2> newBcastIndices;
    for (llvm::APInt index : indices.getAsValueRange<mlir::IntegerAttr>()) {
      newBcastIndices.push_back(static_cast<int32_t>(index.getLimitedValue()));
    }

    assert(newBcastIndices.size() == 2 &&
           "[MOGG_OPS] If kernel has bcast indices it should only have two.");

    bcastIndices = newBcastIndices;
  }

  // Allocate buffers to hold all the operation's result tensors. At this stage
  // we gate the allocation only on the overall 'execute' block's chain. The
  // PeepholeTensorExtract pass will change this to use the chain which signals
  // when all of the operation's input buffers have been written to, thus
  // ensuring we allocate buffers just-in-time for their writers.
  auto errOr =
      helper.lowerOutputBuffer(originalName, loweringState.getExecuteChain(),
                               inputTensors, bcastIndices);
  if (errOr.isError())
    return failure();
  SmallVector<LoweredTensor> outputBufferAndSpec = errOr.takeValue();

  size_t numOutputBuffers = outputBufferAndSpec.size();
  SmallVector<Value> operands =
      helper.getOperands(inputTensors, outputBufferAndSpec);

  // Create a call to a custom invoke function we can put in the MEF.
  mlir::OperationState opState{op->getLoc(), "mop.kgen.execute_kernel"};

  // For compiled ops add the context as argument and all the metadata needed to
  // extract a compiled kernel from the context.
  operands.insert(operands.begin(), loweringState.getExecuteMgpContext());
  opState.addAttribute("kernelName", op.getKernelNameAttr());
  opState.addAttribute("jitCtxSlot",
                       loweringState.getExecuteOpBuilder().getUI32IntegerAttr(
                           loweringState.getJitCtxSlot()));

  // The operation's execution will be gated by a chain. As for output
  // allocation above, at this stage use the 'execute' block's chain.
  // The PeepholeTensorExtract will change this.
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

static ParseResult
parseKernel(OpAsmParser &p,
            SmallVectorImpl<OpAsmParser::UnresolvedOperand> &operands,
            SmallVectorImpl<Type> &operandTypes,
            SmallVectorImpl<Type> &resultTypes, Region &region) {
  SmallVector<OpAsmParser::Argument> args;
  llvm::SMLoc loc;
  KGEN::SignatureType sigType;
  FunctionType functionType;
  KGEN::ParamDeclArrayAttr inputParams, resultParams;

  llvm::SMLoc sigLoc;
  if (p.parseOperandList(operands, AsmParser::Delimiter::Paren) ||
      p.parseColon() || p.getCurrentLocation(&sigLoc) ||
      KGEN::parseFunctionSignature(p, args, inputParams, resultParams,
                                   functionType, sigType) ||
      p.getCurrentLocation(&loc) || p.parseRegion(region, args))
    return failure();
  if (!inputParams.empty() || !resultParams.empty())
    return p.emitError(sigLoc, "unexpected input or result parameters");

  for (const OpAsmParser::Argument &arg : args)
    operandTypes.push_back(arg.type);

  for (Type res : functionType.getResults())
    resultTypes.push_back(res);

  return success();
}

static void printKernel(OpAsmPrinter &p, mlir::Operation *kernel,
                        ValueRange operands, TypeRange operandTypes,
                        TypeRange resultTypes, Region &region) {
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
