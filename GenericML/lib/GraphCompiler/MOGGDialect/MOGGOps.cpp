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
#include "KGEN/KGENDialect/KGENUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/IR/FunctionImplementation.h"
#include "llvm/ADT/StringMap.h"

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
  ErrorOr<SmallVector<LoweredTensor>>
  lowerOutputBuffer(llvm::StringRef opName,
                    const SmallVector<LoweredTensor> &inputTensors);

  // Convert the tensors to a list of buffer/spec value pairs.
  SmallVector<Value> getOperands(SmallVector<LoweredTensor> &in,
                                 SmallVector<LoweredTensor> &outs);

  // Fix up the outputs
  void allocateOutputTensor(SmallVector<LoweredTensor> &outputTensors);

private:
  MGP::MOGraphLoweringState &loweringState;
  mlir::Operation *opBeingLowered;
  mlir::Location loc;

  ErrorOr<LoweredTensor>
  lowerDynamicOutput(MGP::CpuTensorSpecType outSpecType, llvm::StringRef opName,
                     const SmallVector<LoweredTensor> &inputTensors);
};

SmallVector<LoweredTensor> LowerHelper::lowerInputs() {
  SmallVector<LoweredTensor> inputs;
  // Allocate space for the buffers + specs per tensor.
  inputs.reserve(opBeingLowered->getNumOperands());

  for (Value operand : opBeingLowered->getOperands()) {
    // Extract the buffer and spec from the MGP tensor.
    mlir::Value mgpTensor = loweringState.lookupExecuteLoweredTensor(operand);

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
    SmallVector<LoweredTensor> &outputTensors) {
  size_t outputIndex = 0;

  // Allocate the output tensors this must be done after as the tensor is read
  // only.
  for (LoweredTensor &tensor : outputTensors) {
    // Get the type of the spec and infer the type of the tensor from that.
    auto ty = cast<MGP::CpuTensorSpecType>(tensor.spec.getType());
    auto tensorType = MGP::CpuTensorType::get(
        loweringState.getExecuteOpBuilder().getContext(), ty);

    // Create an immutable tensor
    auto tensorCreateOp =
        loweringState.getExecuteOpBuilder().create<MGP::CpuTensorCreateOp>(
            loc, tensorType, loweringState.getExecuteChain(), tensor.buffer,
            tensor.spec);

    // Update the MO tensor to now refer to this mgp tensor.
    loweringState.emplaceExecuteLoweredTensor(
        opBeingLowered->getResult(outputIndex), tensorCreateOp.getResult());
    outputIndex++;
  }
}

ErrorOr<LoweredTensor> LowerHelper::lowerDynamicOutput(
    MGP::CpuTensorSpecType outSpecType, llvm::StringRef opName,
    const SmallVector<LoweredTensor> &inputTensors) {
  // Dynamic Case
  // TODO: This is hacky, hacky, hacky, remove as soon as we can lower
  // parametric shapes to the runtime
  // TODO: If we decide we want this longer term (say, for MLAS kernels)
  // We should refactor it out as a Trait maybe? and make this a method
  // and data member of the op class.

  // (arg0Spec, arg1Spec, ..., argNSpec) -> outSpec
  static const llvm::StringMap<std::string> specNaryShapeFuncs{
      {"mop.kernels.abs", "mop.shape.unary"},
      {"mop.kernels.erf", "mop.shape.unary"},
      {"mop.kernels.exp", "mop.shape.unary"},
      {"mop.kernels.pow3", "mop.shape.unary"},
      {"mop.kernels.relu", "mop.shape.unary"},
      {"mop.kernels.rsqrt", "mop.shape.unary"},
      {"mop.kernels.select", "mop.shape.select"},
      {"mop.kernels.sigmoid", "mop.shape.unary"},
      {"mop.kernels.softmax", "mop.shape.unary"},
      {"mop.kernels.sqrt", "mop.shape.unary"},
      {"mop.kernels.tanh", "mop.shape.unary"},
      {"mop.kernels.add", "mop.shape.binary"},
      {"mop.kernels.div", "mop.shape.binary"},
      {"mop.kernels.max", "mop.shape.binary"},
      {"mop.kernels.min", "mop.shape.binary"},
      {"mop.kernels.mul", "mop.shape.binary"},
      {"mop.kernels.sub", "mop.shape.binary"},
      {"mop.kernels.matrix_band_part", "mop.shape.matrix_band_part"},
      {"mop.kernels.matmul", "mop.shape.matmul"},
      {"mop.kernels.batch_matmul", "mop.shape.batch_matmul"},
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

  auto getOutputSpec = [&](const std::string &opName,
                           const SmallVector<Value> &inputs) {
    mlir::OperationState opState{loc, opName};
    // inputs are input specs
    opState.addOperands(inputs);
    // The return is the new spec
    opState.addTypes(cast<mlir::Type>(outSpecType));

    // Add it to the MLIR.
    mlir::Operation *invokeOp =
        loweringState.getExecuteOpBuilder().create(opState);
    return cast<TypedValue<MGP::CpuTensorSpecType>>(invokeOp->getResult(0));
  };

  LoweredTensor output;

  // Hackery! Pure Hackery! I'd start setting up classes for this, but
  // again, I don't expect this to survive E
  if (specNaryShapeFuncs.count(opName)) {
    // Get just the specs for tensors.
    SmallVector<Value> inputSpecs;
    inputSpecs.reserve(inputTensors.size());
    for (const LoweredTensor &tensor : inputTensors)
      inputSpecs.push_back(tensor.spec);

    // Create a call to a custom invoke function we can put in the MEF.
    output.spec = getOutputSpec(specNaryShapeFuncs.lookup(opName), inputSpecs);
  } else if (specAndTensorBinaryShapeFuncs.count(opName)) {
    const auto arg0SpecVal = inputTensors[0].spec;
    const auto arg1 = opBeingLowered->getOperands()[1];
    const auto arg1Val = loweringState.lookupExecuteLoweredTensor(arg1);
    output.spec = getOutputSpec(specAndTensorBinaryShapeFuncs.lookup(opName),
                                {arg0SpecVal, arg1Val});

  } else if (opName == "mop.kernels.gather") {
    Value target_axis = loweringState.lookupExecuteLoweredTensor(
        opBeingLowered->getOperands()[2]);
    output.spec =
        getOutputSpec("mop.shape.gather", {inputTensors[0].spec,
                                           inputTensors[1].spec, target_axis});
  } else if (opName == "mop.kernels.conv") {
    Value stride = loweringState.lookupExecuteLoweredTensor(
        opBeingLowered->getOperands()[2]);
    Value dilation = loweringState.lookupExecuteLoweredTensor(
        opBeingLowered->getOperands()[3]);
    output.spec = getOutputSpec(
        "mop.shape.conv",
        {inputTensors[0].spec, inputTensors[1].spec, stride, dilation});
  } else if (opName == "mop.kernels.range") {
    Value start = loweringState.lookupExecuteLoweredTensor(
        opBeingLowered->getOperands()[0]);
    Value limit = loweringState.lookupExecuteLoweredTensor(
        opBeingLowered->getOperands()[1]);
    Value step = loweringState.lookupExecuteLoweredTensor(
        opBeingLowered->getOperands()[2]);
    output.spec = getOutputSpec("mop.shape.range", {start, limit, step});
  } else {
    opBeingLowered->emitError("op requires output to have static shape");
    return Error("op requires output to have static shape");
  }
  output.buffer = loweringState.getExecuteOpBuilder()
                      .create<MGP::CpuBufferAllocSpecOp>(
                          loc,
                          /*theContext=*/loweringState.getExecuteMgpContext(),
                          /*runtimeSpec=*/output.spec,
                          /*rawAlign=*/loweringState.getMinimumCpuAlignment())
                      .getResult();

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
LowerHelper::lowerOutputBuffer(llvm::StringRef opName,
                               const SmallVector<LoweredTensor> &inputTensors) {

  SmallVector<LoweredTensor> outputBufferAndSpec;

  // Allocate the output buffer for each result and add as argument.
  for (Type t : opBeingLowered->getResultTypes()) {
    MGP::CpuTensorSpecType outSpecType =
        MGP::createCpuTensorSpec(t.cast<MO::TensorType>());

    Optional<uint64_t> bufferSize = outSpecType.getSizeInBytes();

    LoweredTensor output;
    if (!bufferSize) {
      // We can't see the size so lower to a dynamic buffer.
      UNWRAP_ERROR(out, lowerDynamicOutput(outSpecType, opName, inputTensors));
      output = out;
    } else {
      // Static Case
      // Allocate the buffer for the output.
      output.buffer =
          loweringState.getExecuteOpBuilder()
              .create<MGP::CpuBufferAllocStaticOp>(
                  loc,
                  /*theContext=*/loweringState.getExecuteMgpContext(),
                  /*aSize=*/*bufferSize,
                  /*bRawAlign=*/loweringState.getMinimumCpuAlignment())
              .getResult();

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

LogicalResult ExtOp::lower(MGP::MOGraphLoweringState &loweringState) {
  ExtOp op = *this;
  LowerHelper helper{loweringState, op};

  // Operands to pass to the operation. Input tensors are expanded to buffers
  // and specs in IR those are passed in.
  auto inputTensors = helper.lowerInputs();

  // Ext ops carry the lowered operation name around with them,
  // others might do something else.
  StringRef opName = op.getOpName();

  // Lower the outputs to buffers which can then be passed back in as
  // parameters.
  auto errOr = helper.lowerOutputBuffer(opName, inputTensors);
  if (errOr.isError())
    return failure();
  SmallVector<LoweredTensor> outputBufferAndSpec = errOr.takeValue();

  SmallVector<Value> operands =
      helper.getOperands(inputTensors, outputBufferAndSpec);

  // Create a call to a custom invoke function we can put in the MEF.
  mlir::OperationState opState{op->getLoc(), op.getOpName()};

  // Add the execute chain as the first argument.
  operands.insert(operands.begin(),
                  loweringState.getExecuteInvokableInputChain());
  opState.addOperands(operands);

  // The return type is just the chain, other outputs are passed as argument by
  // their buffer pointers.
  opState.addTypes({loweringState.getExecuteInvokableInputChain().getType()});

  // Add it to the MLIR.
  mlir::Operation *invokeOp =
      loweringState.getExecuteOpBuilder().create(opState);
  loweringState.setExecuteChain(invokeOp->getResult(0));

  // We can't create the tensors until after the operation as they are
  // immutable.
  helper.allocateOutputTensor(outputBufferAndSpec);

  return success();
}

LogicalResult LoadJitKernelOp::lower(MGP::MOGraphLoweringState &loweringState) {
  MOGG::LoadJitKernelOp kernel = *this;

  SmallVector<Value> operands{
      loweringState.getInitChain(),
      loweringState.getInitMgpContext(),
  };

  // MEF can't currenly work with arrays of strings. To curcumvent this
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
  mlir::OperationState opState{kernel.getLoc(), "mop.load_jit_object"};
  opState.addOperands(operands);

  // This will load and store the kernel at the given index into memory. The
  // address will be tracked in the symbol table and it can then be accessed by
  // refering to the index in the table. IMPORTANT! Order in MEF is alphabetical
  // by attribute name.
  opState.addAttribute("jitCtxSlot",
                       loweringState.getInitOpBuilder().getUI32IntegerAttr(
                           loweringState.getJitCtxSlot()));
  opState.addAttribute("kernelName", newSymAttrs);
  opState.addAttribute("objectData", kernel.getValue());

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

  // Lower the outputs to buffers which can then be passed back in as
  // parameters.
  auto errOr = helper.lowerOutputBuffer(originalName, inputTensors);
  if (errOr.isError())
    return failure();
  SmallVector<LoweredTensor> outputBufferAndSpec = errOr.takeValue();

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

  // Add the execute chain as the first argument.
  operands.insert(operands.begin(),
                  loweringState.getExecuteInvokableInputChain());
  opState.addOperands(operands);

  // The return type is just the chain, other outputs are passed as argument by
  // their buffer pointers.
  opState.addTypes({loweringState.getExecuteInvokableInputChain().getType()});

  // Add it to the MLIR.
  mlir::Operation *invokeOp =
      loweringState.getExecuteOpBuilder().create(opState);
  loweringState.setExecuteChain(invokeOp->getResult(0));

  // We can't create the tensors until after the operation as they are
  // immutable.
  helper.allocateOutputTensor(outputBufferAndSpec);

  return success();
}

static ParseResult parseLambda(OpAsmParser &p, StringAttr &name, TypeAttr &sig,
                               Region &region) {
  SmallVector<OpAsmParser::Argument> args;
  llvm::SMLoc loc;
  KGEN::SignatureType sigType;

  if (p.parseSymbolName(name) || p.parseEqual() ||
      KGEN::parseFunctionSignature(p, args, sigType) ||
      p.getCurrentLocation(&loc) || p.parseRegion(region, args))
    return failure();

  SmallVector<Type> argTypes;
  for (const OpAsmParser::Argument &arg : args)
    argTypes.push_back(arg.type);

  sig = TypeAttr::get(
      p.getBuilder().getFunctionType(argTypes, sigType.getValueResults()));
  return success();
}

static void printLambda(OpAsmPrinter &p, mlir::Operation *lambda,
                        StringAttr name, TypeAttr sig, Region &region) {
  p.printSymbolName(name.strref());
  p << " = ";

  auto func = cast<FunctionType>(sig.getValue());

  auto sigTy = KGEN::SignatureType::get(lambda->getContext(), func.getInputs(),
                                        func.getResults());
  KGEN::printFunctionSignature(p, region, sigTy);
  p.printRegion(region, /*printEntryBlockArgs=*/false);
}

static ParseResult
parseKernel(OpAsmParser &p,
            SmallVectorImpl<OpAsmParser::UnresolvedOperand> &operands,
            SmallVectorImpl<Type> &operandTypes,
            SmallVectorImpl<Type> &resultTypes, Region &region) {
  SmallVector<OpAsmParser::Argument> args;
  llvm::SMLoc loc;
  KGEN::SignatureType sig;

  if (p.parseOperandList(operands, AsmParser::Delimiter::Paren) ||
      p.parseColon() || KGEN::parseFunctionSignature(p, args, sig) ||
      p.getCurrentLocation(&loc) || p.parseRegion(region, args))
    return failure();

  for (const OpAsmParser::Argument &arg : args)
    operandTypes.push_back(arg.type);

  for (Type res : sig.getValueResults())
    resultTypes.push_back(res);

  return success();
}

static void printKernel(OpAsmPrinter &p, mlir::Operation *kernel,
                        ValueRange operands, TypeRange operandTypes,
                        TypeRange resultTypes, Region &region) {
  p << '(';
  p.printOperands(operands);
  p << ") : ";

  auto sigTy =
      KGEN::SignatureType::get(kernel->getContext(), operandTypes, resultTypes);

  KGEN::printFunctionSignature(p, region, sigTy);
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
