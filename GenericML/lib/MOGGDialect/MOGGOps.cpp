//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements the MO dialect operations.
//
//===----------------------------------------------------------------------===//

#include "GenericML/MOGGDialect/MOGGOps.h"
#include "GenericML/Conversion/MOToPrimitives.h"
#include "GenericML/MGPDialect/MGPOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/FunctionImplementation.h"

using namespace M;
using namespace MOGG;

LogicalResult ExtOp::lower(MGP::MOGraphLoweringState &loweringState) {
  auto op = static_cast<Op *>(this);
  auto loc = op->getLoc();

  // Operands to pass to the operation. Input tensors are expanded to buffers
  // and specs in IR those are passed in.
  SmallVector<Value> operands;
  SmallVector<Value> input_specs;
  for (Value operand : op->getOperands()) {
    // Extract the buffer and spec from the MGP tensor.
    mlir::Value mgpTensor = loweringState.lookupExecuteLoweredTensor(operand);

    auto extract =
        loweringState.getExecuteOpBuilder().create<MGP::CpuTensorExtractOp>(
            loc, mgpTensor);

    // Add the expanded buffer and spec as arguments to the operation.
    operands.push_back(extract.getBufferResult());
    operands.push_back(extract.getTensorSpecResult());
    input_specs.push_back(extract.getTensorSpecResult());
  }

  // We can't create the tensors until after the operation as they are
  // immutable.
  SmallVector<std::pair<TypedValue<MGP::CpuBufferType>,
                        TypedValue<MGP::CpuTensorSpecType>>>
      outputBufferAndSpec;

  // Allocate the output buffer for each result and add as argument.
  for (Type t : op->getResultTypes()) {
    MGP::CpuTensorSpecType outSpecType =
        MGP::createCpuTensorSpec(t.cast<MO::TensorType>());
    Optional<uint64_t> bufferSize = outSpecType.getSizeInBytes();
    TypedValue<MGP::CpuBufferType> buf = Value();
    TypedValue<MGP::CpuTensorSpecType> spec = Value();
    if (!bufferSize) {
      // Dynamic Case
      // TODO: This is hacky, hacky, hacky, remove as soon as we can lower
      // parametric shapes to the runtime
      // TODO: If we decide we want this longer term (say, for MLAS kernels)
      // We should refactor it out as a Trait maybe? and make this a method
      // and data member of the op class.
      static const std::map<llvm::StringRef, std::string> shapeFuncs{
          {"mop.kernels.abs", "mop.shape.unary"},
          {"mop.kernels.erf", "mop.shape.unary"},
          {"mop.kernels.exp", "mop.shape.unary"},
          {"mop.kernels.gelu", "mop.shape.unary"},
          {"mop.kernels.relu", "mop.shape.unary"},
          {"mop.kernels.softmax", "mop.shape.unary"},
          {"mop.kernels.tanh", "mop.shape.unary"},
          {"mop.kernels.add", "mop.shape.binary"},
          {"mop.kernels.div", "mop.shape.binary"},
          {"mop.kernels.mul", "mop.shape.binary"},
          {"mop.kernels.sub", "mop.shape.binary"},
          {"mop.kernels.matmul", "mop.shape.matmul"}};

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
        return invokeOp->getResult(0);
      };

      StringRef opName = this->getOpName();
      if (shapeFuncs.count(opName)) {
        // Create a call to a custom invoke function we can put in the BEF.
        spec = getOutputSpec(shapeFuncs.at(opName), input_specs);
      } else if (opName == "mop.kernels.broadcast_to") {
        // Hackery! Pure Hackery! I'd start setting up classes for this, but
        // again, I don't expect this to survive E
        Value target_shape =
            loweringState.lookupExecuteLoweredTensor(op->getOperands()[1]);
        spec = getOutputSpec("mop.shape.broadcast_to",
                             {input_specs[0], target_shape});
      } else {
        return op->emitError("op requires output to have static shape");
      }
      buf = loweringState.getExecuteOpBuilder()
                .create<MGP::CpuBufferAllocSpecOp>(
                    loc, loweringState.getExecuteMgpContext(), spec)
                .getResult();
    } else {
      // Static Case
      // Allocate the buffer for the output.
      buf = loweringState.getExecuteOpBuilder()
                .create<MGP::CpuBufferAllocStaticOp>(
                    loc, loweringState.getExecuteMgpContext(), *bufferSize)
                .getResult();

      // Create an MGP tensor to match the MO output tensor.
      spec = loweringState.getExecuteOpBuilder()
                 .create<MGP::CpuTensorSpecCreateOp>(loc, outSpecType)
                 .getResult();
    }
    // Add the output buffer and spec as inputs to the operation.
    operands.push_back(buf);
    operands.push_back(spec);

    // Track the buffer and spec so we can create the immutable tensor after the
    // operation.
    outputBufferAndSpec.push_back({buf, spec});
  }

  // Create a call to a custom invoke function we can put in the BEF.
  mlir::OperationState opState{loc, this->getOpName()};

  // Add the execute chain as the first argument.
  operands.insert(operands.begin(), loweringState.getExecuteChain());
  opState.addOperands(operands);

  // The return type is just the chain, other outputs are passed as argument by
  // their buffer pointers.
  opState.addTypes({loweringState.getExecuteChain().getType()});

  // Add it to the MLIR.
  mlir::Operation *invokeOp =
      loweringState.getExecuteOpBuilder().create(opState);
  loweringState.setExecuteChain(invokeOp->getResult(0));

  size_t outputIndex = 0;

  // Allocate the output tensors this must be done after as the tensor is read
  // only.
  for (auto &[buffer, spec] : outputBufferAndSpec) {
    // Get the type of the spec and infer the type of the tensor from that.
    auto ty = cast<MGP::CpuTensorSpecType>(spec.getType());
    auto tensorType = MGP::CpuTensorType::get(
        loweringState.getExecuteOpBuilder().getContext(), ty);

    // Create an immutable tensor
    auto tensorCreateOp =
        loweringState.getExecuteOpBuilder().create<MGP::CpuTensorCreateOp>(
            loc, tensorType, loweringState.getExecuteChain(), buffer, spec);

    // Update the MO tensor to now refer to this mgp tensor.
    loweringState.emplaceExecuteLoweredTensor(op->getResult(outputIndex),
                                              tensorCreateOp.getResult());
    outputIndex++;
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

void MOGGDialect::registerOperations() {
  addOperations<
#define GET_OP_LIST
#include "GenericML/MOGGDialect/MOGG.cpp.inc"
      >();
}

#define GET_OP_CLASSES
#include "GenericML/MOGGDialect/MOGG.cpp.inc"
