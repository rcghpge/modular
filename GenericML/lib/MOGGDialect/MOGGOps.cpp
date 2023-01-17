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

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

LogicalResult ExtOp::lower(MGP::MOGraphLoweringState &loweringState) {
  auto op = static_cast<Op *>(this);
  auto loc = op->getLoc();

  // Operands to pass to the operation. Input tensors are expanded to buffers
  // and specs in IR those are passed in.
  SmallVector<Value> operands;
  for (Value operand : op->getOperands()) {
    // Extract the buffer and spec from the MGP tensor.
    mlir::Value mgpTensor = loweringState.lookupExecuteLoweredTensor(operand);

    auto extract =
        loweringState.getExecuteOpBuilder().create<MGP::CpuTensorExtractOp>(
            loc, mgpTensor);

    // Add the expanded buffer and spec as arguments to the operation.
    operands.push_back(extract.getBufferResult());
    operands.push_back(extract.getTensorSpecResult());
  }

  // We can't create the tensors until after the operation as they are
  // immutable.
  SmallVector<
      std::pair<MGP::CpuBufferAllocStaticOp, MGP::CpuTensorSpecCreateOp>>
      outputBufferAndSpec;

  // Allocate the output buffer for each result and add as argument.
  for (Type t : op->getResultTypes()) {
    auto outSpecType = MGP::createCpuTensorSpec(t.cast<MO::TensorType>());
    Optional<uint64_t> bufferSize = outSpecType.getSizeInBytes();
    // Fail for dynamic shapes for now.
    if (!bufferSize)
      return failure();

    // Allocate the buffer for the output.
    auto bufferOp =
        loweringState.getExecuteOpBuilder().create<MGP::CpuBufferAllocStaticOp>(
            loc, loweringState.getExecuteMgpContext(), *bufferSize);

    // Create an MGP tensor to match the MO output tensor.
    auto specOp =
        loweringState.getExecuteOpBuilder().create<MGP::CpuTensorSpecCreateOp>(
            loc, outSpecType);

    // Add the output buffer and spec as inputs to the operation.
    operands.push_back(bufferOp.getResult());
    operands.push_back(specOp.getResult());

    // Track the buffer and spec so we can create the immutable tensor after the
    // operation.
    outputBufferAndSpec.push_back({bufferOp, specOp});
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
    auto ty = cast<MGP::CpuTensorSpecType>(spec->getResultTypes()[0]);
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

void MOGGDialect::registerOperations() {
  addOperations<
#define GET_OP_LIST
#include "GenericML/MOGGDialect/MOGG.cpp.inc"
      >();
}

#define GET_OP_CLASSES
#include "GenericML/MOGGDialect/MOGG.cpp.inc"
