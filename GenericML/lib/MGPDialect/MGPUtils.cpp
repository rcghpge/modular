//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/MGPDialect/MGPUtils.h"

#include "GenericML/MGPDialect/MGPOps.h"
#include "GenericML/MGPDialect/MGPTypes.h"
#include "mlir/IR/Builders.h"
#include "llvm/Support/ErrorHandling.h"

using namespace M;
using namespace MGP;

namespace {
/// Name of attribute describing number of CpuBufferType operands which
/// will be written to by the operation, ie used as outputs. All
/// other CpuBufferType operands are consider to be read-only, ie used as
/// inputs. Input buffers must come before output buffers.
constexpr const char *kNumOutputBuffers = "numOutputBuffers";
} // namespace

void M::MGP::setNumOutputBuffers(Operation *op, size_t numOutputs) {
  OpBuilder builder(op->getContext());
  op->setAttr(kNumOutputBuffers, builder.getUI32IntegerAttr(numOutputs));
}

size_t M::MGP::getNumOutputBuffers(Operation *op) {
  auto numOutputsAttr = op->getAttrOfType<IntegerAttr>(kNumOutputBuffers);
  if (!numOutputsAttr)
    return 0;
  return static_cast<size_t>(numOutputsAttr.getUInt());
}

void M::MGP::removeNumOutputBuffers(Operation *op) {
  op->removeAttr(kNumOutputBuffers);
}

/// Returns the operand index for the first output buffer argument to op.
/// Return the op's arity if it has no output buffer arguments.
static size_t getFirstOutputBufferOperandIndex(Operation *op) {
  size_t numOutputBuffers = getNumOutputBuffers(op);
  if (numOutputBuffers == 0)
    return op->getNumOperands();
  size_t numBuffers = llvm::count_if(op->getOperands(), [](Value v) {
    return isa<CpuBufferType>(v.getType());
  });
  assert(numOutputBuffers <= numBuffers &&
         "operation has fewer buffer operands than its numOutputBuffers "
         "attribute");
  size_t numInputBuffers = numBuffers - numOutputBuffers;
  size_t bufferIndex = 0; // index for next buffer operand
  for (size_t i = 0, n = op->getNumOperands(); i < n; ++i) {
    Value operand = op->getOperand(i);
    if (isa<CpuBufferType>(operand.getType())) {
      if (bufferIndex == numInputBuffers)
        return i;
      ++bufferIndex;
    }
  }
  llvm_unreachable("didn't find first output buffer");
}

bool M::MGP::isBufferReadingOpOperand(OpOperand &opOperand) {
  return isa<CpuBufferType>(opOperand.get().getType()) &&
         opOperand.getOperandNumber() <
             getFirstOutputBufferOperandIndex(opOperand.getOwner());
}

bool M::MGP::isBufferWritingOpOperand(OpOperand &opOperand) {
  return isa<CpuBufferType>(opOperand.get().getType()) &&
         opOperand.getOperandNumber() >=
             getFirstOutputBufferOperandIndex(opOperand.getOwner());
}
