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
/// inputs. We must support two conventions depending on relative order
/// of output buffers w.r.t. input buffers.
constexpr const char *kNumFrontOutputBuffers = "numFrontOutputBuffers";
constexpr const char *kNumBackOutputBuffers = "numBackOutputBuffers";
} // namespace

void M::MGP::setNumFrontOutputBuffers(Operation *op, size_t numFrontOutputs) {
  OpBuilder builder(op->getContext());
  op->setAttr(kNumFrontOutputBuffers,
              builder.getUI32IntegerAttr(numFrontOutputs));
}

void M::MGP::setNumBackOutputBuffers(Operation *op, size_t numBackOutputs) {
  OpBuilder builder(op->getContext());
  op->setAttr(kNumBackOutputBuffers,
              builder.getUI32IntegerAttr(numBackOutputs));
}

void M::MGP::removeNumOutputBuffers(Operation *op) {
  op->removeAttr(kNumFrontOutputBuffers);
  op->removeAttr(kNumBackOutputBuffers);
}

static size_t getNumFrontOutputBuffers(Operation *op) {
  auto numFrontOutputsAttr =
      op->getAttrOfType<IntegerAttr>(kNumFrontOutputBuffers);
  if (!numFrontOutputsAttr)
    return 0;
  return static_cast<size_t>(numFrontOutputsAttr.getUInt());
}

static size_t getNumBackOutputBuffers(Operation *op) {
  auto numBackOutputsAttr =
      op->getAttrOfType<IntegerAttr>(kNumBackOutputBuffers);
  if (!numBackOutputsAttr)
    return 0;
  return static_cast<size_t>(numBackOutputsAttr.getUInt());
}

/// Returns true if n'th operand to op is written to.
static bool isWrittenTo(Operation *op, size_t n) {
  // Is opOperand in the first numFrontOutputBuffer's buffer operands?
  size_t numFrontOutputBuffers = getNumFrontOutputBuffers(op);
  for (size_t i = 0; i <= n && numFrontOutputBuffers > 0; ++i) {
    if (isa<CpuBufferType>(op->getOperand(i).getType())) {
      if (i == n)
        return true;
      --numFrontOutputBuffers;
    }
  }

  // Is opOperand in the last numBackOutputBuffer's buffer operands?
  size_t numBackOutputBuffers = getNumBackOutputBuffers(op);
  for (size_t ip1 = op->getNumOperands();
       ip1 >= n + 1 && numBackOutputBuffers > 0; --ip1) {
    if (isa<CpuBufferType>(op->getOperand(ip1 - 1).getType())) {
      if (ip1 == n + 1)
        return true;
      --numBackOutputBuffers;
    }
  }

  return false;
}

bool M::MGP::isBufferReadingOpOperand(OpOperand &opOperand) {
  return isa<CpuBufferType>(opOperand.get().getType()) &&
         !isWrittenTo(opOperand.getOwner(), opOperand.getOperandNumber());
}

bool M::MGP::isBufferWritingOpOperand(OpOperand &opOperand) {
  return isa<CpuBufferType>(opOperand.get().getType()) &&
         isWrittenTo(opOperand.getOwner(), opOperand.getOperandNumber());
}
