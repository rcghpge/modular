//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/MGPDialect/BufferBuilder.h"
#include "GenericML/MGPDialect/MGPOps.h"
#include "mlir/IR/Builders.h"

using namespace M;
using namespace M::MGP;

//===----------------------------------------------------------------------===//
// CpuBufferBuilder
//===----------------------------------------------------------------------===//

Value CpuBufferBuilder::bufferConstant(OpBuilder &builder, Location loc,
                                       ElementsAttr value) {
  return builder.create<CpuBufferConstantOp>(loc, /*value=*/value);
}

std::pair<Value, TypedValue<CpuTensorSpecType>>
CpuBufferBuilder::tensorExtract(OpBuilder &builder, Location loc,
                                Value tensor) {
  auto extractOp = builder.create<CpuTensorExtractOp>(loc, /*tensor=*/tensor);
  return {extractOp.getBufferResult(), extractOp.getTensorSpecResult()};
}

TypedValue<CpuTensorSpecType>
CpuBufferBuilder::tensorExtractShape(OpBuilder &builder, Location loc,
                                     Value tensor) {
  return builder.create<CpuTensorExtractShapeOp>(loc, /*tensor=*/tensor);
}

Value CpuBufferBuilder::tensorCreate(OpBuilder &builder, Location loc,
                                     Value inChain, Value buffer,
                                     TypedValue<CpuTensorSpecType> tensorSpec) {
  return builder.create<CpuTensorCreateOp>(
      loc,
      /*result=*/CpuTensorType::get(builder.getContext(), tensorSpec.getType()),
      /*inChain=*/inChain,
      /*buffer=*/buffer,
      /*tensorSpec=*/tensorSpec);
}

Value CpuBufferBuilder::bufferAllocStatic(OpBuilder &builder, Location loc,
                                          Value inChain, uint64_t size,
                                          uint64_t alignment) {
  return builder.create<CpuBufferAllocStaticOp>(loc,
                                                /*inChain=*/inChain,
                                                /*aSize=*/size,
                                                /*bRawAlign=*/alignment);
}

Value CpuBufferBuilder::bufferAllocSpec(
    OpBuilder &builder, Location loc, Value inChain,
    TypedValue<MGP::CpuTensorSpecType> tensorSpec, uint64_t alignment) {
  return builder.create<CpuBufferAllocSpecOp>(loc,
                                              /*inChain=*/inChain,
                                              /*runtimeSpec=*/tensorSpec,
                                              /*rawAlign=*/alignment);
}

Value CpuBufferBuilder::bufferTransferFromHost(
    OpBuilder &builder, Location loc, Value inChain,
    TypedValue<M::MGP::CpuBufferType> srcBuffer, Value dstBuffer) {
  llvm::llvm_unreachable_internal("not implemented");
}

Value CpuBufferBuilder::bufferTransferToHost(
    OpBuilder &builder, Location loc, Value inChain, Value srcBuffer,
    TypedValue<M::MGP::CpuBufferType> dstBuffer) {
  llvm::llvm_unreachable_internal("not implemented");
}

Value CpuBufferBuilder::debugTensorPrint(OpBuilder &builder, Location loc,
                                         Value inChain, Value tensor,
                                         StringAttr label) {
  return builder.create<DebugCpuTensorPrintOp>(loc, /*inChain=*/inChain,
                                               /*tensor=*/tensor,
                                               /*label=*/label);
}
