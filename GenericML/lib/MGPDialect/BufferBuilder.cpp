//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/MGPDialect/BufferBuilder.h"
#include "GenericML/MGPDialect/MGPOps.h"
#include "GenericML/MGPDialect/MGPUtils.h"
#include "mlir/IR/Builders.h"

using namespace M;
using namespace M::MGP;

//===----------------------------------------------------------------------===//
// TypeBuilder
//===----------------------------------------------------------------------===//

void TypeBuilder::vtableAnchor() {}

//===----------------------------------------------------------------------===//
// BufferBuilder
//===----------------------------------------------------------------------===//

void BufferBuilder::vtableAnchor() {}

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

Value CpuBufferBuilder::kernelLoadArchive(
    OpBuilder &builder, Location loc, Value inChain,
    TypedValue<MGP::ContextType> initContext, ElementsAttr archive,
    ArrayAttr kernelNames, ArrayAttr constantNames) {
  assert((constantNames == nullptr || constantNames.empty()) &&
         "CPU archives do not currently support constants");

  // Collect the kernel symbols into an #M.multiline attribute, and keep track
  // of their symbol-to-index mappings.
  SmallVector<StringAttr> kernelNameAttrs;
  size_t index = 0;
  for (StringRef symbol : kernelNames.getAsValueRange<mlir::StringAttr>()) {
    kernelNameAttrs.emplace_back(builder.getStringAttr(symbol));
    trackCompiledKernel(symbol, index);
    index++;
  }
  auto kernelNamesAttr =
      MultiLineStringAttr::get(builder.getContext(), kernelNameAttrs);

  return builder.create<CpuKernelLoadArchiveOp>(
      loc,
      /*inChain=*/inChain,
      /*theContext=*/initContext,
      /*aSlot=*/slot,
      /*bArchive=*/archive,
      /*cKernelNames=*/kernelNamesAttr);
}

Value CpuBufferBuilder::kernelExecute(
    OpBuilder &builder, Location loc, StringRef kernelName, bool isViaHost,
    Value inChain, TypedValue<MGP::ContextType> execContext,
    SmallVector<Value> buffers,
    ArrayRef<InOutSignatureAttr::InOutSemantics> bufferSemantics,
    SmallVector<Value> inDimensions, uint64_t numOutDimensions,
    SmallVectorImpl<Value> &outDimensions) {
  uint64_t index = getCompiledKernelIndex(kernelName);

  auto executeOp = builder.create<MGP::CpuKernelExecuteOp>(
      loc,
      /*inChain=*/inChain,
      /*theContext=*/execContext,
      /*buffers=*/buffers,
      /*inDimensions=*/inDimensions,
      /*aSlot=*/slot,
      /*bKernelIndex=*/index,
      /*cInOutSignature=*/
      InOutSignatureAttr::get(builder.getContext(), bufferSemantics),
      /*dKernelName=*/kernelName,
      /*eNumOutDimensions=*/numOutDimensions);

  // Downstream passes may need to distinguish input from output buffers.
  // TODO: Redundant with in/out signature, will be removed
  // once transitioned away from num{Front,Back}OutputBuffers
  // convention.
  MGP::setNumBackOutputBuffers(
      executeOp, llvm::count(bufferSemantics, InOutSignatureAttr::kOut));

  for (Value outDimension : executeOp.getOutDimensions())
    outDimensions.emplace_back(outDimension);

  return executeOp.getOutChain();
}
