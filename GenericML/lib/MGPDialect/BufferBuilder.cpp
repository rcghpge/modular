//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/MGPDialect/BufferBuilder.h"
#include "GenericML/MGPDialect/MGPOps.h"
#include "GenericML/MGPDialect/MGPUtils.h"
#include "GraphRT/GRTDialect/GRTOps.h"
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

TypedValue<MGP::ContextType> BufferBuilder::getContextForOp(Operation *op) {
  return getContextForBlock(op->getBlock());
}

void BufferBuilder::vtableAnchor() {}

//===----------------------------------------------------------------------===//
// CpuBufferBuilder
//===----------------------------------------------------------------------===//

Value CpuBufferBuilder::bufferConstant(OpBuilder &builder, Location loc,
                                       TypedValue<MGP::ContextType> context,
                                       ElementsAttr value) {
  return builder.create<CpuBufferConstantOp>(loc, /*value=*/value);
}

std::pair<Value, TypedValue<CpuTensorSpecType>>
CpuBufferBuilder::tensorExtract(OpBuilder &builder, Location loc,
                                TypedValue<MGP::ContextType> context,
                                Value tensor) {
  auto extractOp = builder.create<CpuTensorExtractOp>(loc, /*tensor=*/tensor);
  return {extractOp.getBufferResult(), extractOp.getTensorSpecResult()};
}

TypedValue<CpuTensorSpecType>
CpuBufferBuilder::tensorExtractShape(OpBuilder &builder, Location loc,
                                     TypedValue<MGP::ContextType> context,
                                     Value tensor) {
  return builder.create<CpuTensorExtractShapeOp>(loc, /*tensor=*/tensor);
}

Value CpuBufferBuilder::tensorCreate(OpBuilder &builder, Location loc,
                                     TypedValue<MGP::ContextType> context,
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
                                          TypedValue<MGP::ContextType> context,
                                          Value inChain, uint64_t size,
                                          uint64_t alignment) {
  return builder.create<CpuBufferAllocStaticOp>(loc,
                                                /*inChain=*/inChain,
                                                /*aSize=*/size,
                                                /*bRawAlign=*/alignment);
}

Value CpuBufferBuilder::bufferAllocSpec(
    OpBuilder &builder, Location loc, TypedValue<MGP::ContextType> context,
    Value inChain, TypedValue<MGP::CpuTensorSpecType> tensorSpec,
    uint64_t alignment) {
  return builder.create<CpuBufferAllocSpecOp>(loc,
                                              /*inChain=*/inChain,
                                              /*runtimeSpec=*/tensorSpec,
                                              /*rawAlign=*/alignment);
}

Value CpuBufferBuilder::bufferAllocDynamic(OpBuilder &builder, Location loc,
                                           Type bufType,
                                           TypedValue<MGP::ContextType> context,
                                           Value inChain, Value size,
                                           uint64_t alignment) {
  return builder.create<CpuBufferAllocDynamicOp>(loc,
                                                 /*outBuffer=*/bufType,
                                                 /*inChain=*/inChain,
                                                 /*aSize=*/size,
                                                 /*bRawAlign=*/alignment);
}

Value CpuBufferBuilder::bufferTransferFromHost(
    OpBuilder &builder, Location loc, TypedValue<MGP::ContextType> context,
    Value inChain, TypedValue<M::MGP::CpuBufferType> srcBuffer,
    Value dstBuffer) {
  llvm::llvm_unreachable_internal("not implemented");
}

Value CpuBufferBuilder::bufferTransferToHost(
    OpBuilder &builder, Location loc, TypedValue<MGP::ContextType> context,
    Value inChain, Value srcBuffer,
    TypedValue<M::MGP::CpuBufferType> dstBuffer) {
  llvm::llvm_unreachable_internal("not implemented");
}

Value CpuBufferBuilder::debugTensorPrint(
    OpBuilder &builder, Location loc, TypedValue<MGP::ContextType> context,
    Value inChain, Value buffer, TypedValue<MGP::CpuTensorSpecType> spec,
    StringAttr label) {
  return builder.create<DebugCpuTensorPrintOp>(loc, /*inChain=*/inChain,
                                               /*buffer=*/buffer,
                                               /*spec=*/spec,
                                               /*label=*/label);
}

Value CpuBufferBuilder::kernelLoadArchive(OpBuilder &builder, Location loc,
                                          TypedValue<MGP::ContextType> context,
                                          Value inChain, ElementsAttr archive,
                                          ArrayAttr kernelNames,
                                          ArrayAttr constantNames) {
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
      /*theContext=*/context,
      /*aSlot=*/slot,
      /*bArchive=*/archive,
      /*cKernelNames=*/kernelNamesAttr);
}

Value CpuBufferBuilder::kernelExecute(
    OpBuilder &builder, Location loc, TypedValue<MGP::ContextType> context,
    StringRef kernelName, Value inChain, ArrayRef<Value> args,
    ArrayRef<InOutSignatureAttr::InOutSemantics> inOutSemantics,
    ArrayRef<Value> inDimensions, uint64_t numOutDimensions,
    SmallVectorImpl<Value> &outDimensions, StringRef summary) {
  // Use the kernel name to determine the index.
  uint64_t index = getCompiledKernelIndex(kernelName);

  auto executeOp = builder.create<MGP::CpuKernelExecuteOp>(
      loc,
      /*inChain=*/inChain,
      /*theContext=*/context,
      /*buffers=*/args,
      /*inDimensions=*/inDimensions,
      /*aSlot=*/slot,
      /*bKernelIndex=*/index,
      /*cInOutSignature=*/
      InOutSignatureAttr::get(builder.getContext(), inOutSemantics),
      // Use the summary for debugging and profiling.
      /*dSummary=*/summary,
      /*eNumOutDimensions=*/numOutDimensions);

  for (Value outDimension : executeOp.getOutDimensions())
    outDimensions.emplace_back(outDimension);

  return executeOp.getOutChain();
}

Value CpuBufferBuilder::chainCreate(OpBuilder &builder, Location loc,
                                    TypedValue<MGP::ContextType> context,
                                    ArrayRef<Value> inChains) {
  return builder.create<GRT::ChainCreateOp>(
      loc,
      /*result=*/MEF::ChainType::get(builder.getContext()),
      /*operands=*/inChains);
}

std::pair<TypedValue<MEF::ChainType>, size_t>
CpuBufferBuilder::cache(OpBuilder &initBuilder, Location loc,
                        TypedValue<MGP::ContextType> context,
                        TypedValue<MEF::ChainType> inChain,
                        ContextCreateOp contextCreateOp, Value buffer) {
  size_t bufferSlot = contextCreateOp.getNumSlots();
  contextCreateOp.setNumSlots(bufferSlot + 1);
  auto cacheOp = initBuilder.create<CpuBufferCacheOp>(
      loc, /*inChain=*/inChain, /*theContext=*/contextCreateOp,
      /*buffer=*/buffer, /*slot=*/bufferSlot);
  return {cacheOp.getOutChain(), bufferSlot};
}

Value CpuBufferBuilder::getCached(OpBuilder &builder, Location loc,
                                  TypedValue<MGP::ContextType> context,
                                  TypedValue<MEF::ChainType> inChain,
                                  size_t bufferSlot, Type expectedBufferType) {
  auto bufferType = dyn_cast<CpuBufferType>(expectedBufferType);
  assert(bufferType &&
         "expected buffer type should be MGP::CpuBufferType for this builder");
  return builder.create<CpuBufferGetCachedOp>(
      loc, /*inChain=*/inChain, /*theContext=*/context,
      /*aSlot=*/bufferSlot, /*bRawSize=*/bufferType.getRawSize().getUInt());
}

TypedValue<MEF::ChainType>
CpuBufferBuilder::toMEFChain(OpBuilder &builder, Location loc,
                             TypedValue<MGP::ContextType> context,
                             Value inChain) {
  assert(isa<MEF::ChainType>(inChain.getType()) &&
         "chain should be of type MEF::ChainType for this builder");
  return cast<TypedValue<MEF::ChainType>>(inChain);
}

Value CpuBufferBuilder::fromMEFChain(OpBuilder &builder, Location loc,
                                     TypedValue<MGP::ContextType> context,
                                     TypedValue<MEF::ChainType> inChain) {
  return inChain;
}

Value CpuBufferBuilder::borrowSectionStatic(OpBuilder &builder, Location loc,
                                            Value inChain, Value inBuffer,
                                            uint64_t offset, uint64_t size) {
  return builder.create<CpuBufferSectionBorrowStaticOp>(loc, inChain, inBuffer,
                                                        offset, size);
}

Value CpuBufferBuilder::borrowSectionDynamic(OpBuilder &builder, Location loc,
                                             Value inChain, Value inBuffer,
                                             Value scale,
                                             uint64_t unscaledOffset,
                                             uint64_t unscaledSize) {
  return builder.create<CpuBufferSectionBorrowDynamicOp>(
      loc, inChain, inBuffer, scale, unscaledOffset, unscaledSize);
}
