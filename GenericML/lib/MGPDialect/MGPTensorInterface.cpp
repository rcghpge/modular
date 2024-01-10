//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/MGPDialect/MGPTensorInterface.h"

#include "GenericML/MGPDialect/MGPOps.h"
#include "GenericML/MGPDialect/MGPTypes.h"
#include "GraphRT/GRTDialect/GRTOps.h"
#include "mlir/IR/Builders.h"

namespace M::MGP {

//===----------------------------------------------------------------------===//
// TensorBufferDialectInterface
//===----------------------------------------------------------------------===//

DenseSet<TensorBufferDialectInterface *>
TensorBufferDialectInterface::getAllInterfaces(ModelOp modelOp) {
  DenseSet<TensorBufferDialectInterface *> interfaces;
  modelOp.walk([&interfaces](Operation *op) {
    if (auto interface =
            dyn_cast_if_present<TensorBufferDialectInterface>(op->getDialect()))
      interfaces.insert(interface);
  });
  return interfaces;
}

BufferBuilderContext
TensorBufferDialectInterface::getAllBufferBuilders(ModelOp modelOp) {
  DenseSet<TensorBufferDialectInterface *> interfaces =
      getAllInterfaces(modelOp);
  BufferBuilderContext builderContext(
      DeviceRefAttr::get(modelOp.getContext(), kCPULabel, 0));

  // First create the host device's builder, since other builder may need
  // to refer to it.
  for (TensorBufferDialectInterface *interface : interfaces)
    interface->addHostBuilder(modelOp, builderContext);

  // Then create the rest of the builders.
  for (TensorBufferDialectInterface *interface : interfaces)
    interface->addNonHostBuilders(modelOp, builderContext);

  return builderContext;
}

TypeBuilder *TensorBufferDialectInterface::getTypeBuilder(
    const BufferBuilderContext &builderContext, Type type) {
  if (auto interface =
          dyn_cast<TensorBufferDialectInterface>(&type.getDialect())) {
    DeviceRefAttr device = interface->getStorageDevice(builderContext, type);
    return builderContext.getTypeBuilder(device);
  }

  return builderContext.getHostTypeBuilder();
}

BufferBuilder *TensorBufferDialectInterface::getBuilderForOpExecution(
    const BufferBuilderContext &builderContext, Operation *op) {
  if (auto interface =
          dyn_cast_if_present<TensorBufferDialectInterface>(op->getDialect())) {
    DeviceRefAttr device = interface->getExecutionDevice(builderContext, op);
    return builderContext.getBufferBuilder(device);
  }

  return builderContext.getHostBufferBuilder();
}

BufferBuilder *TensorBufferDialectInterface::getBuilderForTypeStorage(
    const BufferBuilderContext &builderContext, Type type) {
  if (auto interface =
          dyn_cast<TensorBufferDialectInterface>(&type.getDialect())) {
    DeviceRefAttr device = interface->getStorageDevice(builderContext, type);
    return builderContext.getBufferBuilder(device);
  }

  return builderContext.getHostBufferBuilder();
}

Value TensorBufferDialectInterface::matchingChain(
    const BufferBuilderContext &builderContext, OpBuilder &builder,
    Location loc, TypedValue<MGP::ContextType> context, Value inChain,
    Type typeToMatch) {
  bool inIsMef = isa<MEF::ChainType>(inChain.getType());
  bool matchIsMef = isa<MEF::ChainType>(typeToMatch);
  if (inChain.getType() == typeToMatch) {
    return inChain;
  } else if (inIsMef) {
    BufferBuilder *bufferBuilder =
        getBuilderForTypeStorage(builderContext, typeToMatch);
    return bufferBuilder->fromMEFChain(
        builder, loc, context, cast<TypedValue<MEF::ChainType>>(inChain));
  } else if (matchIsMef) {
    BufferBuilder *bufferBuilder =
        getBuilderForTypeStorage(builderContext, inChain.getType());
    return bufferBuilder->toMEFChain(builder, loc, context, inChain);
  } else {
    BufferBuilder *bufferBuilder =
        getBuilderForTypeStorage(builderContext, inChain.getType());
    auto mefChain = bufferBuilder->toMEFChain(builder, loc, context, inChain);
    bufferBuilder = getBuilderForTypeStorage(builderContext, typeToMatch);
    return bufferBuilder->fromMEFChain(
        builder, loc, context, cast<TypedValue<MEF::ChainType>>(mefChain));
  }
}

void TensorBufferDialectInterface::vtableAnchor() {}

//===----------------------------------------------------------------------===//
// CpuTensorBufferDialectInterface
//===----------------------------------------------------------------------===//

void CpuTensorBufferDialectInterface::addHostBuilder(
    ModelOp modelOp, BufferBuilderContext &builderContext) {
  // The type builder is easy.
  builderContext.registerTypeBuilder(builderContext.getHostDevice(),
                                     std::make_unique<CpuTypeBuilder>());

  // For the buffer builder, we need to know which slot holds the compiled
  // kernels context. We'll leave it as max value if the model has no complied
  // kernels.
  size_t slot = ~0;
  modelOp.walk([&slot](CpuKernelCreateContextOp createContextOp) {
    assert(slot == (size_t)~0 &&
           "MGP ModelOp contains multiple mgp.cpu.kernel.create_context ops");
    slot = createContextOp.getSlot();
  });

  auto cpuBufferBuilder = std::make_unique<CpuBufferBuilder>(slot);

  // For good hygiene, also recover the compiled kernel name to index map.
  modelOp.walk([cpuBufferBuilder = cpuBufferBuilder.get()](
                   CpuKernelLoadArchiveOp loadArchiveOp) {
    for (auto [index, name] :
         llvm::enumerate(loadArchiveOp.getCKernelNames())) {
      cpuBufferBuilder->trackCompiledKernel(name, index);
    }
  });

  builderContext.registerBufferBuilder(builderContext.getHostDevice(),
                                       std::move(cpuBufferBuilder));
}

void CpuTensorBufferDialectInterface::addNonHostBuilders(
    ModelOp modelOp, BufferBuilderContext &builderContext) {}

DeviceRefAttr CpuTensorBufferDialectInterface::getExecutionDevice(
    const BufferBuilderContext &builderContext, Operation *op) {
  return builderContext.getHostDevice();
}

DeviceRefAttr CpuTensorBufferDialectInterface::getStorageDevice(
    const BufferBuilderContext &builderContext, Type type) {
  return builderContext.getHostDevice();
}

} // namespace M::MGP
