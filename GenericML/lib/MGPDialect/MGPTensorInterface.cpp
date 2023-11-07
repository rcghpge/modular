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

BufferBuilder *TensorBufferDialectInterface::getBuilderForOp(
    const BufferBuilderContext &builderContext, Operation *op) {
  if (auto interface =
          dyn_cast<TensorBufferDialectInterface>(op->getDialect())) {
    DeviceRefAttr device = interface->getDevice(builderContext, op);
    return builderContext.getBufferBuilder(device);
  }

  return builderContext.getHostBufferBuilder();
}

MGPTensorExtractOpInterface CpuTensorBufferDialectInterface::createExtract(
    OpBuilder *builder, mlir::Location loc, Value tensor) {
  return dyn_cast<MGPTensorExtractOpInterface>(
      *builder->create<CpuTensorExtractOp>(loc, tensor));
}

GRT::GRTChainOpInterface CpuTensorBufferDialectInterface::createChain(
    OpBuilder *builder, mlir::Location loc, ValueRange chains) {
  return dyn_cast<GRT::GRTChainOpInterface>(
      *builder->create<GRT::ChainCreateOp>(loc, chains));
}

GRT::GRTChainOpInterface CpuTensorBufferDialectInterface::createChainForTensor(
    OpBuilder *builder, mlir::Location loc, Value tensor) {
  return dyn_cast<GRT::GRTChainOpInterface>(
      *builder->create<GRT::ChainCreateOp>(loc));
}

Type CpuTensorBufferDialectInterface::createChainType(OpBuilder *builder) {
  return builder->getType<MEF::ChainType>();
}

Type CpuTensorBufferDialectInterface::createBufferType(OpBuilder *builder,
                                                       Type tensorType) {
  return CpuBufferType::get(
      dyn_cast<CpuTensorType>(tensorType).getTensorSpec());
}

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

DeviceRefAttr CpuTensorBufferDialectInterface::getDevice(
    const BufferBuilderContext &builderContext, Operation *op) {
  return builderContext.getHostDevice();
}

} // namespace M::MGP
