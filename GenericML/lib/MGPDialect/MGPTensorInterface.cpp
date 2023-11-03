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

MGPTensorExtractOpInterface CpuTensorBufferDialectInterface::createExtract(
    OpBuilder *builder, mlir::Location loc, Value tensor) {
  return dyn_cast<MGPTensorExtractOpInterface>(
      *builder->create<CpuTensorExtractOp>(loc, tensor));
}
GRT::GRTChainOpInterface
CpuTensorBufferDialectInterface::createChain(OpBuilder *builder,
                                             mlir::Location loc, Value tensor) {
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

void CpuTensorBufferDialectInterface::addBuilders(
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

DeviceRefAttr CpuTensorBufferDialectInterface::getDevice(
    const BufferBuilderContext &builderContext, Operation *op) {
  return builderContext.getHostDevice();
}

} // namespace M::MGP
