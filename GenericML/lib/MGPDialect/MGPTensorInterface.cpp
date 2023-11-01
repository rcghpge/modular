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

} // namespace M::MGP
