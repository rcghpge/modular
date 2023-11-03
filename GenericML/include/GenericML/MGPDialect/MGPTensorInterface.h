//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Dialect-level interfaces for tensors in the MGP dialect.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_MGPDIALECT_MGPTENSORINTERFACE_H
#define GENERICML_MGPDIALECT_MGPTENSORINTERFACE_H

#include "GenericML/MGPDialect/BufferBuilder.h"
#include "GenericML/MGPDialect/MGPOpInterfaces.h"
#include "GenericML/MGPDialect/MGPOps.h"
#include "GenericML/MGPDialect/MGPTypes.h"
#include "GraphRT/GRTDialect/GRTChainInterface.h"
#include "mlir/IR/DialectInterface.h"
#include "mlir/IR/OpDefinition.h"

namespace M::MGP {

//===----------------------------------------------------------------------===//
// TensorBufferDialectInterface
//===----------------------------------------------------------------------===//

/// Interface for dialects that represent tensors as a buffer and a spec. Uses
/// chains to serialize access to a buffer.
struct TensorBufferDialectInterface
    : public mlir::DialectInterface::Base<TensorBufferDialectInterface> {

  /// Returns all the dialect interfaces active in the given MGP::ModelOp.
  static DenseSet<TensorBufferDialectInterface *>
  getAllInterfaces(ModelOp modelOp);

  TensorBufferDialectInterface(Dialect *dialect) : Base(dialect) {}
  virtual MGPTensorExtractOpInterface
  createExtract(OpBuilder *builder, mlir::Location loc, Value tensor) = 0;
  /// Create a new chain that depends on the existing `chains`
  virtual GRT::GRTChainOpInterface
  createChain(OpBuilder *builder, mlir::Location loc, ValueRange chains) = 0;
  /// Create a new chain that will protect access to the extracted buffer of a
  /// tensor.
  virtual GRT::GRTChainOpInterface createChainForTensor(OpBuilder *builder,
                                                        mlir::Location loc,
                                                        Value tensor) = 0;
  virtual Type createChainType(OpBuilder *builder) = 0;
  virtual Type createBufferType(OpBuilder *builder, Type spec) = 0;

  /// Adds to BuilderContext all the type and buffer builder instances
  /// for the devices relevant to this dialect which are in use in the given
  /// MGP::ModelOp.
  virtual void addBuilders(ModelOp modelOp,
                           BufferBuilderContext &builderContext) = 0;

  /// Returns the device for the given op given the builder context for the
  /// MGP::ModelOp we are working on.
  virtual DeviceRefAttr getDevice(const BufferBuilderContext &builderContext,
                                  Operation *op) = 0;
};

struct CpuTensorBufferDialectInterface : public TensorBufferDialectInterface {
  using TensorBufferDialectInterface::TensorBufferDialectInterface;
  virtual MGPTensorExtractOpInterface
  createExtract(OpBuilder *builder, mlir::Location loc, Value tensor) override;
  virtual GRT::GRTChainOpInterface createChain(OpBuilder *builder,
                                               mlir::Location loc,
                                               ValueRange chains) override;
  virtual GRT::GRTChainOpInterface createChainForTensor(OpBuilder *builder,
                                                        mlir::Location loc,
                                                        Value tensor) override;
  virtual Type createChainType(OpBuilder *builder) override;
  virtual Type createBufferType(OpBuilder *builder, Type tensorType) override;

  void addBuilders(ModelOp modelOp,
                   BufferBuilderContext &builderContext) override;
  DeviceRefAttr getDevice(const BufferBuilderContext &builderContext,
                          Operation *op) override;
};

} // namespace M::MGP

#endif // GENERICML_MGPDIALECT_MGPTENSORINTERFACE_H
