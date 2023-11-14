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

/// Interface for dialects which must manipulate device-specific tensors,
/// buffers and chain values.
struct TensorBufferDialectInterface
    : public mlir::DialectInterface::Base<TensorBufferDialectInterface> {

  /// Returns all the interfaces of this type active in the given MGP::ModelOp.
  static DenseSet<TensorBufferDialectInterface *>
  getAllInterfaces(ModelOp modelOp);

  /// Adds to builderContext builder for all the devices in the given model.
  static BufferBuilderContext getAllBufferBuilders(ModelOp modelOp);

  /// Returns the buffer builder appropriate for the device which will execute
  /// op.
  static BufferBuilder *
  getBuilderForOp(const BufferBuilderContext &builderContext, Operation *op);

  /// Returns the buffer builder appropriate for the device which will store
  /// values of type.
  static BufferBuilder *
  getBuilderForType(const BufferBuilderContext &builderContext, Type type);

  TensorBufferDialectInterface(Dialect *dialect) : Base(dialect) {}

  virtual MGPTensorExtractOpInterface
  createExtract(OpBuilder *builder, mlir::Location loc, Value tensor) = 0;
  /// Create a new chain that depends on the existing `chains`. Input op is the
  /// op the chains will be used with/from.
  virtual GRT::GRTChainCreateOpInterface createChain(OpBuilder *builder,
                                                     mlir::Location loc,
                                                     Operation *op,
                                                     ValueRange chains) = 0;
  /// Create a new chain that will protect access to the extracted buffer of a
  /// tensor.
  virtual GRT::GRTChainCreateOpInterface
  createChainForTensor(OpBuilder *builder, mlir::Location loc,
                       Value tensor) = 0;
  virtual Type createChainType(OpBuilder *builder, Type tensorType) = 0;
  virtual Type createBufferType(OpBuilder *builder, Type tensorType) = 0;

  /// Adds to BuilderContext the type and buffer builder for the 'host' device
  /// for this MGP::ModelOp if it is associated with this dialect.
  virtual void addHostBuilder(ModelOp modelOp,
                              BufferBuilderContext &builderContext) = 0;

  /// Adds to BuilderContext all the type and buffer builder instances
  /// for the devices (other than the host device) relevant to this dialect
  /// which are in use in the given MGP::ModelOp.
  virtual void addNonHostBuilders(ModelOp modelOp,
                                  BufferBuilderContext &builderContext) = 0;

  /// Returns the device on which the op will be executed, given the builder
  /// context for the entire MGP::ModelOp we are working on.
  virtual DeviceRefAttr
  getExecutionDevice(const BufferBuilderContext &builderContext,
                     Operation *op) = 0;

  /// Returns the device on which a value of type will be stored, given the
  /// builder context for the entire MGP::ModelOp we are working on.
  virtual DeviceRefAttr
  getStorageDevice(const BufferBuilderContext &builderContext, Type type) = 0;

  virtual void vtableAnchor();
};

//===----------------------------------------------------------------------===//
// CpuTensorBufferDialectInterface
//===----------------------------------------------------------------------===//

/// Concrete TensorBufferDialectInterface for the MGP dialect.
struct CpuTensorBufferDialectInterface : public TensorBufferDialectInterface {
  using TensorBufferDialectInterface::TensorBufferDialectInterface;

  virtual MGPTensorExtractOpInterface
  createExtract(OpBuilder *builder, mlir::Location loc, Value tensor) override;
  virtual GRT::GRTChainCreateOpInterface
  createChain(OpBuilder *builder, mlir::Location loc, Operation *op,
              ValueRange chains) override;
  virtual GRT::GRTChainCreateOpInterface
  createChainForTensor(OpBuilder *builder, mlir::Location loc,
                       Value tensor) override;
  virtual Type createChainType(OpBuilder *builder, Type tensorType) override;
  virtual Type createBufferType(OpBuilder *builder, Type tensorType) override;

  void addHostBuilder(ModelOp modelOp,
                      BufferBuilderContext &builderContext) override;
  void addNonHostBuilders(ModelOp modelOp,
                          BufferBuilderContext &builderContext) override;
  DeviceRefAttr getExecutionDevice(const BufferBuilderContext &builderContext,
                                   Operation *op) override;
  DeviceRefAttr getStorageDevice(const BufferBuilderContext &builderContext,
                                 Type type) override;
};

} // namespace M::MGP

#endif // GENERICML_MGPDIALECT_MGPTENSORINTERFACE_H
