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

  /// Returns the type builder appropriate for the device which will store
  /// values of type. Returns the builder for the 'host' device for types
  /// which don't have a specific device.
  static TypeBuilder *getTypeBuilder(const BufferBuilderContext &builderContext,
                                     Type type);

  /// Returns the buffer builder appropriate for the device which will execute
  /// op. Returns the builder for the 'host' device for ops which don't have
  /// a specific device.
  static BufferBuilder *
  getBuilderForOpExecution(const BufferBuilderContext &builderContext,
                           Operation *op);

  /// Returns the buffer builder appropriate for the device which will store
  /// values of type.
  static BufferBuilder *
  getBuilderForTypeStorage(const BufferBuilderContext &builderContext,
                           Type type);

  TensorBufferDialectInterface(Dialect *dialect) : Base(dialect) {}

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
