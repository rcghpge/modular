//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Interfaces for tensors in the MGP dialect.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_MGPDIALECT_MGPTENSORINTERFACE_H
#define GENERICML_MGPDIALECT_MGPTENSORINTERFACE_H

#include "GenericML/MGPDialect/MGPTypes.h"
#include "GraphRT/GRTDialect/GRTChainInterface.h"
#include "mlir/IR/DialectInterface.h"
#include "mlir/IR/OpDefinition.h"

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "GenericML/MGPDialect/MGPTensorInterface.h.inc"
namespace M::MGP {

//===----------------------------------------------------------------------===//
// TensorBufferDialectInterface
//===----------------------------------------------------------------------===//

/// Interface for dialects that represent tensors as a buffer and a spec. Uses
/// chains to serialize access to a buffer.
struct TensorBufferDialectInterface
    : public mlir::DialectInterface::Base<TensorBufferDialectInterface> {
  TensorBufferDialectInterface(Dialect *dialect) : Base(dialect) {}
  virtual MGPTensorExtractOpInterface
  createExtract(OpBuilder *builder, mlir::Location loc, Value tensor) = 0;
  virtual GRT::GRTChainOpInterface
  createChain(OpBuilder *builder, mlir::Location loc, Value tensor) = 0;
  virtual Type createChainType(OpBuilder *builder) = 0;
  virtual Type createBufferType(OpBuilder *builder, Type spec) = 0;
  // virtual Type createTensorType(OpBuilder *builder, Type input) = 0;
};

struct CpuTensorBufferDialectInterface : public TensorBufferDialectInterface {
  using TensorBufferDialectInterface::TensorBufferDialectInterface;
  virtual MGPTensorExtractOpInterface
  createExtract(OpBuilder *builder, mlir::Location loc, Value tensor) override;
  virtual GRT::GRTChainOpInterface
  createChain(OpBuilder *builder, mlir::Location loc, Value tensor) override;
  virtual Type createChainType(OpBuilder *builder) override;
  virtual Type createBufferType(OpBuilder *builder, Type tensorType) override;
  // virtual Type createTensorType(OpBuilder *builder, Type input) override;
};
}; // namespace M::MGP

#endif // GENERICML_MGPDIALECT_MGPTENSORINTERFACE_H
