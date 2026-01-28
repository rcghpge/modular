//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Helper for lowering MO graphs into device-dependent buffer-semantic form.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_INTERFACES_MOGGKERNELOPLOWERINGSTATE_H
#define GENERICML_INTERFACES_MOGGKERNELOPLOWERINGSTATE_H

#include "GenericML/GraphCompiler/MODialect/MOOps.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGOps.h"

namespace M::MGP {

// This class holds the state and methods needed to lower MOGG kernels to MGP
class MOGGKernelOpLoweringState {

public:
  MOGGKernelOpLoweringState(Operation *op,
                            MGP::MOGraphLoweringState &loweringState);

  // Lower an KernelOp to MOGG.
  LogicalResult lowerKernelOp(MOGG::KernelOp moggKernelOp);

private:
  mlir::Operation *moggOp;
  MGP::MOGraphLoweringState &loweringState;
  MGP::MOOpLoweringState opLoweringState;
  OpBuilder &builder;
  DeviceRefAttr device;

  /// Built output KernelStub for the mojo kernel implementation.
  MOGG::KernelStubOp kernelStub;

  /// Lowered output region.
  MOGG::GenericCompiledRegionOp mgpRegion;

  /// Block of the lowered output region.
  mlir::Block *mgpRegionBody = nullptr;

  // Staging all the necessary inputs to create the mgp.generic.compiled.region
  // op.
  SmallVector<Value> operands;
  SmallVector<InOutSignatureAttr::InOutSemantics> operandSemantics;

  // Map between a parameter to a lowered SSA value. We use TypedAttr instead of
  // ParamDeclRefAttr to make it possible to search integers (they won't be
  // present obviously).
  DenseMap<StringAttr, Value> parameterToOutsideSSAValue;

  // This maps an outside SSA value into its position in the operand list of the
  // mogg.generic.compiled.region.
  DenseMap<Value, uint64_t> outsideValueToOperandIndex;

  // This mapping is similar to `outsideValueToOperandIndex`.
  // But it covers the parameters needed inside the region stub.
  // Usually these are fused parameters of fused-intermediate tensors.
  // They are created inside the stub directly.
  DenseMap<Value, uint64_t> outsideValueToStubOperandIndex;

  // List of the operands for `callStubOp`.
  SmallVector<Value> stubCallOperands;

  // Created call op to the KernelStubOp.
  MOGG::CallStubOp callStubOp;

  // Add and map a new kernel operand.
  void addOperandAndMapToIndex(Value operand,
                               InOutSignatureAttr::InOutSemantics sig);

  // Get the dynamic value corresponding to the parametric dimension `dim`.
  // `isInsideStub` is true if we need it inside the kernel stub (and not the
  // region).
  ErrorOr<Value> getDynamicDimension(StringAttr dim, bool isInsideStub);

  // Get the list of dynamic values corresponding to the parametric dimensions
  // of the shape of `asTensor`. `inInsideTrue` is true if we need it inside the
  // kernel stub (and not the region).
  ErrorOr<SmallVector<Value>>
  getTensorDynamicDimensions(MO::TensorType asTensor, bool isInsideStub);

  // Register `dim` as an extra parametric dimension to be converted to a
  // dynamic value and passed to the kernel stub as argument.
  void passDynDimensionToKernelStub(Location loc, StringAttr dim);

  // Register all parametric dimensions of `asTensor` to be converted to dynamic
  // values and passed to the kernel stub as arguments.
  void passTensorDynDimensionsToKernelStub(Location loc,
                                           MO::TensorType asTensor);

  // Lower the DeviceContextPlaceholderOp to be compatible with the MGP region.
  void
  lowerDeviceContextPlaceholder(MOGG::DeviceContextPlaceholderOp moggContext);

  // Lower the BimdLambdaOp to be compatible with the MGP region.
  LogicalResult lowerBindLambdaOp(MOGG::BindLambdaOp bindOp);

  // Lower the ConstantOp to be compatible with the MGP region.
  void lowerConstantOp(MO::ConstantOp constantOp);

  // Lower the GetDimOp to be compatible with the MGP region.
  LogicalResult lowerGetDimOp(MOGG::GetDimOp getDimOp);

  // Lower the TensorGetShapeOp to be compatible with the MGP region.
  LogicalResult lowerTensorGetShapeOp(MOGG::TensorGetShapeOp getShapeOp);

  // Lower all kernel stubs ops to be compatible with the MGP region.
  LogicalResult lowerKernelStub();
};

} // namespace M::MGP

#endif // GENERICML_INTERFACES_MOGGKERNELOPLOWERINGSTATE_H
