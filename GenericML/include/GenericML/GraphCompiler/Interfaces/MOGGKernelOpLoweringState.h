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

// Helper class to keep track of the mapping between top-level value, and the
// corresponding MGP region / KernelStub argument mapping. It also keeps track
// of the mapping from parameter attributes to their corresponding dynamic
// values.
class KernelOperandsMapping {

public:
private:
  struct OperandValue {
    Value operandValue;
    BlockArgument mgpRegionOperand;
    BlockArgument kernelStubOperand;
    Type kernelStubOperandType;
    std::optional<unsigned> mgpRegionOperandIdx;
    std::optional<unsigned> kernelperandIdx;
  };

private:
  // List of all the kernel operands.
  llvm::SmallVector<OperandValue> operands;

  // Mapping from key to OperandValue (index of `operands` array).
  llvm::DenseMap<Value, unsigned> valToOperandsMap;

  // Mapping from parameter name to OperandValue (index of `operands` array).
  DenseMap<StringAttr, unsigned> parameterToOperandsMap;

  // Operands list to build the MGP Region.
  SmallVector<Value> mgpRegionOperands;
  SmallVector<InOutSignatureAttr::InOutSemantics> mgpRegionOperandSemantics;

  // Operands list for the KernelStub op.
  SmallVector<Value> kernelStubOperands;

public:
  ArrayRef<Value> getMGPRegionOperands() const { return mgpRegionOperands; }
  ArrayRef<InOutSignatureAttr::InOutSemantics>
  getMGPRegionOperandsSemantics() const {
    return mgpRegionOperandSemantics;
  }
  ArrayRef<Value> getKernelStubOperands() const { return kernelStubOperands; }

  // Returns the total number of operands (including MGP / Stub arguments).
  size_t getNumOperands() const { return operands.size(); }

  // Add a new kernel operand with value `operandValue`.
  // `isMGPRegionOperand` indicates the operand is an argument of the
  // mgp.compiled.region op. `isKernelStubOperand` indicates the operand is an
  // argument of the KernelStubOp. It'll be forwarded automatically from the MGP
  // region if needed. `sig` is the operand semantic, needed only if
  // `isMGPRegionOperand` is true. The mapped operand can be retrieved with
  // `get(operandValue, isInsideStub)`. If `keyValue` is set, it's used instead
  // as a key for retrieval: `get(keyValue, isInsideStub)`.
  // If set, `stubOperandType` change the type of the operand of the KernelStub.
  // This is only needed for special arguments like DeviceContext where the MGP
  // / MOGG types differs.
  void addOperand(Value operandValue, bool isMGPRegionOperand,
                  bool isKernelStubOperand,
                  InOutSignatureAttr::InOutSemantics sig =
                      InOutSignatureAttr::InOutSemantics::kNone,
                  Value keyValue = nullptr, Type stubOperandType = nullptr);

  // Add a new kernel operand for the parameter `paramName`.
  // `paramValue` is the corresponding runtime value of the parameter.
  // If `isKernelStubOperand`, the operand will also be forwarded automatically
  // to the KernelStub. The mapped operand can be retrieved with `get(paramName,
  // isInsideStub)`.
  void addOperand(KGEN::ParamDeclRefAttr paramName, Value paramValue,
                  bool isKernelStubOperand);

  // Change the value of a KernelStub operand.
  // `operandIdx is the global operand index (not just for the KernelStub).
  // This functions helps for operands that must be updated when passed between
  // the MGP -> KernelStub region.
  void remapKernelStubOperand(unsigned operandIdx, Value newValue);

  // Get the mapped operand #operandIdx.
  // `isInsideStub` is true if we need it inside the kernel stub (and not the
  // region).
  Value get(unsigned operandIdx, bool isInsideStub) const;

  // Get the mapped operand corresponding to `val`.
  // `isInsideStub` is true if we need it inside the kernel stub (and not the
  // region).
  Value get(Value val, bool isInsideStub) const;

  // Get the dynamic value corresponding to the parameter `paramName`.
  // `isInsideStub` is true if we need it inside the kernel stub (and not the
  // region).
  Value get(StringAttr paramName, bool isInsideStub) const;

  // Get the list of dynamic values corresponding to the parametric dimensions
  // of the shape of `asTensor`. `inInsideTrue` is true if we need it inside the
  // kernel stub (and not the region).
  SmallVector<Value> getTensorDynamicDimensions(MO::TensorType asTensor,
                                                bool isInsideStub);

  // Setup all the arguments of `mgpRegion` and update all mapping tables.
  // This must be done only after all operands where added with
  // `addOperand`.
  void setupMGPRegionArgs(Block *mgpRegion);

  // Setup all the arguments of `kernelStub` and update all mapping tables.
  // This must be done only after all operands where added with
  // `addOperand`.
  void setupKernelStubArgs(Block *kernelStubBody);
};

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

  KernelOperandsMapping operandsMapping;

  /// Built output KernelStub for the mojo kernel implementation.
  MOGG::KernelStubOp kernelStub;

  /// Lowered output region.
  MOGG::GenericCompiledRegionOp mgpRegion;

  /// Block of the lowered output region.
  mlir::Block *mgpRegionBody = nullptr;

  // Created call op to the KernelStubOp.
  MOGG::CallStubOp callStubOp;

  // This is a list of all the outputs of the kernel but lowered to MGP.
  SmallVector<MGP::MOOutputStorage> loweredOutputsStorage;

  struct TensorOperandInfos {
    Value stubValue;   // Tensor value inside the KernelStub
    size_t operandIdx; // Argument index for the OperandsMapping.
    bool isOutput;
  };

  // Keep track all inputs / output / scratchpad operands;
  SmallVector<TensorOperandInfos> tensorOperandInfos;

  /// Build a ManagedTensorSlice object corresponding to the KernelStubOp
  /// argument `stubArgument` from the buffer pointer `bufferValue`.
  Value buildMojoIOTensorObject(Value bufferValue,
                                const TensorOperandInfos &tensorInfos);

  // Look for extra arguments that must be passed from the mgp.model to the
  // mgp.region and / or the kernel stub. These includes dynamic dimensions,
  // extra buffers and device context objects.
  LogicalResult findExtraKernelArguments(
      Value chainValue, ArrayRef<KGEN::ParamDeclRefAttr> usedKernelParameters);

  // Lower the BimdLambdaOp to be compatible with the MGP region.
  void lowerBindLambdaOp(MOGG::BindLambdaOp bindOp);

  // Lower the ConstantOp to be compatible with the MGP region.
  void lowerConstantOp(MO::ConstantOp constantOp);

  // Lower the GetDimOp to be compatible with the MGP region.
  void lowerGetDimOp(MOGG::GetDimOp getDimOp);

  // Lower the TensorGetShapeOp to be compatible with the MGP region.
  void lowerTensorGetShapeOp(MOGG::TensorGetShapeOp getShapeOp);

  // Lower `op` to be compatible with the MGP region.
  // Simply use the OperandsMapping to replace the op.
  // This is needed for ops like OutputPlaceholder that are replaced with a
  // kernel argument.
  void lowerKernelOpWithOperandsMapping(Operation *op);

  // Lower all kernel stubs ops to be compatible with the MGP region.
  void lowerKernelStub();
};

} // namespace M::MGP

#endif // GENERICML_INTERFACES_MOGGKERNELOPLOWERINGSTATE_H
