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

#include "GraphCompiler/GraphCompiler/MODialect/MOOps.h"
#include "GraphCompiler/GraphCompiler/MOGGDialect/MOGGOps.h"

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
    std::optional<unsigned> kernelOperandIdx;
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

  // Returns the kernel-stub block-argument index for the operand at global
  // index `operandIdx` (the slot in the stub signature), or nullopt if the
  // operand is not a kernel-stub argument. Used to map a trait tensor operand
  // back to the stub argument it feeds, so emit can give that argument a
  // generic (T-inferred) type instead of `ManagedTensorSlice`.
  std::optional<unsigned> getKernelStubArgIndex(unsigned operandIdx) const {
    return operands[operandIdx].kernelOperandIdx;
  }

  // Returns the global operand index for `key`, or nullopt if not mapped.
  std::optional<unsigned> lookupOperandIdx(Value key) const {
    auto it = valToOperandsMap.find(key);
    if (it == valToOperandsMap.end())
      return std::nullopt;
    return it->second;
  }

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

/// Information about a kernel stub created during lowering.
struct KernelStubInfo {
  MOGG::KernelStubOp stub;
  StringAttr stubName;
  size_t numKernelInputs;
  MOGG::OutputPlaceholdersOp outputPlaceholdersOp;
  MOGG::ScratchPlaceholdersOp scratchPlaceholdersOp;
};

// This class holds the state and methods needed to lower MOGG kernels to MGP
class MOGGKernelOpLoweringState {

public:
  MOGGKernelOpLoweringState(Operation *op,
                            MGP::MOGraphLoweringState &loweringState);

  // Lower a KernelOp to MOGG.
  LogicalResult lowerKernelOp(MOGG::KernelOp moggKernelOp);

  // Accessors for subclasses/parallel lowering
  MGP::MOGraphLoweringState &getGraphLoweringState() { return loweringState; }
  MGP::MOOpLoweringState &getOpLoweringState() { return opLoweringState; }
  OpBuilder &getBuilder() { return builder; }
  KernelOperandsMapping &getOperandsMapping() { return operandsMapping; }
  MOGG::KernelStubOp getKernelStub() { return kernelStub; }
  // Returns the MGP region produced by `lowerKernelOp`. Only valid after
  // `lowerKernelOp` has succeeded.
  MOGG::GenericCompiledRegionOp getMGPRegion() { return mgpRegion; }
  DeviceRefAttr getDevice() { return device; }

  // Info about tensor operands used during kernel lowering.
  struct TensorOperandInfos {
    Value stubValue;   // Tensor value inside the KernelStub
    size_t operandIdx; // Argument index for the OperandsMapping.
    bool isOutput;
    // Trait lane (D1/D5): the distilled `#mogg.tensor_arg` contract for a
    // trait-bound argument, or null for a legacy `ManagedTensorSlice`
    // argument. When set, `buildMojoIOTensorObject` builds
    // `mogg.tensor_arg.__init__` carrying this contract instead of the legacy
    // `mogg.tensor.__init__`.
    MOGG::TensorArgContractAttr traitContract = nullptr;
  };

protected:
  // Create a kernel stub from a KernelOp, outlining its body.
  // Returns info about the created stub including placeholder ops.
  FailureOr<KernelStubInfo> createKernelStub(MOGG::KernelOp kernelOp,
                                             StringRef stubNamePrefix);

  // Set up the operands mapping for kernel inputs (chain + tensor inputs).
  // Populates operandsMapping and tensorOperandInfos for the inputs.
  void setupInputOperandsMapping(Value chain,
                                 ArrayRef<Value> loweredInputOperands);

  /// Build a ManagedTensorSlice object corresponding to the KernelStubOp
  /// argument `stubArgument` from the buffer pointer `bufferValue`.
  Value buildMojoIOTensorObject(Value bufferValue,
                                const TensorOperandInfos &tensorInfos);

  LogicalResult findExtraKernelArguments(
      Value chainValue, ArrayRef<KGEN::ParamDeclRefAttr> usedKernelParameters);

  void findExtraKernelStubOnlyArguments();

  /// Materialize `mogg.tensor.fused` from `mogg.bind` ops inside the stub.
  void materializeInStubTensorFusedOps();

  /// Wire fused bind fusibles / companions to the stub block-arg capture that
  /// the region passed in (after `setupKernelStubArgs`).
  void wireTraitFusedCapturesToStubArgs();

  void lowerKernelStub();

  // Keep track all inputs / output / scratchpad operands;
  SmallVector<TensorOperandInfos> tensorOperandInfos;

private:
  mlir::Operation *moggOp;
  MGP::MOGraphLoweringState &loweringState;
  MGP::MOOpLoweringState opLoweringState;
  OpBuilder &builder;
  DeviceRefAttr device;

  KernelOperandsMapping operandsMapping;

  /// Built output KernelStub for the mojo kernel implementation.
  MOGG::KernelStubOp kernelStub;

  // Lowered output region.
  MOGG::GenericCompiledRegionOp mgpRegion;

  /// Block of the lowered output region.
  mlir::Block *mgpRegionBody = nullptr;

  // Created call op to the KernelStubOp.
  MOGG::CallStubOp callStubOp;

  // This is a list of all the outputs of the kernel but lowered to MGP.
  SmallVector<MGP::MOOutputStorage> loweredOutputsStorage;

  // Lower the BindLambdaOp to be compatible with the MGP region.
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
};

} // namespace M::MGP

#endif // GENERICML_INTERFACES_MOGGKERNELOPLOWERINGSTATE_H
