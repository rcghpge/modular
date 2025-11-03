//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/Interfaces/EmitMojoState.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GenericML/GraphCompiler/MOGGDialect/Support/MOToMOGGLowering.h"
#include "GenericML/GraphCompiler/MOGGDialect/Support/MojoBuilder.h"
#include "GenericML/MGPDialect/MGPOps.h"
#include "GenericML/MGPDialect/MGPTypes.h"
#include "GenericML/Support/Diagnostic.h"
#include "KGEN/LITDialect/LITOps.h"
#include "Support/AssertStream.h"
#include "Support/Compiler/VerifyUtils.h"
#include "Support/ErrorOr.h"
#include "Support/MDialect/MAttrs.h"
#include "Support/MDialect/ParserUtils.h"
#include "Support/ML/CompiledFrameworkLabel.h"
#include "Support/ML/SizeUtils.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Base64.h"

using namespace M;
using namespace MGP;

//===----------------------------------------------------------------------===//
// Utils
//===----------------------------------------------------------------------===//

/// Assigns to device from attribute with name in attrs. Assigns <"cpu", 0>
/// if no such attribute.
static void getDeviceAttrOrDefault(MLIRContext *ctx, DeviceRefAttr &device) {
  if (!device)
    device = DeviceRefAttr::getDefault(ctx);
}

/// Returns the raw alignment for an elements attribute.
static uint64_t getRawAlign(ElementsAttr value) {
  ErrorOr<std::optional<uint64_t>> errOrOptAlign = getImpliedAlignment(value);
  if (errOrOptAlign.isError())
    return 0;

  // If error we'll report it when the result is validated.
  return optSizeToRawSize(*errOrOptAlign);
}

//===----------------------------------------------------------------------===//
// RuntimeCreateOp
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// DeviceContextProfileStartOp
//===----------------------------------------------------------------------===//

LogicalResult DeviceContextProfileStartOp::verify() {
  return verifyConsistentDevices(getOperation(), {getBDevice()});
}

//===----------------------------------------------------------------------===//
// DeviceContextProfileEndOp
//===----------------------------------------------------------------------===//

LogicalResult DeviceContextProfileEndOp::verify() {
  return verifyConsistentDevices(getOperation(), {getBDevice()});
}

//===----------------------------------------------------------------------===//
// ModuleLoadOp
//===----------------------------------------------------------------------===//

LogicalResult ModuleLoadOp::verify() {
  return verifyConsistentDevices(getOperation(), {getEDevice()});
}

//===----------------------------------------------------------------------===//
// BufferConstantExternalOp
//===----------------------------------------------------------------------===//

LogicalResult BufferConstantExternalOp::verify() {
  uint64_t align = getDAlign();
  if (!isValidOptAlign(align))
    return emitOpError("invalid alignment ") << align << ".";

  return success();
}

LogicalResult BufferConstantExternalOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, Adaptor adaptor,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  DeviceRefAttr device = adaptor.getEDeviceAttr();
  getDeviceAttrOrDefault(context, device);
  inferredReturnTypes.emplace_back(DeviceBufferType::get(context, device));
  return success();
}

//===----------------------------------------------------------------------===//
// BufferConstantOp
//===----------------------------------------------------------------------===//

void BufferConstantOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                             ElementsAttr aValue) {
  return build(odsBuilder, odsState, aValue, getRawAlign(aValue));
}

/// Verifies the attribute and result types are consistent, and the attribute
/// is supported.
LogicalResult BufferConstantOp::verify() {
  // We can only construct from attributes which are dense arrays (ie rank-1)
  // of uint8_ts.
  ElementsAttr value = getAValue();
  ShapedType shapedType = cast<ShapedType>(value.getType());
  if (value.size() != shapedType.getNumElements())
    return emitOpError("does not support splat attributes.");
  if (!shapedType.hasRank() || shapedType.getShape().size() != 1)
    return emitOpError("only supports rank-1 attributes.");
  Type elemType = value.getElementType();
  if (!elemType.isUnsignedInteger(8))
    return emitOpError("only supports attributes of uint8_t elements")
           << " (have " << elemType << ").";

  // Only BufferAttr and DenseResourceElementsAttr can convey an align
  // constraint. Check they are valid.
  ErrorOr<std::optional<uint64_t>> errOrOptAlign = getImpliedAlignment(value);
  if (auto *error = errOrOptAlign.getError())
    return emitOpError(error);

  if (!isValidOptAlign(*errOrOptAlign))
    return emitOpError("invalid alignment ") << **errOrOptAlign << ".";

  // The implied alignment must match the actual attribute alignment.
  if (getOptAlign().has_value() != errOrOptAlign->has_value()) {
    return emitOpError("expected alignment operand if and only if value "
                       "operand has alignment.");
  }
  if (getOptAlign().has_value() && *getOptAlign() != **errOrOptAlign) {
    return emitOpError(
               "expected operand alignment to match attribute's alignment (")
           << *getOptAlign() << " vs " << **errOrOptAlign << ").";
  }

  return success();
}

LogicalResult BufferConstantOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, Adaptor adaptor,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  DeviceRefAttr device = DeviceRefAttr::getDefault(context);
  inferredReturnTypes.emplace_back(DeviceBufferType::get(context, device));
  return success();
}

std::optional<uint64_t> BufferConstantOp::getOptAlign() {
  return rawSizeToOptSize(getBRawAlign());
}

//===----------------------------------------------------------------------===//
// BufferAllocOp
//===----------------------------------------------------------------------===//
bool BufferAllocOp::isRaising() { return true; }

LogicalResult BufferAllocOp::emitMojo(MOGG::EmitMojoState &state) {
  using namespace MOGG;

  KGEN::LIT::FnOp litFunction =
      state.analysis.getRegisteredLITMojoKernelFor(getOperationName());
  MojoValue function = state.getOriginalMojoNameFor(litFunction);

  MojoValue byteSizeVar = state.getMojoValueFor(getRuntimeSize());
  MojoValue deviceCtxVar = state.getMojoValueFor(getDeviceContext());

  MojoValue result =
      state.builder.var(function({}, {byteSizeVar, deviceCtxVar}));
  state.setMojoValueFor(getResult(), result);

  return success();
}

LogicalResult BufferAllocOp::verify() {
  if (getCDevice() != getOutBuffer().getType().getDevice()) {
    return emitOpError("expected result buffer device does not match actual "
                       "result buffer device (")
           << getCDevice() << " vs " << getOutBuffer().getType().getDevice()
           << ").";
  }
  if (!isValidRawAlign(getBRawAlign()))
    return emitOpError("invalid alignment ") << getBRawAlign() << ".";
  return verifyConsistentDevices(getOperation(), {getCDevice()});
}

LogicalResult BufferAllocOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, Adaptor adaptor,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  DeviceRefAttr device = adaptor.getCDevice();
  getDeviceAttrOrDefault(context, device);
  inferredReturnTypes.emplace_back(DeviceBufferType::get(context, device));
  return success();
}

void BufferAllocOp::setInChain(Value chain) {
  getInChainMutable().assign(chain);
}

//===----------------------------------------------------------------------===//
// BufferHostToDeviceOp
//===----------------------------------------------------------------------===//

LogicalResult BufferHostToDeviceOp::verify() {
  if (getHostBuffer().getType().getDevice() != getCHostDevice())
    return emitOpError("expected host buffer to be on 'host' device ")
           << getCHostDevice() << ".";
  if (getOtherBuffer().getType().getDevice() != getDOtherDevice())
    return emitOpError("expected other buffer to be on 'other' device ")
           << getDOtherDevice() << ".";
  return success();
}

InOutSignatureAttr::InOutSemantics
BufferHostToDeviceOp::getInOutSemantics(size_t index) {
  assert(index < getNumOperands());
  return index == getOtherBufferMutable().getOperandNumber()
             ? InOutSignatureAttr::kOut
             : InOutSignatureAttr::kIn;
}

//===----------------------------------------------------------------------===//
// BufferDeviceToHostOp
//===----------------------------------------------------------------------===//

LogicalResult BufferDeviceToHostOp::verify() {
  if (getHostBuffer().getType().getDevice() != getDHostDevice())
    return emitOpError("expected host buffer to be on 'host' device ")
           << getDHostDevice() << ".";
  if (getOtherBuffer().getType().getDevice() != getCOtherDevice())
    return emitOpError("expected other buffer to be on 'other' device ")
           << getCOtherDevice() << ".";
  return success();
}

InOutSignatureAttr::InOutSemantics
BufferDeviceToHostOp::getInOutSemantics(size_t index) {
  assert(index < getNumOperands());
  return index == getHostBufferMutable().getOperandNumber()
             ? InOutSignatureAttr::kOut
             : InOutSignatureAttr::kIn;
}

//===----------------------------------------------------------------------===//
// BufferDeviceToDeviceOp
//===----------------------------------------------------------------------===//

LogicalResult BufferDeviceToDeviceOp::verify() {
  if (getCSrcDevice().getLabel() != getDDstDevice().getLabel()) {
    return emitOpError("expecting source and destination devices to have the "
                       "same device label.");
  }
  if (getSrcBuffer().getType().getDevice() != getCSrcDevice() ||
      getDstBuffer().getType().getDevice() != getDDstDevice()) {
    return emitOpError(
               "expecting input chain and source buffer to be on device '")
           << getCSrcDevice()
           << "', and output chain and destination buffer to be on device '"
           << getDDstDevice() << "'.";
  }
  return success();
}

InOutSignatureAttr::InOutSemantics
BufferDeviceToDeviceOp::getInOutSemantics(size_t index) {
  assert(index < getNumOperands());
  return index == getDstBufferMutable().getOperandNumber()
             ? InOutSignatureAttr::kOut
             : InOutSignatureAttr::kIn;
}

//===----------------------------------------------------------------------===//
// BufferConcatOp
//===----------------------------------------------------------------------===//

LogicalResult BufferConcatOp::verify() {
  if (getOutBuffer().getType().getDevice() != getBDevice()) {
    return emitOpError("expected output buffer device does not match actual "
                       "output buffer device (")
           << getBDevice() << " vs " << getOutBuffer().getType().getDevice()
           << ").";
  }
  return verifyConsistentDevices(getOperation(), {getBDevice()});
}

InOutSignatureAttr::InOutSemantics
BufferConcatOp::getInOutSemantics(size_t index) {
  assert(index < getNumOperands());
  return index == getOutBufferMutable().getOperandNumber()
             ? InOutSignatureAttr::kOut
             : InOutSignatureAttr::kIn;
}

//===----------------------------------------------------------------------===//
// BufferSliceOp
//===----------------------------------------------------------------------===//

LogicalResult BufferSliceOp::verify() { return success(); }

LogicalResult BufferSliceOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, Adaptor adaptor,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  auto inBufferTy = dyn_cast<DeviceBufferType>(adaptor.getInBuffer().getType());
  if (!inBufferTy)
    return failure();
  inferredReturnTypes.emplace_back(
      DeviceBufferType::get(context, inBufferTy.getDevice()));
  return success();
}

bool BufferSliceOp::extendsLifetime(size_t operandIndex, size_t resultIndex) {
  return operandIndex == getInBufferMutable().getOperandNumber();
}

//===----------------------------------------------------------------------===//
// BufferToIndexOp
//===----------------------------------------------------------------------===//
bool BufferToIndexOp::isRaising() { return true; }

LogicalResult BufferToIndexOp::emitMojo(MOGG::EmitMojoState &state) {
  using namespace MOGG;

  KGEN::LIT::FnOp litFunction =
      state.analysis.getRegisteredLITMojoKernelFor(getOperationName());
  MojoValue function = state.getOriginalMojoNameFor(litFunction);

  MojoValue bufferVar = state.getMojoValueFor(getInBuffer());

  MojoValue result = state.builder.var(function({}, {bufferVar}));

  state.setMojoValueFor(getOut(), result);

  return success();
}

//===----------------------------------------------------------------------===//
// BufferToBoolOp
//===----------------------------------------------------------------------===//
LogicalResult BufferToBoolOp::emitMojo(MOGG::EmitMojoState &state) {
  using namespace MOGG;
  KGEN::LIT::FnOp litFunction =
      state.analysis.getRegisteredLITMojoKernelFor(getOperationName());
  MojoValue function = state.getOriginalMojoNameFor(litFunction);

  DeviceBufferType bufferType = getBuffer().getType();
  MojoValue deviceParam = MojoValue(bufferType.getDevice().getLabel()).quote();

  state.setMojoValueFor(
      getResult(),
      function({deviceParam}, {state.getMojoValueFor(getBuffer())}));
  return success();
}

LogicalResult BufferToBoolOp::verify() {
  return verifyConsistentDevices(getOperation(), {getBDevice()});
}

//===----------------------------------------------------------------------===//
// BufferSetWithIndexOp
//===----------------------------------------------------------------------===//
bool BufferSetWithIndexOp::isRaising() { return true; }

LogicalResult BufferSetWithIndexOp::emitMojo(MOGG::EmitMojoState &state) {
  using namespace MOGG;

  KGEN::LIT::FnOp litFunction =
      state.analysis.getRegisteredLITMojoKernelFor(getOperationName());
  MojoValue function = state.getOriginalMojoNameFor(litFunction);

  DeviceBufferType bufferType = getBuffer().getType();
  MojoValue deviceParam = MojoValue(bufferType.getDevice().getLabel()).quote();

  SmallVector<MojoValue> args;
  args.reserve(getNumOperands());
  args.emplace_back(state.getMojoValueFor(getBuffer()));
  args.append(state.getMojoValuesFor(getValues()));

  state.builder.call(function, {deviceParam}, args);

  return success();
}

LogicalResult BufferSetWithIndexOp::verify() {
  return verifyConsistentDevices(getOperation(), {getBDevice()});
}

InOutSignatureAttr::InOutSemantics
BufferSetWithIndexOp::getInOutSemantics(size_t index) {
  assert(index < getNumOperands());
  return index == getBufferMutable().getOperandNumber()
             ? InOutSignatureAttr::kOut
             : InOutSignatureAttr::kIn;
}

//===----------------------------------------------------------------------===//
// BufferPlanOp
//===----------------------------------------------------------------------===//

LogicalResult BufferPlanOp::verify() {
  if (!llvm::all_equal({getRuntimeSizes().size(), getOffsets().size(),
                        getAAlignments().size(), getBMinPre().size(),
                        getCMinPost().size(), getDMaxPre().size(),
                        getEMaxPost().size()}))
    return emitOpError("expected number of sizes, output offsets, alignments "
                       "and lifetime info attributes to "
                       "be equal.");
  return success();
}

LogicalResult BufferPlanOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, Adaptor adaptor,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  ValueRange operands = adaptor.getOperands();
  // All outputs are index
  inferredReturnTypes.clear();
  inferredReturnTypes.resize(operands.size() + 1, IndexType::get(context));

  return success();
}

//===----------------------------------------------------------------------===//
// TensorCreateOp
//===----------------------------------------------------------------------===//

LogicalResult TensorCreateOp::verify() {
  auto expectedOutTensorType = DeviceTensorType::get(
      getBuffer().getType().getDevice(), getTensorSpec().getType());
  if (expectedOutTensorType != getOutTensor().getType()) {
    return emitOpError("expected output tensor type does not match "
                       "actual output tensor type (")
           << expectedOutTensorType << " vs " << getOutTensor().getType()
           << ").";
  }
  return success();
}

LogicalResult TensorCreateOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, Adaptor adaptor,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  auto tensorSpecType =
      dyn_cast<TensorSpecType>(adaptor.getTensorSpec().getType());
  if (!tensorSpecType) {
    return mlir::emitError(location.value_or(adaptor.getInChain().getLoc()),
                           "expected tensor spec operand.");
  }

  DeviceRefAttr device =
      cast<DeviceBufferType>(adaptor.getBuffer().getType()).getDevice();
  inferredReturnTypes.emplace_back(
      DeviceTensorType::get(device, tensorSpecType));
  return success();
}

bool TensorCreateOp::mayAlias(size_t operandIndex, size_t resultIndex) {
  return operandIndex == getBufferMutable().getOperandNumber();
}

//===----------------------------------------------------------------------===//
// DebugTensorPrintOp
//===----------------------------------------------------------------------===//
bool DebugTensorPrintOp::isRaising() { return true; }

LogicalResult DebugTensorPrintOp::emitMojo(MOGG::EmitMojoState &state) {
  KGEN::LIT::FnOp litFunction =
      state.analysis.getRegisteredLITMojoKernelFor(getOperationName());
  MOGG::MojoValue function = state.getOriginalMojoNameFor(litFunction);

  auto tensorSpec = getSpec().getType();
  MOGG::MojoValue specRankParam = MOGG::MojoValue(int(tensorSpec.getRank()));
  MOGG::MojoValue dtypeParam =
      state.mlirDTypeToMojoType(tensorSpec.getOptDType().value());

  MOGG::MojoValue tensorSpecAsyncVar = state.getMojoValueFor(getSpec());
  MOGG::MojoValue bufferVar = state.getMojoValueFor(getBuffer());
  MOGG::MojoValue labelLiteral = MOGG::MojoValue(getBLabel()).quote();
  MOGG::MojoValue labelString = state.builder.var(MOGG::MojoValue(
      MOGG::HardcodedMojoSymbols::struct_String)({}, {labelLiteral}));
  MOGG::MojoValue labelPtrVar = labelString.get("unsafe_ptr")();
  MOGG::MojoValue labelSizeVar = labelString.get("__len__")();

  state.builder.call(
      function, {specRankParam, dtypeParam},
      {bufferVar, tensorSpecAsyncVar, labelPtrVar, labelSizeVar});

  return success();
}
