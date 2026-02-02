//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements the MGP dialect operations.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/MGPDialect/Dialect/MGPOps.h"
#include "GenericML/GraphCompiler/Interfaces/EmitMojoState.h"
#include "GenericML/GraphCompiler/MGPDialect/Dialect/BufferBuilder.h"
#include "GenericML/GraphCompiler/MGPDialect/Dialect/MGPTypes.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GenericML/GraphCompiler/MOGGDialect/Support/MojoBuilder.h"
#include "GenericML/GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/POPDialect/POPOps.h"
#include "KGEN/POPDialect/POPTypes.h"
#include "MLRT/GraphRT/GRTDialect/GRTOps.h"
#include "Support/AssertStream.h"
#include "Support/Compiler/VerifyUtils.h"
#include "Support/ErrorOr.h"
#include "Support/LogicalResult.h"
#include "Support/MDialect/MAttrs.h"
#include "Support/ML/CompiledFrameworkLabel.h"
#include "Support/ML/SizeUtils.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/Support/Base64.h"
#include <string>

using namespace M;
using namespace MGP;

//===----------------------------------------------------------------------===//
// Utils
//===----------------------------------------------------------------------===//

ErrorOr<std::optional<uint64_t>>
M::MGP::getImpliedAlignment(ElementsAttr value) {
  if (auto resourceAttr = dyn_cast<DenseResourceElementsAttr>(value)) {
    DenseResourceElementsHandle handle = resourceAttr.getRawHandle();
    mlir::AsmResourceBlob *blob = handle.getBlob();
    if (!blob && !isa<AlignedBytesType>(resourceAttr.getType())) {
      // Report a user-friendly error in this situation.
      return Error(Twine("dense_resource blob is not available, so require "
                         "type annotation to supply an alignment"));
    }

    if (blob) {
      // Since the blob is available, check its size is consistent with the
      // attribute size.
      if (static_cast<int64_t>(blob->getData().size()) !=
          value.getNumElements())
        return Error(Twine("the resource size does not match the "
                           "attribute's element size (" +
                           Twine(blob->getData().size()) + " vs " +
                           Twine(value.getNumElements()) + ")"));
    }
    // DenseResourceElementsAttrs support HasAlignedBytesInterface, so fall
    // through.
  }

  if (auto hasAlignedBytesAttr = dyn_cast<HasAlignedBytesInterface>(value)) {
    // The attribute has an alignment or default alignment we can use.
    return hasAlignedBytesAttr.getAlignedBytesType().getAlign();
  }

  return Error("cannot supply an alignment from type annotation for "
               "this attribute value.");
}

/// Verifies the types of all operands of op uses devices drawn from the
/// given devices.
LogicalResult M::MGP::verifyConsistentDevices(
    Operation *op, std::initializer_list<DeviceRefAttr> expectedDevices) {
  llvm::SmallSetVector<DeviceRefAttr, 2> allDevices;
  for (auto device : expectedDevices)
    allDevices.insert(device);
  size_t expectedNumDevices = allDevices.size();
  std::string expectedStr;
  {
    llvm::raw_string_ostream os(expectedStr);
    llvm::interleaveComma(allDevices, os,
                          [&os](DeviceRefAttr device) { os << device; });
  }

  for (OpOperand &opOperand : op->getOpOperands()) {
    if (auto hasDeviceType =
            dyn_cast<MGPHasDeviceTypeInterface>(opOperand.get().getType()))
      allDevices.insert(hasDeviceType.getDevice());
  }
  for (OpResult opResult : op->getResults()) {
    if (auto hasDeviceType =
            dyn_cast<MGPHasDeviceTypeInterface>(opResult.getType()))
      allDevices.insert(hasDeviceType.getDevice());
  }

  if (allDevices.size() == expectedNumDevices)
    return success();

  std::string actualStr;
  {
    llvm::raw_string_ostream os(actualStr);
    llvm::interleaveComma(allDevices, os,
                          [&os](DeviceRefAttr device) { os << device; });
  }

  return op->emitOpError("expected all operands, results, and attributes to "
                         "use the device(s) {")
         << expectedStr << "}, but they instead use the devices(s) {"
         << actualStr << "}.";
}

//===----------------------------------------------------------------------===//
// Custom parsers
//===----------------------------------------------------------------------===//

/// Parses a !mgp.buffer type, extracts its device attribute,
/// assigns chains the type !mgp.chain<device>.
static ParseResult parseBufferTypeToDeviceImpl(OpAsmParser &p, Type &buffer,
                                               DeviceRefAttr &device,
                                               ArrayRef<Type *> chains) {
  llvm::SMLoc loc;
  if (p.getCurrentLocation(&loc) || p.parseType(buffer))
    return failure();
  auto bufferType = dyn_cast<DeviceBufferType>(buffer);
  if (bufferType == nullptr)
    return p.emitError(loc) << "expected an !mgp.buffer type.";
  device = bufferType.getDevice();
  for (Type *chain : chains)
    *chain = MEF::ChainType::get(p.getContext());
  return success();
}

static ParseResult parseBufferTypeToDevice(OpAsmParser &p, Type &buffer,
                                           DeviceRefAttr &device) {
  return parseBufferTypeToDeviceImpl(p, buffer, device, {});
}

static void printBufferTypeToDevice(OpAsmPrinter &p, Operation *op,
                                    DeviceBufferType buffer,
                                    DeviceRefAttr device) {
  p << buffer;
}

static ParseResult parseBufferTypeToDevice(OpAsmParser &p, Type &buffer,
                                           DeviceRefAttr &device, Type &chain) {
  return parseBufferTypeToDeviceImpl(p, buffer, device, {&chain});
}

static void printBufferTypeToDevice(OpAsmPrinter &p, Operation *op,
                                    DeviceBufferType buffer,
                                    DeviceRefAttr device,
                                    MEF::ChainType chain) {
  p << buffer;
}

static ParseResult parseBufferTypeToChain(OpAsmParser &p, Type &buffer,
                                          Type &chain) {
  DeviceRefAttr device;
  return parseBufferTypeToDeviceImpl(p, buffer, device, {&chain});
}

static void printBufferTypeToChain(OpAsmPrinter &p, Operation *op,
                                   DeviceBufferType buffer,
                                   MEF::ChainType chain) {
  p << buffer;
}

/// Parses a !mgp.tensor type and extracts its device, buffer type, and
/// tensor spec type, and assigns chain the type !mgp.chain<device>.
/// If isShapeOnly is true, the tensor spec will have 'unk' dtype.
static ParseResult parseTensorTypeToDeviceBufferSpecImpl(
    OpAsmParser &p, Type &tensor, DeviceRefAttr &device, Type &buffer,
    Type &tensorSpec, ArrayRef<Type *> chains) {
  llvm::SMLoc loc;
  if (p.getCurrentLocation(&loc) || p.parseType(tensor))
    return failure();
  auto tensorType = dyn_cast<DeviceTensorType>(tensor);
  if (tensorType == nullptr)
    return p.emitError(loc) << "expected an !mgp.tensor type.";
  device = tensorType.getDevice();

  tensorSpec = tensorType.getTensorSpec();
  buffer = DeviceBufferType::get(tensorType);
  for (Type *chain : chains)
    *chain = MEF::ChainType::get(p.getContext());
  return success();
}

static ParseResult parseTensorTypeToDeviceBufferSpec(OpAsmParser &p,
                                                     Type &tensor, Type &buffer,
                                                     Type &tensorSpec,
                                                     Type &chain) {
  DeviceRefAttr device;
  return parseTensorTypeToDeviceBufferSpecImpl(p, tensor, device, buffer,
                                               tensorSpec, {&chain});
}

static void printTensorTypeToDeviceBufferSpec(OpAsmPrinter &p, Operation *op,
                                              DeviceTensorType tensorType,
                                              DeviceBufferType buffer,
                                              TensorSpecType tensorSpec,
                                              MEF::ChainType chain) {
  p << tensorType;
}

/// Parses a function type consisting of one input buffer and one output buffer
/// type, assigns their devices to srcDevice and dstDevice, and assigns inChain
/// the type !mgp.chain<srcDevice>.
static ParseResult
parseSrcDstBufferTypesToDevices(OpAsmParser &p, Type &srcBuffer,
                                Type &dstBuffer, DeviceRefAttr &srcDevice,
                                DeviceRefAttr &dstDevice, Type &inChain) {
  mlir::SMLoc loc;
  if (p.getCurrentLocation(&loc) || p.parseType(srcBuffer))
    return failure();
  auto srcBufferType = dyn_cast<DeviceBufferType>(srcBuffer);
  if (!srcBufferType)
    return p.emitError(loc, "expected an !mgp.buffer type");
  if (p.parseArrow())
    return failure();
  if (p.getCurrentLocation(&loc) || p.parseType(dstBuffer))
    return failure();
  auto dstBufferType = dyn_cast<DeviceBufferType>(dstBuffer);
  if (!dstBufferType)
    return p.emitError(loc, "expected an !mgp.buffer type");
  srcDevice = srcBufferType.getDevice();
  dstDevice = dstBufferType.getDevice();
  inChain = MEF::ChainType::get(p.getContext());
  return success();
}

static void printSrcDstBufferTypesToDevices(OpAsmPrinter &p, Operation *op,
                                            DeviceBufferType srcBuffer,
                                            DeviceBufferType dstBuffer,
                                            DeviceRefAttr srcDevice,
                                            DeviceRefAttr dstDevice,
                                            MEF::ChainType inChain) {
  p << srcBuffer;
  p << " -> ";
  p << dstBuffer;
}

//===----------------------------------------------------------------------===//
// ResourceConstantOp
//===----------------------------------------------------------------------===//

LogicalResult ResourceConstantOp::emitMojo(MOGG::EmitMojoState &state) {
  KGEN::LIT::FnOp func = state.analysis.getRegisteredLITMojoKernelFor(
      BufferConstantOp::getOperationName());
  MOGG::MojoValue function = state.getOriginalMojoNameFor(func);
  auto &[ptr, bytecount] =
      state.mojoConstantResources[getResourceSlot().getZExtValue()];
  MOGG::MojoValue result = state.builder.var(function({}, {ptr, bytecount}));
  state.setMojoValueFor(getOutBuffer(), result);

  return success();
}

//===----------------------------------------------------------------------===//
// BufferConstantOp
//===----------------------------------------------------------------------===//

/// Parses an elements attribute, and sets rawAlign to its alignment constraint.
static ParseResult parseElementsAttrWithAlign(AsmParser &p, ElementsAttr &value,
                                              IntegerAttr &rawAlign) {
  llvm::SMLoc loc;
  Attribute anyAttr;
  if (p.getCurrentLocation(&loc) || p.parseAttribute(anyAttr))
    return failure();
  value = dyn_cast<ElementsAttr>(anyAttr);
  if (!value)
    return p.emitError(loc, "expected elements attribute");

  ErrorOr<std::optional<uint64_t>> errOrOptAlign = getImpliedAlignment(value);
  if (const char *err = errOrOptAlign.getError())
    return p.emitError(loc, err);

  Type ui64 = IntegerType::get(p.getContext(), 64, IntegerType::Unsigned);
  APInt apint(ui64.getIntOrFloatBitWidth(), optSizeToRawSize(*errOrOptAlign),
              ui64.isSignedInteger());
  rawAlign = IntegerAttr::get(ui64, apint);
  return success();
}

static void printElementsAttrWithAlign(AsmPrinter &p, Operation *op,
                                       ElementsAttr value,
                                       IntegerAttr rawAlign) {
  p << value;
}

llvm::BitVector BufferConstantOp::resultsToPack() {
  return llvm::BitVector(1, false);
}

LogicalResult BufferConstantOp::emitMojo(MOGG::EmitMojoState &state) {
  KGEN::LIT::FnOp func =
      state.analysis.getRegisteredLITMojoKernelFor(getOperationName());
  MOGG::MojoValue function = state.getOriginalMojoNameFor(func);

  auto elementsAttr = getAValue();
  auto kernelOp = getOperation()->getParentOfType<MOGG::KernelStubOp>();
  auto constantResources = kernelOp.getConstantResources();
  auto it = llvm::find(constantResources, elementsAttr);
  if (it == constantResources.end())
    return emitError("The constant resource can't be found in the kernel");

  uint64_t resourceIndex = std::distance(constantResources.begin(), it);
  auto &[ptr, bytecount] = state.mojoConstantResources[resourceIndex];

  MOGG::MojoValue result = state.builder.var(function({}, {ptr, bytecount}));
  state.setMojoValueFor(getOutBuffer(), result);

  return success();
}

//===----------------------------------------------------------------------===//
// TensorSpecCreateOp
//===----------------------------------------------------------------------===//

LogicalResult TensorSpecCreateOp::emitMojo(MOGG::EmitMojoState &state) {
  using namespace MOGG;

  KGEN::LIT::FnOp func =
      state.analysis.getRegisteredLITMojoKernelFor(getOperationName());
  MOGG::MojoValue function = state.getOriginalMojoNameFor(func);

  TensorSpecType tensorSpec = getResult().getType();
  size_t rank = tensorSpec.getRank();
  MojoValue rawDimsRankParam = rank;

  // TODO:Update mgp_tensor_spec_create to use DimList directly, which will let
  // us use EmitMojoState::getAsDimListMojoValue. Currently, the behavior of the
  // mojo function is incompatible with the EmitMojoState function and the mojo
  // function can't be updated until the EmitMojo path is the default path
  // because the soon-to-be-defunct relies on the current behavior.
  SmallVector<MojoValue> dims;
  dims.reserve(rank);
  llvm::transform(tensorSpec.getDims(), std::back_inserter(dims),
                  [](const std::optional<int64_t> &d) {
                    return d.value_or(mlir::ShapedType::kDynamic);
                  });
  MojoValue rawDimsParam =
      MojoValue(HardcodedMojoSymbols::struct_DimList)({}, dims);

  MojoValue result = state.builder.var(function(
      {rawDimsParam, rawDimsRankParam}, state.getMojoValuesFor(getOperands())));
  state.setMojoValueFor(getResult(), result);

  return success();
}
void TensorSpecCreateOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                               TensorSpecType spec, ValueRange runtimeDims) {
  ASSERT_STREAM(spec.hasRank(),
                << "can only construct an mgp.tensor_spec.create "
                   "for tensor specs of statically known rank");
  TensorSpecCreateOp::build(odsBuilder, odsState,
                            /*runtimeDims=*/runtimeDims,
                            /*aRawDims=*/spec.getRawDims(),
                            /*bRawDType=*/spec.getRawDType());
}

/// Parses a !mgp.tensor_spec type where the element type must be known,
/// but represented in 'raw form'.
static ParseResult parseRawDimsAndDType(AsmParser &p,
                                        IntArrayElementsAttr &rawDims,
                                        IntegerAttr &rawDType) {
  Type type;
  llvm::SMLoc loc;
  if (p.getCurrentLocation(&loc) || p.parseType(type))
    return failure();
  auto tensorSpec = dyn_cast<TensorSpecType>(type);
  if (tensorSpec == nullptr)
    return p.emitError(loc) << "expected a !mgp.tensor_spec type.";
  if (!tensorSpec.hasRank())
    return p.emitError(loc) << "expected !mgp.tensor_spec type to have a "
                               "statically known rank.";
  if (!tensorSpec.getOptDType())
    return p.emitError(loc) << "expected !mgp.tensor_spec type to have a "
                               "known element type.";

  rawDims = tensorSpec.getRawDims();
  rawDType = tensorSpec.getRawDType();

  return success();
}

/// Prints a !mgp.tensor_spec type, but represented in 'raw form'.
static void printRawDimsAndDType(AsmPrinter &p, const TensorSpecCreateOp &op,
                                 IntArrayElementsAttr rawDims,
                                 IntegerAttr rawDType) {
  p << TensorSpecType::get(rawDims, rawDType);
}

LogicalResult TensorSpecCreateOp::verify() {
  auto impliedSpecType =
      TensorSpecType::getChecked([this]() { return emitOpError(); },
                                 getContext(), getARawDims(), getBRawDType());
  if (impliedSpecType == nullptr)
    return failure();
  assert(impliedSpecType.hasRank());
  TensorSpecType actualSpecType = getResult().getType();
  if (actualSpecType != impliedSpecType)
    return emitOpError()
           << "expected result type is " << actualSpecType
           << ", but static dimensions and element type arguments "
              "suggest it should be "
           << impliedSpecType << ".";
  if (getRuntimeDims().size() != impliedSpecType.getNumDynamicDims())
    return emitOpError() << "expected " << actualSpecType.getNumDynamicDims()
                         << " dimension operands to match " << actualSpecType
                         << ", but given " << getRuntimeDims().size() << ".";
  return success();
}

LogicalResult TensorSpecCreateOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, Adaptor adaptor,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  auto rawDimsAttr = adaptor.getARawDimsAttr();
  auto rawDTypeAttr = adaptor.getBRawDTypeAttr();
  ASSERT_STREAM(rawDimsAttr && rawDTypeAttr, << "mismatched operand type");
  inferredReturnTypes.emplace_back(
      TensorSpecType::get(rawDimsAttr, rawDTypeAttr));
  return success();
}

//===----------------------------------------------------------------------===//
// BufferRemoveCachedOp
//===----------------------------------------------------------------------===//

llvm::BitVector BufferRemoveCachedOp::operandsToUnpack() {
  llvm::BitVector res(getNumOperands(), true);
  res[1] = false; // Do not unpack the context because the type is defined in
                  // C++ only.
  return res;
}

LogicalResult BufferRemoveCachedOp::emitMojo(MOGG::EmitMojoState &state) {
  /// TODO: Add error checking. The Mojo/C++ can return an error that we
  auto mgpBufferRemoveCachedFunc =
      state.getOriginalMojoNameFor(state.analysis.getRegisteredLITMojoKernelFor(
          "tmp.mgp.buffer.remove_cached"));
  state.builder.call(mgpBufferRemoveCachedFunc, {},
                     {state.getMojoValueFor(getTheContext()),
                      state.getMojoValueFor(getBBufferSlot())});

  return success();
}

//===----------------------------------------------------------------------===//
// BufferGetSizeOp
//===----------------------------------------------------------------------===//

OpFoldResult BufferGetSizeOp::fold(FoldAdaptor adaptor) {
  auto bufferAlloc = getBuffer().getDefiningOp<BufferAllocOp>();
  if (bufferAlloc)
    return bufferAlloc.getRuntimeSize();

  auto bufferConstant = getBuffer().getDefiningOp<BufferConstantOp>();
  if (bufferConstant)
    return IntegerAttr::get(IndexType::get(getContext()),
                            bufferConstant.getAValue().size());

  return {};
}

LogicalResult BufferGetSizeOp::emitMojo(MOGG::EmitMojoState &state) {
  using namespace MOGG;

  KGEN::LIT::FnOp generator =
      state.analysis.getRegisteredLITMojoKernelFor(getOperationName());
  MojoValue getSizeFunc = state.getOriginalMojoNameFor(generator);

  MojoValue bufferVar = state.getMojoValueFor(getBuffer());
  state.setMojoValueFor(getResult(), getSizeFunc({}, {bufferVar}));

  return success();
}

//===----------------------------------------------------------------------===//
// TensorSpecGetDimOp
//===----------------------------------------------------------------------===//

LogicalResult TensorSpecGetDimOp::emitMojo(MOGG::EmitMojoState &state) {
  KGEN::LIT::FnOp generator =
      state.analysis.getRegisteredLITMojoKernelFor(getOperationName());
  MOGG::MojoValue getDimFunc = state.getOriginalMojoNameFor(generator);

  MOGG::MojoValue axisParam = MOGG::MojoValue(getAxis());
  MOGG::MojoValue rankParam = getSpec().getType().getRank();
  MOGG::MojoValue specVar = state.getMojoValueFor(getSpec());

  state.setMojoValueFor(getResult(), state.builder.var(getDimFunc(
                                         {rankParam, axisParam}, {specVar})));

  return success();
}

LogicalResult TensorSpecGetDimOp::verify() {
  TensorSpecType specType = getSpec().getType();
  if (!specType.hasRank())
    return emitOpError() << "expected tensor spec to have known rank";
  if (specType.hasRank() && getAxis() >= specType.getRank())
    return emitOpError() << "expected axis " << getAxis()
                         << " to be less than the rank of the tensor spec "
                         << specType.getRank();
  return success();
}

OpFoldResult TensorSpecGetDimOp::fold(FoldAdaptor adaptor) {
  Operation *specProducer = getSpec().getDefiningOp();
  TensorSpecType specType = getSpec().getType();
  if (!specType.hasRank())
    return {};
  if (getAxis() >= specType.getRank())
    return {};
  if (!specType.isDynamicDim(getAxis())) {
    auto indexTy = IndexType::get(getContext());
    uint64_t dim = specType.getDims()[getAxis()].value();
    return IntegerAttr::get(indexTy, dim);
  }
  if (auto specCreate = dyn_cast_or_null<TensorSpecCreateOp>(specProducer)) {
    return specCreate.getRuntimeDims()[specType.getDynamicDimIndex(getAxis())];
  }

  return {};
}

//===----------------------------------------------------------------------===//
// BufferGetCachedOp
//===----------------------------------------------------------------------===//

llvm::BitVector BufferGetCachedOp::operandsToUnpack() {
  llvm::BitVector res(getNumOperands(), true);
  res[1] = false; // Do not unpack the context because the type is defined in
                  // C++ only.
  return res;
}

llvm::BitVector BufferGetCachedOp::resultsToPack() {
  return llvm::BitVector(2, false);
}

LogicalResult BufferGetCachedOp::emitMojo(MOGG::EmitMojoState &state) {
  using namespace MOGG;
  auto mgpBufferGetCacheFunc =
      state.getOriginalMojoNameFor(state.analysis.getRegisteredLITMojoKernelFor(
          "tmp.mgp.buffer.get_cached"));
  auto res = state.builder.var(
      mgpBufferGetCacheFunc({}, {state.getMojoValueFor(getTheContext()),
                                 state.getMojoValueFor(getBBufferSlot())}));
  state.setMojoValueFor(getResult(0), res[0]);

  if (!getOptionalMemStorageHandle().empty())
    state.setMojoValueFor(getMemStorageHandle(), res[1]);

  return success();
}

LogicalResult BufferGetCachedOp::verify() {
  if (getOptionalMemStorageHandle().size() > 1)
    return emitOpError("Unexpected mem storage handle result");
  return success();
}

LogicalResult BufferGetCachedOp::canonicalize(BufferGetCachedOp op,
                                              PatternRewriter &rewriter) {
  if (op.getOptionalMemStorageHandle().size() == 1 &&
      op.getMemStorageHandle().use_empty()) {
    auto cachedBufferOp = BufferGetCachedOp::create(
        rewriter, op->getLoc(), op.getOutBuffer().getType(), op->getOperands(),
        op->getAttrs());
    rewriter.replaceAllUsesWith(op.getOutBuffer(),
                                cachedBufferOp.getOutBuffer());
    rewriter.eraseOp(op);
  }
  return success();
}

//===----------------------------------------------------------------------===//
// AssertOp
//===----------------------------------------------------------------------===//

bool AssertOp::isThrows() { return true; }

LogicalResult AssertOp::emitMojo(MOGG::EmitMojoState &state) {
  KGEN::LIT::FnOp func =
      state.analysis.getRegisteredLITMojoKernelFor(getOperationName());
  MOGG::MojoValue function = state.getOriginalMojoNameFor(func);

  MOGG::MojoValue conditionVar = state.getMojoValueFor(getCond());
  MOGG::MojoValue messageLiteral = MOGG::MojoValue(getMessage()).quote();
  MOGG::MojoValue messageString = state.builder.var(MOGG::MojoValue(
      MOGG::HardcodedMojoSymbols::struct_String)({}, {messageLiteral}));
  MOGG::MojoValue messagePtr = messageString.get("unsafe_ptr")();
  MOGG::MojoValue messageSize = messageString.get("__len__")();

  state.builder.call(function, {}, {conditionVar, messagePtr, messageSize});

  return success();
}

SmallVector<Attribute> AssertOp::getLiftableAttributes() {
  return {getMessageAttr()};
}

LogicalResult AssertOp::canonicalize(AssertOp op, PatternRewriter &rewriter) {
  bool isInKernel = false;
  if (op->getParentOfType<MOGG::GenericCompiledRegionOp>())
    isInKernel = true;

  if (op->getParentOfType<MOGG::KernelStubOp>())
    isInKernel = true;

  // remove assert(true)
  BoolAttr condAttr;
  if (mlir::matchPattern(op.getCond(), mlir::m_Constant(&condAttr))) {
    if (condAttr.getValue()) {
      if (isInKernel) {
        rewriter.replaceOpWithNewOp<MOGG::ChainDummyOp>(op);
      } else {
        rewriter.replaceOpWithNewOp<GRT::ChainCreateOp>(op);
      }
      return success();
    }
  }
  return failure();
}

//===----------------------------------------------------------------------===//
// TensorExtractBufferOp
//===----------------------------------------------------------------------===//
mlir::BitVector TensorExtractBufferOp::resultsToPack() {
  return mlir::BitVector{1, false};
}

LogicalResult TensorExtractBufferOp::emitMojo(MOGG::EmitMojoState &state) {
  using namespace MOGG;

  KGEN::LIT::FnOp litFunction =
      state.analysis.getRegisteredLITMojoKernelFor(getOperationName());
  MojoValue function = state.getOriginalMojoNameFor(litFunction);

  MojoValue res =
      state.builder.var(function({}, state.getMojoValueFor(getTensor())));
  state.setMojoValueFor(getOutBuffer(), res);

  return success();
}

LogicalResult TensorExtractBufferOp::verify() {
  DeviceBufferType expectedBufferType =
      DeviceBufferType::get(getTensor().getType());
  if (expectedBufferType != getOutBuffer().getType()) {
    return emitOpError("expected the result buffer to match the type "
                       "implied by the tensor type (")
           << getOutBuffer().getType() << " vs " << expectedBufferType << ").";
  }
  return success();
}

LogicalResult TensorExtractBufferOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, Adaptor adaptor,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  auto tensorType = dyn_cast<DeviceTensorType>(adaptor.getTensor().getType());
  if (!tensorType) {
    return mlir::emitError(location.value_or(adaptor.getTensor().getLoc()),
                           "expected tensor operand.");
  }
  inferredReturnTypes.emplace_back(DeviceBufferType::get(tensorType));
  return success();
}

bool TensorExtractBufferOp::mayAlias(size_t operandIndex, size_t resultIndex) {
  return operandIndex == getTensorMutable().getOperandNumber();
}

OpFoldResult TensorExtractBufferOp::fold(FoldAdaptor adaptor) {
  Operation *tensorProducer = getTensor().getDefiningOp();
  if (auto create = dyn_cast_or_null<TensorCreateOp>(tensorProducer))
    return create.getBuffer();

  return {};
}

//===----------------------------------------------------------------------===//
// TensorExtractTensorSpecOp
//===----------------------------------------------------------------------===//
LogicalResult TensorExtractTensorSpecOp::emitMojo(MOGG::EmitMojoState &state) {
  using namespace MOGG;

  KGEN::LIT::FnOp litFunction =
      state.analysis.getRegisteredLITMojoKernelFor(getOperationName());
  MojoValue function = state.getOriginalMojoNameFor(litFunction);

  DeviceTensorType tensorType = getTensor().getType();
  MojoValue tensorRankParam = tensorType.getRank();
  MojoValue bufferRankParam =
      tensorType.getRank() == 0 ? MojoValue{1} : tensorRankParam;
  auto tensorSpec = tensorType.getTensorSpec();
  MojoValue tensorDtypeParam =
      state.mlirDTypeToMojoType(tensorSpec.getOptDType().value());

  MojoValue result = state.builder.var(
      function({tensorRankParam, bufferRankParam, tensorDtypeParam},
               state.getMojoValueFor(getTensor())));
  state.setMojoValueFor(getResult(), result);
  return success();
}

LogicalResult TensorExtractTensorSpecOp::verify() {
  TensorSpecType expectedSpecType = getTensor().getType().getTensorSpec();
  if (expectedSpecType != getOutSpec().getType()) {
    return emitOpError("expected the result tensor spec to match the type "
                       "implied by the tensor type (")
           << getOutSpec().getType() << " vs " << expectedSpecType << ").";
  }
  if (!expectedSpecType.hasRank())
    return emitError() << "Expected to have rank information";
  if (!expectedSpecType.getOptDType().has_value())
    return emitError() << "Expected to have DType information for operand";
  return success();
}

LogicalResult TensorExtractTensorSpecOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, Adaptor adaptor,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  auto tensorType = dyn_cast<DeviceTensorType>(adaptor.getTensor().getType());
  if (!tensorType) {
    return mlir::emitError(location.value_or(adaptor.getTensor().getLoc()),
                           "expected tensor operand.");
  }
  inferredReturnTypes.emplace_back(tensorType.getTensorSpec());
  return success();
}

OpFoldResult TensorExtractTensorSpecOp::fold(FoldAdaptor adaptor) {
  Operation *tensorProducer = getTensor().getDefiningOp();
  if (auto create = dyn_cast_or_null<TensorCreateOp>(tensorProducer))
    return create.getTensorSpec();
  return {};
}

//===----------------------------------------------------------------------===//
// GuardOp
//===----------------------------------------------------------------------===//

LogicalResult GuardOp::verify() {
  if (getInputs().size() != getResults().size()) {
    return emitError() << "Expected the number of inputs and outputs to be "
                          "exactly the same";
  }
  return success();
}

LogicalResult
GuardOp::inferReturnTypes(MLIRContext *context,
                          std::optional<Location> location, Adaptor adaptor,
                          SmallVectorImpl<Type> &inferredReturnTypes) {
  TypeRange inputTypes = adaptor.getInputs().getTypes();
  inferredReturnTypes.append(inputTypes.begin(), inputTypes.end());
  return success();
}

bool GuardOp::mayAlias(size_t operandIndex, size_t resultIndex) { return true; }

//===----------------------------------------------------------------------===//
// TensorCreateOp
//===----------------------------------------------------------------------===//

mlir::BitVector TensorCreateOp::resultsToPack() {
  return mlir::BitVector{1, false};
}

LogicalResult TensorCreateOp::emitMojo(MOGG::EmitMojoState &state) {
  using namespace MOGG;

  KGEN::LIT::FnOp litFunction =
      state.analysis.getRegisteredLITMojoKernelFor(getOperationName());
  MojoValue function = state.getOriginalMojoNameFor(litFunction);

  DeviceTensorType tensorType = getOutTensor().getType();
  MojoValue tensorRankParam = tensorType.getRank();
  MojoValue bufferRankParam =
      tensorType.getRank() == 0 ? MojoValue{1} : tensorRankParam;
  auto tensorSpec = tensorType.getTensorSpec();
  MojoValue tensorDtypeParam =
      state.mlirDTypeToMojoType(tensorSpec.getOptDType().value());

  MojoValue result = state.builder.var(
      function({tensorRankParam, bufferRankParam, tensorDtypeParam},
               state.getMojoValuesFor(getOperands().drop_front())));
  state.setMojoValueFor(getOutTensor(), result);

  return success();
}
bool TensorCreateOp::extendsLifetime(size_t operandIndex, size_t resultIndex) {
  return operandIndex == getBufferMutable().getOperandNumber();
}

//===----------------------------------------------------------------------===//
// TensorExtractBufferOp
//===----------------------------------------------------------------------===//

bool TensorExtractBufferOp::extendsLifetime(size_t operandIndex,
                                            size_t resultIndex) {
  return operandIndex == getTensorMutable().getOperandNumber();
}

//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

LogicalResult
CallOp::verifySymbolUses(mlir::SymbolTableCollection &symbolTable) {
  auto targetModel =
      symbolTable.lookupNearestSymbolFrom<ModelOp>(*this, getCalleeAttr());
  if (!targetModel)
    return emitOpError() << "failed to find symbol " << getCalleeAttr();

  // Verify operand types match the execute block argument types.
  Block *executeBlock = targetModel.getExecuteBlock();
  auto expectedArgs = executeBlock->getArguments();
  auto actualArgTypes = getOperandTypes();
  if (expectedArgs.size() != actualArgTypes.size()) {
    return emitOpError() << "operand types " << actualArgTypes
                         << " do not match expected argument types "
                         << executeBlock->getArgumentTypes() << " of "
                         << getCalleeAttr() << "'s execute block.";
  }

  for (auto [expectedArg, actualArgType] :
       llvm::zip(expectedArgs, actualArgTypes)) {
    if (expectedArg.getType() != actualArgType) {
      return emitOpError() << "operand types " << actualArgTypes
                           << " do not match expected argument types "
                           << executeBlock->getArgumentTypes() << " of "
                           << getCalleeAttr() << "'s execute block.";
    }
  }

  // Verify result types match the execute block result types.
  auto outputOp = dyn_cast<MEF::OutputOp>(executeBlock->getTerminator());
  if (!outputOp) {
    return emitOpError() << "execute block of " << getCalleeAttr()
                         << " does not end with an mef.output op.";
  }

  auto expectedResultTypes = outputOp.getOperandTypes();
  auto actualResultTypes = getResultTypes();
  if (actualResultTypes != expectedResultTypes) {
    return emitOpError() << "result types " << actualResultTypes
                         << " do not match expected result types "
                         << expectedResultTypes << " of " << getCalleeAttr()
                         << "'s execute block.";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// DebugPrintOp
//===----------------------------------------------------------------------===//

bool DebugPrintOp::isThrows() { return true; }

LogicalResult DebugPrintOp::emitMojo(MOGG::EmitMojoState &state) {
  KGEN::LIT::FnOp litFunction =
      state.analysis.getRegisteredLITMojoKernelFor(getOperationName());
  MOGG::MojoValue function = state.getOriginalMojoNameFor(litFunction);

  MOGG::MojoValue debugStringParam = MOGG::MojoValue(getADebugString()).quote();
  MOGG::MojoValue labelParam = MOGG::MojoValue(getBLabel()).quote();
  MOGG::MojoValue ctxVar = state.getMojoValueFor(getTheContext());

  state.builder.call(function, {debugStringParam, labelParam}, ctxVar);

  return success();
}

//===----------------------------------------------------------------------===//
// ModelOp
//===----------------------------------------------------------------------===//

// Builds an !mgp.opaque type with the given symbol name.
static OpaqueType opaqueType(MLIRContext *ctx, StringRef symbol) {
  return OpaqueType::get(ctx, StringAttr::get(ctx, symbol),
                         /*metadata=*/{});
}

// Extends args with the weights registry and device context implicit args.
static void appendInitExecImplicitArgs(SmallVectorImpl<Type> &args,
                                       MLIRContext *ctx,
                                       ArrayRef<DeviceRef> devices) {
  // Add the weights registry to the implicit signature.
  args.push_back(opaqueType(ctx, "weights_registry"));

  // Add all the device contexts to the implicit signature.
  for (const DeviceRef &device : devices) {
    args.push_back(DeviceContextType::get(
        ctx, DeviceRefAttr::get(ctx, device.label, device.id)));
  }
}

/// Returns the implicit signature of the 'init' entry point.
static FunctionType implicitInitSignature(Builder &builder,
                                          ArrayRef<DeviceRef> devices) {
  SmallVector<Type> inputs;
  appendInitExecImplicitArgs(inputs, builder.getContext(), devices);

  SmallVector<Type> outputs = {builder.getType<MEF::ChainType>(),
                               builder.getType<MGP::ContextType>()};
  appendInitExecImplicitArgs(outputs, builder.getContext(), devices);

  return builder.getFunctionType(inputs, outputs);
}

/// Returns the implicit signature of the 'execute' entry point, before
/// any explicit inputs and results have been included.
static FunctionType implicitExecuteSignature(Builder &builder,
                                             ArrayRef<DeviceRef> devices) {
  SmallVector<Type> inputs = {builder.getType<MEF::ChainType>(),
                              builder.getType<MGP::ContextType>()};

  appendInitExecImplicitArgs(inputs, builder.getContext(), devices);

  return builder.getFunctionType(
      inputs, /*results=*/{builder.getType<MEF::ChainType>()});
}

/// Returns true if requiredTypes is a prefix of actualTypes, in which case
/// the prefix is removed from actualTypes.
static bool checkAndRemovePrefix(ArrayRef<Type> requiredTypes,
                                 SmallVectorImpl<Type> &actualTypes) {
  auto [mismatchRequired, mismatchActual] = std::mismatch(
      requiredTypes.begin(), requiredTypes.end(), actualTypes.begin());
  if (mismatchRequired != requiredTypes.end())
    return false;
  actualTypes.erase(actualTypes.begin(),
                    actualTypes.begin() + requiredTypes.size());
  return true;
}

/// Parse the frameworkName attribute as a keyword.
static ParseResult parseClassName(OpAsmParser &p, StringAttr &frameworkName) {
  StringRef keyword;
  if (p.parseKeyword(&keyword))
    return failure();
  frameworkName = StringAttr::get(p.getContext(), keyword);
  return success();
}

/// Prints the frameworkName attribute as a keyword.
static void printClassName(OpAsmPrinter &p, const ModelOp &op,
                           StringAttr frameworkName) {
  p << frameworkName.getValue();
}

/// Parse the init region.
static ParseResult parseInitRegion(OpAsmParser &p, Region &initRegion) {
  llvm::SMLoc argLoc;
  SmallVector<OpAsmParser::Argument> args;
  SmallVector<Type> outputTypes;
  if (p.getCurrentLocation(&argLoc) ||
      p.parseArgumentList(args, OpAsmParser::Delimiter::Paren,
                          /*allowType=*/true) ||
      p.parseRegion(initRegion, args, /*enableNameShadowing=*/true))
    return failure();

  return success();
}

/// Print the init region.
static void printInitRegion(OpAsmPrinter &p, const ModelOp &op,
                            Region &initRegion) {
  Block &initBlock = initRegion.front();
  p << "(";
  llvm::interleaveComma(initBlock.getArguments(), p.getStream(),
                        [&](auto arg) { p.printRegionArgument(arg); });
  p << ") ";
  p.printRegion(initRegion, /*printEntryBlockArgs=*/false);
}

/// Parses the 'execute' body, and extracts from it the implicit signature.
static ParseResult parseExecuteRegion(OpAsmParser &p,
                                      TypeAttr &inputOutputTypes,
                                      Region &executeRegion,
                                      ArrayAttr &deviceAttrs) {
  llvm::SMLoc argLoc;
  SmallVector<OpAsmParser::Argument> args;
  SmallVector<Type> outputTypes;
  if (p.getCurrentLocation(&argLoc) ||
      p.parseArgumentList(args, OpAsmParser::Delimiter::Paren,
                          /*allowType=*/true) ||
      p.parseArrowTypeList(outputTypes) ||
      p.parseRegion(executeRegion, args, /*enableNameShadowing=*/true))
    return failure();

  // Check the execute region's actual signature, and save its implicit
  // signature into inputOutputTypes.
  SmallVector<DeviceRef> devices;
  for (auto attr : deviceAttrs) {
    if (auto deviceAttr = dyn_cast<DeviceRefAttr>(attr))
      devices.emplace_back(
          DeviceRef(deviceAttr.getLabel().str(), deviceAttr.getId()));
  }
  FunctionType implicitExecSig =
      implicitExecuteSignature(p.getBuilder(), devices);
  SmallVector<Type> inputTypes(executeRegion.getArgumentTypes());
  if (!checkAndRemovePrefix(implicitExecSig.getInputs(), inputTypes))
    return p.emitError(argLoc)
           << "expected the 'execute' region arguments to begin with types "
           << implicitExecSig.getInputs() << ".";
  if (!checkAndRemovePrefix(implicitExecSig.getResults(), outputTypes))
    return p.emitError(argLoc)
           << "expected the 'execute' region outputs to begin with types "
           << implicitExecSig.getResults() << ".";

  inputOutputTypes =
      TypeAttr::get(p.getBuilder().getFunctionType(inputTypes, outputTypes));
  return success();
}

/// Prints the 'execute' body. The inputOutputTypes are ignored.
static void printExecuteRegion(OpAsmPrinter &p, const ModelOp &op,
                               TypeAttr inputOutputTypes, Region &executeRegion,
                               ArrayAttr devices) {
  Block &executeBlock = executeRegion.front();
  p << "(";
  llvm::interleaveComma(executeBlock.getArguments(), p.getStream(),
                        [&](auto arg) { p.printRegionArgument(arg); });
  p << ")";
  p.printArrowTypeList(executeBlock.getTerminator()->getOperandTypes());
  p << " ";
  p.printRegion(executeRegion, /*printEntryBlockArgs=*/false);
}

void ModelOp::build(OpBuilder &builder, OperationState &state, StringAttr name,
                    StringRef frameworkName, ArrayRef<DeviceRef> devices,
                    ArrayRef<Type> inputTypes, ArrayRef<Type> outputTypes,
                    ArrayRef<Location> inputArgLocs, bool isSubgraph,
                    ArrayRef<NamedAttribute> extraAttrs) {
  MLIRContext *ctx = state.getContext();

  SmallVector<Type> finalInputTypes(inputTypes.begin(), inputTypes.end());
  SmallVector<Location> finalInputLocs(inputArgLocs.begin(),
                                       inputArgLocs.end());
  if (isSubgraph)
    state.addAttribute(getIsSubgraphAttrName(state.name), UnitAttr::get(ctx));

  DictionaryAttr extraAttrsDict =
      extraAttrs.empty() ? nullptr : DictionaryAttr::get(ctx, extraAttrs);
  auto deviceAttrs = llvm::map_to_vector(devices, [ctx](const DeviceRef &ref) {
    return dyn_cast<Attribute>(DeviceRefAttr::get(ctx, ref.label, ref.id));
  });
  build(builder, state, ArrayAttr::get(ctx, deviceAttrs),
        StringAttr::get(ctx, frameworkName),
        TypeAttr::get(builder.getFunctionType(finalInputTypes, outputTypes)),
        name, isSubgraph ? builder.getUnitAttr() : nullptr, extraAttrsDict);

  Region *initRegion = state.regions[0].get();
  initRegion->emplaceBlock();

  // Add the implicit weights registry and device context args to init.
  for (auto type : implicitInitSignature(builder, devices).getInputs())
    initRegion->addArgument(type, state.location);

  // The 'execute' entry point has both implicit and explicit arguments.
  Region *executeRegion = state.regions[1].get();
  executeRegion->emplaceBlock();
  for (auto type : implicitExecuteSignature(builder, devices).getInputs())
    executeRegion->addArgument(type, state.location);
  for (auto [type, loc] : llvm::zip(finalInputTypes, finalInputLocs))
    executeRegion->addArgument(type, loc);
}

/// Verifies the init and execute regions of the model.
LogicalResult ModelOp::verifyRegions() {
  using BlockArgTypeRange = mlir::ValueTypeRange<mlir::Block::BlockArgListType>;

  Builder builder(getContext());

  SmallVector<DeviceRef> devices;
  for (auto attr : getDevices()) {
    if (auto deviceAttr = dyn_cast<DeviceRefAttr>(attr))
      devices.emplace_back(
          DeviceRef(deviceAttr.getLabel().str(), deviceAttr.getId()));
  }

  // The type of the init region is
  //  () -> (!mef.chain, !mgp.context)
  Block *initBlock = getInitBlock();
  BlockArgTypeRange initArgTypes = initBlock->getArgumentTypes();
  if (auto output = dyn_cast<MEF::OutputOp>(initBlock->getTerminator())) {
    FunctionType implicitInitSig = implicitInitSignature(builder, devices);

    // Check that the number of init block arguments matches the number of
    // devices + 1 for the weights registry.
    size_t numDeviceArgs = initArgTypes.size() - 1;
    if (numDeviceArgs != getDevices().size()) {
      return emitOpError() << "Expected the number of arguments to the 'init' "
                              "body to match the number of devices, but got "
                           << numDeviceArgs << " vs " << getDevices().size()
                           << ".";
    }
    for (auto t : initArgTypes) {
      if (!mlir::isa<OpaqueType, DeviceContextType>(t)) {
        return emitOpError() << "init block arguments must be !mgp.opaque or "
                                "!mgp.device_context, but got "
                             << t;
      }
    }

    // Verify that there is exactly one ContextCreateOp in the init block.
    int contextCreateOpCount = 0;
    initBlock->walk([&](ContextCreateOp) { contextCreateOpCount++; });
    if (contextCreateOpCount != 1) {
      return emitOpError()
             << "expected exactly one mgp.context_create op in the "
                "'init' block, but found "
             << contextCreateOpCount;
    }

    if (failed(checkOperandTypes(output, implicitInitSig.getResults())))
      return failure();
  }

  // The type of the execute region is:
  //   (!mef.chain, !mgp.context, weights_registry, dev_1, ..., dev_n, arg_1,
  //   ... arg_n) ->
  //   (!mef.chain, res_1, ... res_m)
  // where the ModelOp's inputOutputTypes are:
  //   (arg_1, ... arg_n) -> (res_1, ... res_m)
  Block *execBlock = getExecuteBlock();
  if (auto output = dyn_cast<MEF::OutputOp>(execBlock->getTerminator())) {
    FunctionType implicitExecSig = implicitExecuteSignature(builder, devices);
    ArrayRef<Type> modelArgTypes = getInputOutputTypes().getInputs();
    SmallVector<Type> expectedArgTypes(implicitExecSig.getInputs());
    expectedArgTypes.append(modelArgTypes.begin(), modelArgTypes.end());
    if (execBlock->getArgumentTypes() != expectedArgTypes) {
      return emitOpError() << "expected the arguments of the "
                              "'execute' body to be "
                           << expectedArgTypes << ", but got "
                           << execBlock->getArgumentTypes() << ".";
    }

    ArrayRef<Type> modelResultTypes = getInputOutputTypes().getResults();
    SmallVector<Type> expectedResultTypes(implicitExecSig.getResults());
    expectedResultTypes.append(modelResultTypes.begin(),
                               modelResultTypes.end());

    if (failed(checkOperandTypes(output, expectedResultTypes)))
      return failure();
  }

  return success();
}

/// Verifies the input/output type signature.
LogicalResult ModelOp::verify() { return success(); }

ContextCreateOp ModelOp::getInitContext() {
  auto range = getInitBlock()->getOps<ContextCreateOp>();
  ASSERT_STREAM(llvm::hasSingleElement(range),
                "should contain one context create op");
  return *std::begin(range);
}

TypedValue<ContextType> ModelOp::getExecuteContext() {
  auto argument = getExecuteBlock()->getArgument(1);
  return cast<TypedValue<ContextType>>(argument);
}

//===----------------------------------------------------------------------===//
// MGPDialect op support
//===----------------------------------------------------------------------===//

void MGPDialect::registerOperations() {
  addOperations<
#define GET_OP_LIST
#include "GenericML/GraphCompiler/MGPDialect/Dialect/MGP.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

using PointerType = KGEN::PointerType;
#define GET_OP_CLASSES
#include "GenericML/GraphCompiler/MGPDialect/Dialect/MGP.cpp.inc"
