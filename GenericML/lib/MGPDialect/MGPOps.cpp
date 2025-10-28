//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements the MGP dialect operations.
//
//===----------------------------------------------------------------------===//

#include "GenericML/MGPDialect/MGPOps.h"
#include "GenericML/GraphCompiler/Interfaces/EmitMojoState.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GenericML/GraphCompiler/MOGGDialect/Support/MojoBuilder.h"
#include "GenericML/GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "GenericML/MGPDialect/BufferBuilder.h"
#include "GraphRT/GRTDialect/GRTOps.h"
#include "KGEN/LITDialect/LITOps.h"
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
// CpuBufferConstantOp
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

//===----------------------------------------------------------------------===//
// TensorSpecCreateOp
//===----------------------------------------------------------------------===//

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

SmallVector<Attribute> BufferRemoveCachedOp::getLiftableAttributes() {
  return {getBufferSlotAttr()};
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

  state.setMojoValueFor(getResult(),
                        getDimFunc({rankParam, axisParam}, {specVar}));

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
  return llvm::BitVector(1, false);
}

LogicalResult BufferGetCachedOp::emitMojo(MOGG::EmitMojoState &state) {
  auto mgpBufferGetCacheFunc =
      state.getOriginalMojoNameFor(state.analysis.getRegisteredLITMojoKernelFor(
          "tmp.mgp.buffer.get_cached"));
  auto cachedBuffer = state.builder.var(
      mgpBufferGetCacheFunc({}, {state.getMojoValueFor(getTheContext()),
                                 state.getMojoValueFor(getBBufferSlot())}));
  state.setMojoValueFor(getResult(), cachedBuffer);

  return success();
}

//===----------------------------------------------------------------------===//
// AssertOp
//===----------------------------------------------------------------------===//

bool AssertOp::isRaising() { return true; }

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
  auto moggAsyncReadFunc =
      state.getOriginalMojoNameFor(state.analysis.getRegisteredLITMojoKernelFor(
          MOGG::AsyncReadyOp::getOperationName()));
  state.builder.call(moggAsyncReadFunc, {}, state.mojoReturnValues[0]);

  return success();
}

SmallVector<Attribute> AssertOp::getLiftableAttributes() {
  return {getMessageAttr()};
}

LogicalResult AssertOp::canonicalize(AssertOp op, PatternRewriter &rewriter) {
  // remove assert(true)
  BoolAttr condAttr;
  if (mlir::matchPattern(op.getCond(), mlir::m_Constant(&condAttr))) {
    if (condAttr.getValue()) {
      rewriter.replaceOpWithNewOp<GRT::ChainCreateOp>(op);
      return success();
    }
  }
  return failure();
}

/// Return whether `ty1` is a subtype of `ty2`, i.e., the set of possible
/// tensors represented by `ty1` is a subset for that of `ty2`.
static bool isSubtype(TensorSpecType ty1, TensorSpecType ty2) {
  // !mgp.tensor_spec<...xf32>  v.s. !mgp.tensor_spec<...xsi32>
  if (ty1.getElementType() != ty2.getElementType())
    return false;

  // Any shape is a subtype of a dynamically ranked shape.
  if (!ty2.hasRank())
    return true;

  // If ty2 is ranked, ty1 must be ranked as well
  if (!ty1.hasRank())
    return false;

  // 1. Make sure # of dims match.
  if (ty1.getRank() != ty2.getRank())
    return false;

  // 2. Make sure ty1's individual dims are subtypes to the corresponding
  // dimensions in ty2.
  ArrayRef<std::optional<uint64_t>> dims1 = ty1.getDims();
  ArrayRef<std::optional<uint64_t>> dims2 = ty2.getDims();
  for (auto [dim1, dim2] : llvm::zip(dims1, dims2)) {
    // If `dim2` is unknown `?`, `dim1` can be anything in order for `ty1` to be
    // a subtype of `ty2`; otherwise the dimensions must match.
    if (dim2.has_value() && dim1 != dim2)
      return false;
  }

  return true;
}

/// Return whether `ty1` is a subtype of `ty2`, i.e., the set of possible
/// tensors represented by `ty1` is a subset for that of `ty2`. Currently only
/// support nontrivial subtyping for `CpuTensorType`.
static bool isSubtype(Type ty1, Type ty2) {
  auto tensorTy1 = dyn_cast<MGPTensorTypeInterface>(ty1);
  auto tensorTy2 = dyn_cast<MGPTensorTypeInterface>(ty2);
  if (tensorTy1 && tensorTy2)
    return isSubtype(tensorTy1.getTensorSpec(), tensorTy2.getTensorSpec());
  return ty1 == ty2;
}

//===----------------------------------------------------------------------===//
// TensorExtractBufferOp
//===----------------------------------------------------------------------===//

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

LogicalResult TensorExtractTensorSpecOp::verify() {
  TensorSpecType expectedSpecType = getTensor().getType().getTensorSpec();
  if (expectedSpecType != getOutSpec().getType()) {
    return emitOpError("expected the result tensor spec to match the type "
                       "implied by the tensor type (")
           << getOutSpec().getType() << " vs " << expectedSpecType << ").";
  }
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
// ListAppendOp
//===----------------------------------------------------------------------===//

LogicalResult ListAppendOp::verify() {
  // Check input element type against output element type.
  Type inputElemTy = getInput().getType().getElementType();
  Type outputElemTy = getType().getElementType();
  if (!isSubtype(inputElemTy, outputElemTy)) {
    return emitError()
           << "input element type must be a subtype of output element type: "
           << inputElemTy << " v.s. " << outputElemTy;
  }

  // Check new element type against list element type.
  Type newElemTy = getElement().getType();
  if (!isSubtype(newElemTy, outputElemTy)) {
    return emitError()
           << "new element type must be a subtype of output element type: "
           << newElemTy << " v.s. " << outputElemTy;
  }
  return success();
}

bool ListAppendOp::mayAlias(size_t operandIndex, size_t resultIndex) {
  // The resulting list contains shallow copies of input list elements and the
  // appended element.
  return true;
}

//===----------------------------------------------------------------------===//
// ListConcatOp
//===----------------------------------------------------------------------===//

LogicalResult ListConcatOp::verify() {
  Type lhsListElementTy = getLhs().getType().getElementType();
  Type rhsListElementTy = getRhs().getType().getElementType();

  Type resultListElementTy = getType().getElementType();

  if (!isSubtype(lhsListElementTy, resultListElementTy)) {
    return emitError() << "LHS list element type must be a subtype of result "
                          "list element type: "
                       << lhsListElementTy << " v.s. " << resultListElementTy;
  }

  if (!isSubtype(rhsListElementTy, resultListElementTy)) {
    return emitError() << "RHS list element type must be a subtype of result "
                          "list element type: "
                       << rhsListElementTy << " v.s. " << resultListElementTy;
  }

  return success();
}

bool ListConcatOp::mayAlias(size_t operandIndex, size_t resultIndex) {
  // The resulting list contains shallow copies of two input lists' elements
  return true;
}

//===----------------------------------------------------------------------===//
// ListCreateOp
//===----------------------------------------------------------------------===//

LogicalResult ListCreateOp::verify() {
  Type resElemTy = getType().getElementType();
  for (Value input : getOperands()) {
    Type inTy = input.getType();
    if (!isSubtype(inTy, resElemTy)) {
      return emitError()
             << "input type must be a subtype of list element type: " << inTy
             << " v.s. " << resElemTy;
    }
  }
  return success();
}

bool ListCreateOp::mayAlias(size_t operandIndex, size_t resultIndex) {
  // The resulting list holds shallow copies of all input values.
  return true;
}

//===----------------------------------------------------------------------===//
// ListGetOp
//===----------------------------------------------------------------------===//

LogicalResult
ListGetOp::inferReturnTypes(MLIRContext *context,
                            std::optional<Location> location, Adaptor adaptor,
                            SmallVectorImpl<Type> &inferredReturnTypes) {
  Type inputListElemTy =
      cast<ListType>(adaptor.getInput().getType()).getElementType();
  inferredReturnTypes.emplace_back(inputListElemTy);
  return success();
}

LogicalResult ListGetOp::verify() {
  // Check index tensor type.
  // TODO: Should be IndexType.
  DeviceTensorType indexTensorTy = getIndex().getType();
  if (indexTensorTy.getDevice().getLabel() != kCPULabel) {
    return emitOpError()
           << "list index to insert must be a cpu tensor but it is "
           << indexTensorTy.getDevice();
  }
  Type indexDtype = indexTensorTy.getElementType();
  if (!indexDtype.isInteger(32) && !indexDtype.isInteger(64)) {
    return emitOpError()
           << "expect index tensor to have dtype si32 or si64, but got: "
           << indexDtype;
  }

  if (indexTensorTy.getTensorSpec().getNumElements() != 1) {
    return emitOpError() << "expect index tensor to have 1 element, but got: "
                         << indexTensorTy;
  }
  return success();
}

bool ListGetOp::mayAlias(size_t operandIndex, size_t resultIndex) {
  // The resulting value is a shallow copy of the specified element in the input
  // list.
  return operandIndex == 0;
}

//===----------------------------------------------------------------------===//
// ListInsertOp
//===----------------------------------------------------------------------===//

LogicalResult ListInsertOp::verify() {
  // Check input element type against output element type.
  Type inputElemTy = getInput().getType().getElementType();
  Type outputElemTy = getType().getElementType();
  if (!isSubtype(inputElemTy, outputElemTy)) {
    return emitError()
           << "input element type must be a subtype of output element type: "
           << inputElemTy << " v.s. " << outputElemTy;
  }

  // Check new element type against list element type.
  Type newElemTy = getElement().getType();
  if (!isSubtype(newElemTy, outputElemTy)) {
    return emitError()
           << "new element type must be a subtype of output element type: "
           << newElemTy << " v.s. " << outputElemTy;
  }

  // Check index tensor type.
  // TODO: Should be IndexType.
  DeviceTensorType indexTensorTy = getIndex().getType();
  if (indexTensorTy.getDevice().getLabel() != kCPULabel) {
    return emitOpError()
           << "list index to insert must be a cpu tensor but it is "
           << indexTensorTy.getDevice();
  }
  Type indexDtype = indexTensorTy.getElementType();
  if (!indexDtype.isInteger(32) && !indexDtype.isInteger(64)) {
    return emitOpError()
           << "expect index tensor to have dtype si32 or si64, but got: "
           << indexDtype;
  }
  if (indexTensorTy.getTensorSpec().getNumElements() != 1) {
    return emitOpError() << "expect index tensor to have 1 element, but got: "
                         << indexTensorTy;
  }
  return success();
}

bool ListInsertOp::mayAlias(size_t operandIndex, size_t resultIndex) {
  // The resulting list contains a shallow copies of input list elements and the
  // inserted element.
  return true;
}

LogicalResult ListSliceOp::verify() {
  DeviceTensorType startTensorTy = getStart().getType();
  DeviceTensorType endTensorTy = getEnd().getType();
  DeviceTensorType stepTensorTy = getStep().getType();

  Type startDtype = startTensorTy.getElementType();
  Type endDtype = endTensorTy.getElementType();
  Type stepDtype = stepTensorTy.getElementType();

  auto isValidIndexDtype = [](Type ty) {
    return ty.isInteger(32) || ty.isInteger(64);
  };

  auto isOneElementTensor = [](DeviceTensorType ty) {
    return ty.getTensorSpec().getNumElements() == 1;
  };

  if (!isValidIndexDtype(startDtype)) {
    return emitOpError()
           << "expected start tensor to have dtype si32 or si64, but got: "
           << startDtype;
  }

  if (!isValidIndexDtype(endDtype)) {
    return emitOpError()
           << "expected end tensor to have dtype si32 or si64, but got: "
           << endDtype;
  }

  if (!isValidIndexDtype(stepDtype)) {
    return emitOpError()
           << "expected step tensor to have dtype si32 or si64, but got: "
           << stepDtype;
  }

  if (!isOneElementTensor(startTensorTy)) {
    return emitOpError() << "expected start tensor to have 1 element, but got: "
                         << startTensorTy;
  }

  if (!isOneElementTensor(endTensorTy)) {
    return emitOpError() << "expected end tensor to have 1 element, but got: "
                         << endTensorTy;
  }

  if (!isOneElementTensor(stepTensorTy)) {
    return emitOpError() << "expected step tensor to have 1 element, but got: "
                         << stepTensorTy;
  }

  return success();
}

bool ListSliceOp::mayAlias(size_t operandIndex, size_t resultIndex) {
  // The resulting list contains a shallow copies of input list elements and the
  // inserted element.
  return true;
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

LogicalResult CallOp::verify() {
  auto module = (*this)->getParentOfType<ModuleOp>();

  if (!module)
    return emitOpError("must be nested inside a ModuleOp.");

  auto targetModel = module.lookupSymbol<ModelOp>(getModelAttr());
  if (!targetModel)
    return emitOpError() << "failed to find symbol " << getModelAttr();

  // Verify operand types match the execute block argument types.
  Block *executeBlock = targetModel.getExecuteBlock();
  // If the index was not yet added, remove the last operand of the execute
  // block because that's the index.
  auto expectedArgs =
      executeBlock->getArguments().drop_back(!getIsGlobalizedIndex());
  auto actualArgTypes = getOperandTypes();
  if (expectedArgs.size() != actualArgTypes.size()) {
    return emitOpError() << "operand types " << actualArgTypes
                         << " do not match expected argument types "
                         << executeBlock->getArgumentTypes() << " of "
                         << getModelAttr() << "'s execute block.";
  }

  for (auto [expectedArg, actualArgType] :
       llvm::zip(expectedArgs, actualArgTypes)) {
    if (expectedArg.getType() != actualArgType) {
      return emitOpError() << "operand types " << actualArgTypes
                           << " do not match expected argument types "
                           << executeBlock->getArgumentTypes() << " of "
                           << getModelAttr() << "'s execute block.";
    }
  }

  // Verify result types match the execute block result types.
  auto outputOp = dyn_cast<MEF::OutputOp>(executeBlock->getTerminator());
  if (!outputOp) {
    return emitOpError() << "execute block of " << getModelAttr()
                         << " does not end with an mef.output op.";
  }

  auto expectedResultTypes = outputOp.getOperandTypes();
  auto actualResultTypes = getResultTypes();
  if (actualResultTypes != expectedResultTypes) {
    return emitOpError() << "result types " << actualResultTypes
                         << " do not match expected result types "
                         << expectedResultTypes << " of " << getModelAttr()
                         << "'s execute block.";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// DebugPrintOp
//===----------------------------------------------------------------------===//

bool DebugPrintOp::isRaising() { return true; }

LogicalResult DebugPrintOp::emitMojo(MOGG::EmitMojoState &state) {
  KGEN::LIT::FnOp litFunction =
      state.analysis.getRegisteredLITMojoKernelFor(getOperationName());
  MOGG::MojoValue function = state.getOriginalMojoNameFor(litFunction);

  MOGG::MojoValue debugStringParam = MOGG::MojoValue(getADebugString()).quote();
  MOGG::MojoValue labelParam = MOGG::MojoValue(getBLabel()).quote();
  MOGG::MojoValue ctxVar = state.getMojoValueFor(getTheContext());

  state.builder.call(function, {debugStringParam, labelParam}, ctxVar);
  auto moggAsyncReadFunc =
      state.getOriginalMojoNameFor(state.analysis.getRegisteredLITMojoKernelFor(
          MOGG::AsyncReadyOp::getOperationName()));
  state.builder.call(moggAsyncReadFunc, {}, state.mojoReturnValues[0]);

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
  if (isSubgraph) {
    state.addAttribute(getIsSubgraphAttrName(state.name), UnitAttr::get(ctx));
    finalInputTypes.push_back(IndexType::get(ctx));
    finalInputLocs.push_back(state.location);
  }

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

  if (getIsSubgraph()) {
    auto argTypes = execBlock->getArgumentTypes();
    if (argTypes.empty() ||
        !isa<IndexType>(*(argTypes.begin() + argTypes.size() - 1))) {
      return emitOpError() << "expected the last argument of a subgraph to be "
                              "an index offset";
    }
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
#include "GenericML/MGPDialect/MGP.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "GenericML/MGPDialect/MGP.cpp.inc"
