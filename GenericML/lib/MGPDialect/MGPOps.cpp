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
#include "GenericML/MGPDialect/BufferBuilder.h"
#include "Support/AssertStream.h"
#include "Support/Compiler/ParserUtils.h"
#include "Support/Compiler/VerifyUtils.h"
#include "Support/ErrorOr.h"
#include "Support/MDialect/MAttrs.h"
#include "Support/ML/CompiledFrameworkLabel.h"
#include "Support/ML/DType.h"
#include "Support/ML/SizeUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/Support/Base64.h"

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

/// Returns device, or <"cpu", 0> if it is null.
static DeviceRefAttr deviceOrDefault(MLIRContext *ctx, DeviceRefAttr device) {
  if (device)
    return device;
  return DeviceRefAttr::getDefault(ctx);
}

/// Parses an optional keyword and #M.device_ref into device, then assigns
/// chains the type !mgp.chain<device>.
static ParseResult parseOptDeviceRefImpl(OpAsmParser &p, StringRef keyword,
                                         DeviceRefAttr &device,
                                         ArrayRef<Type *> chains) {
  if (succeeded(p.parseOptionalKeyword(keyword))) {
    llvm::SMLoc loc;
    if (p.getCurrentLocation(&loc))
      return failure();
    device = dyn_cast_if_present<DeviceRefAttr>(
        DeviceRefAttr::parse(p, /*odsType=*/{}));
    if (!device)
      return p.emitError(loc, "expecting #M.device_ref attribute");
  } else {
    device = DeviceRefAttr::getDefault(p.getContext());
  }
  for (Type *chain : chains)
    *chain = DeviceChainType::get(device);
  return success();
}

static ParseResult parseOptDeviceRef(OpAsmParser &p, StringRef keyword,
                                     DeviceRefAttr &device, Type &chain) {
  return parseOptDeviceRefImpl(p, keyword, device, {&chain});
}

static void printOptDeviceRef(OpAsmPrinter &p, Operation *op, StringRef keyword,
                              DeviceRefAttr device, DeviceChainType chain) {
  if (device != DeviceRefAttr::getDefault(op->getContext())) {
    p.printKeywordOrString(keyword);
    p.getStream() << " ";
    device.print(p);
  }
}

/// Parses a !mgp.buffer type, extracts its raw size and device attributes,
/// assigns chains the type !mgp.chain<device>.
static ParseResult parseBufferTypeToRawSizeDeviceImpl(OpAsmParser &p,
                                                      Type &buffer,
                                                      IntegerAttr &rawSize,
                                                      DeviceRefAttr &device,
                                                      ArrayRef<Type *> chains) {
  llvm::SMLoc loc;
  if (p.getCurrentLocation(&loc) || p.parseType(buffer))
    return failure();
  auto bufferType = dyn_cast<DeviceBufferType>(buffer);
  if (bufferType == nullptr)
    return p.emitError(loc) << "expected an !mgp.buffer type.";
  rawSize = bufferType.getRawSize();
  device = bufferType.getDevice();
  for (Type *chain : chains)
    *chain = DeviceChainType::get(device);
  return success();
}

static ParseResult parseBufferTypeToRawSizeDevice(OpAsmParser &p, Type &buffer,
                                                  IntegerAttr &rawSize,
                                                  DeviceRefAttr &device) {
  return parseBufferTypeToRawSizeDeviceImpl(p, buffer, rawSize, device, {});
}

static void printBufferTypeToRawSizeDevice(OpAsmPrinter &p, Operation *op,
                                           DeviceBufferType buffer,
                                           IntegerAttr rawSize,
                                           DeviceRefAttr device) {
  p << buffer;
}

static ParseResult parseStaticBufferTypeToSizeDevice(OpAsmParser &p,
                                                     Type &buffer,
                                                     IntegerAttr &size,
                                                     DeviceRefAttr &device) {
  mlir::SMLoc loc;
  if (p.getCurrentLocation(&loc) ||
      parseBufferTypeToRawSizeDeviceImpl(p, buffer, size, device, {}))
    return failure();
  if (!rawSizeToOptSize(size.getUInt()))
    return p.emitError(loc, "expecting statically sized !mgp.buffer type");
  return success();
}

static void printStaticBufferTypeToSizeDevice(OpAsmPrinter &p, Operation *op,
                                              DeviceBufferType buffer,
                                              IntegerAttr rawSize,
                                              DeviceRefAttr device) {
  p << buffer;
}

static ParseResult parseStaticBufferTypeToSizeDevice(OpAsmParser &p,
                                                     Type &buffer,
                                                     IntegerAttr &size,
                                                     DeviceRefAttr &device,
                                                     Type &chain) {
  mlir::SMLoc loc;
  if (p.getCurrentLocation(&loc) ||
      parseBufferTypeToRawSizeDeviceImpl(p, buffer, size, device, {&chain}))
    return failure();
  if (!rawSizeToOptSize(size.getUInt()))
    return p.emitError(loc, "expecting statically sized !mgp.buffer type");
  return success();
}

static void printStaticBufferTypeToSizeDevice(OpAsmPrinter &p, Operation *op,
                                              DeviceBufferType buffer,
                                              IntegerAttr rawSize,
                                              DeviceRefAttr device,
                                              DeviceChainType chain) {
  p << buffer;
}

static ParseResult parseStaticBufferTypeToSizeDeviceBuffer(
    OpAsmParser &p, Type &staticBuffer, IntegerAttr &size,
    DeviceRefAttr &device, Type &dynamicBuffer) {
  mlir::SMLoc loc;
  if (p.getCurrentLocation(&loc) ||
      parseBufferTypeToRawSizeDeviceImpl(p, staticBuffer, size, device, {}))
    return failure();
  if (!rawSizeToOptSize(size.getUInt()))
    return p.emitError(loc, "expecting statically sized !mgp.buffer type");
  dynamicBuffer = DeviceBufferType::get(device, std::optional<uint64_t>());
  return success();
}

static void printStaticBufferTypeToSizeDeviceBuffer(
    OpAsmPrinter &p, Operation *op, DeviceBufferType staticBuffer,
    IntegerAttr rawSize, DeviceRefAttr device, DeviceBufferType dynamicBuffer) {
  p << staticBuffer;
}

static ParseResult parseDynamicBufferTypeToDevice(OpAsmParser &p, Type &buffer,
                                                  DeviceRefAttr &device,
                                                  Type &chain) {
  IntegerAttr rawSize;
  mlir::SMLoc loc;
  if (p.getCurrentLocation(&loc) ||
      parseBufferTypeToRawSizeDeviceImpl(p, buffer, rawSize, device, {&chain}))
    return failure();
  if (rawSizeToOptSize(rawSize.getUInt()))
    return p.emitError(loc, "expecting dynamically sized !mgp.buffer type");
  return success();
}

static void printDynamicBufferTypeToDevice(OpAsmPrinter &p, Operation *op,
                                           DeviceBufferType buffer,
                                           DeviceRefAttr device,
                                           DeviceChainType chain) {
  p << buffer;
}

static ParseResult parseBufferTypeToDevice(OpAsmParser &p, Type &buffer,
                                           DeviceRefAttr &device) {
  IntegerAttr rawSize;
  return parseBufferTypeToRawSizeDeviceImpl(p, buffer, rawSize, device, {});
}

static void printBufferTypeToDevice(OpAsmPrinter &p, Operation *op,
                                    DeviceBufferType buffer,
                                    DeviceRefAttr device) {
  p << buffer;
}

static ParseResult parseBufferTypeToDevice(OpAsmParser &p, Type &buffer,
                                           DeviceRefAttr &device, Type &chain) {
  IntegerAttr rawSize;
  return parseBufferTypeToRawSizeDeviceImpl(p, buffer, rawSize, device,
                                            {&chain});
}

static void printBufferTypeToDevice(OpAsmPrinter &p, Operation *op,
                                    DeviceBufferType buffer,
                                    DeviceRefAttr device,
                                    DeviceChainType chain) {
  p << buffer;
}

static ParseResult parseBufferTypeToChain(OpAsmParser &p, Type &buffer,
                                          Type &chain) {
  IntegerAttr rawSize;
  DeviceRefAttr device;
  return parseBufferTypeToRawSizeDeviceImpl(p, buffer, rawSize, device,
                                            {&chain});
}

static void printBufferTypeToChain(OpAsmPrinter &p, Operation *op,
                                   DeviceBufferType buffer,
                                   DeviceChainType chain) {
  p << buffer;
}

/// Parses a !mgp.tensor type and extracts its device, buffer type, and
/// tensor spec type, and assigns chain the type !mgp.chain<device>.
/// If isShapeOnly is true, the tensor spec will have 'unk' dtype.
static ParseResult parseTensorTypeToDeviceBufferSpecImpl(
    bool isShapeOnly, OpAsmParser &p, Type &tensor, DeviceRefAttr &device,
    Type &buffer, Type &tensorSpec, ArrayRef<Type *> chains) {
  llvm::SMLoc loc;
  if (p.getCurrentLocation(&loc) || p.parseType(tensor))
    return failure();
  auto tensorType = dyn_cast<DeviceTensorType>(tensor);
  if (tensorType == nullptr)
    return p.emitError(loc) << "expected an !mgp.tensor type.";
  device = tensorType.getDevice();
  if (isShapeOnly) {
    tensorSpec = TensorSpecType::get(p.getContext(),
                                     tensorType.getTensorSpec().getDims(),
                                     std::optional<DType>());
  } else {
    tensorSpec = tensorType.getTensorSpec();
  }
  buffer = DeviceBufferType::get(tensorType);
  for (Type *chain : chains)
    *chain = DeviceChainType::get(device);
  return success();
}

static ParseResult parseTensorTypeToDeviceSpecOrShape(OpAsmParser &p,
                                                      BoolAttr isShapeOnly,
                                                      Type &tensor,
                                                      DeviceRefAttr &device,
                                                      Type &shape) {
  Type buffer;
  return parseTensorTypeToDeviceBufferSpecImpl(
      isShapeOnly && isShapeOnly.getValue(), p, tensor, device, buffer, shape,
      {});
}

static void printTensorTypeToDeviceSpecOrShape(OpAsmPrinter &p, Operation *op,
                                               BoolAttr isShapeOnly,
                                               DeviceTensorType tensor,
                                               DeviceRefAttr device,
                                               TensorSpecType shape) {
  p << tensor;
}

static ParseResult parseTensorTypeToDeviceBuffer(OpAsmParser &p, Type &tensor,
                                                 DeviceRefAttr &device,
                                                 Type &buffer) {
  Type tensorSpec;
  return parseTensorTypeToDeviceBufferSpecImpl(/*isShapeOnly=*/false, p, tensor,
                                               device, buffer, tensorSpec, {});
}

static void printTensorTypeToDeviceBuffer(OpAsmPrinter &p, Operation *op,
                                          DeviceTensorType tensor,
                                          DeviceRefAttr device,
                                          DeviceBufferType buffer) {
  p << tensor;
}

static ParseResult parseTensorTypeToDeviceBufferSpec(OpAsmParser &p,
                                                     DeviceRefAttr &device,
                                                     Type &buffer,
                                                     Type &tensorSpec) {
  Type tensor;
  return parseTensorTypeToDeviceBufferSpecImpl(/*isShapeOnly=*/false, p, tensor,
                                               device, buffer, tensorSpec, {});
}

static void printTensorTypeToDeviceBufferSpec(OpAsmPrinter &p, Operation *op,
                                              DeviceRefAttr device,
                                              DeviceBufferType buffer,
                                              TensorSpecType tensorSpec) {
  auto tensorType = DeviceTensorType::get(device, tensorSpec);
  p << tensorType;
}

static ParseResult
parseTensorTypeToDeviceBufferSpec(OpAsmParser &p, Type &tensor,
                                  DeviceRefAttr &device, Type &buffer,
                                  Type &tensorSpec, Type &chain) {
  return parseTensorTypeToDeviceBufferSpecImpl(
      /*isShapeOnly=*/false, p, tensor, device, buffer, tensorSpec, {&chain});
}

static void printTensorTypeToDeviceBufferSpec(OpAsmPrinter &p, Operation *op,
                                              DeviceTensorType tensorType,
                                              DeviceRefAttr device,
                                              DeviceBufferType buffer,
                                              TensorSpecType tensorSpec,
                                              DeviceChainType chain) {
  p << tensorType;
}

/// Parses a function type consisting of one input buffer and N output
/// buffer types, all sharing the same device assigned to device, assigns
/// outBufferRawSizes the raw sizes from the outBuffer types, and assigns chain
/// the type !mgp.chain<device>.
static ParseResult parseBufferTypesToBufferRawSizesDeviceOutBuffers(
    OpAsmParser &p, Type &inBuffer, IntArrayElementsAttr &outBufferRawSizesAttr,
    DeviceRefAttr &device, SmallVectorImpl<Type> &outBuffers, Type &chain) {
  llvm::SMLoc loc;
  if (p.getCurrentLocation(&loc) || p.parseType(inBuffer))
    return failure();
  auto inBufferType = dyn_cast<DeviceBufferType>(inBuffer);
  if (!inBufferType)
    return p.emitError(loc, "expected an !mgp.buffer type");
  device = inBufferType.getDevice();
  chain = DeviceChainType::get(device);
  SmallVector<APInt> rawDims;
  if (p.getCurrentLocation(&loc) || p.parseArrowTypeList(outBuffers))
    return failure();
  rawDims.reserve(outBuffers.size());
  for (const auto &outBuffer : outBuffers) {
    auto outBufferType = dyn_cast<DeviceBufferType>(outBuffer);
    if (!outBufferType)
      return p.emitError(loc, "expected an !mgp.buffer type");
    if (outBufferType.getDevice() != device)
      return p.emitError(loc, "all !mgp.buffer types must agree on device");
    rawDims.emplace_back(outBufferType.getRawSize().getValue());
  }
  Type ui64 = IntegerType::get(p.getContext(), 64, IntegerType::Unsigned);
  outBufferRawSizesAttr = IntArrayElementsAttr::get(
      ArrayType::get(static_cast<int64_t>(rawDims.size()), ui64), rawDims);
  return success();
}

static void printBufferTypesToBufferRawSizesDeviceOutBuffers(
    OpAsmPrinter &p, Operation *op, DeviceBufferType inBuffer,
    IntArrayElementsAttr outBufferRawSizesAttr, DeviceRefAttr device,
    TypeRange outBuffers, DeviceChainType chain) {
  p << inBuffer;
  p.printArrowTypeList(outBuffers);
}

// Parses a function type consisting of N input buffers and one output
// buffer type, assigns their (consistent!) device to device, and
// assigns chain the type !mgp.chain<device>.
static ParseResult
parseBufferTypesToDevice(OpAsmParser &p, SmallVectorImpl<Type> &inBuffers,
                         Type &outBuffer, DeviceRefAttr &device, Type &chain) {
  llvm::SMLoc loc;
  if (p.getCurrentLocation(&loc) || p.parseLParen() ||
      p.parseTypeList(inBuffers) || p.parseRParen() || p.parseArrow() ||
      p.parseType(outBuffer))
    return failure();

  for (auto inBuffer : inBuffers) {
    auto inBufferType = dyn_cast<DeviceBufferType>(inBuffer);
    if (!inBufferType)
      return p.emitError(loc, "expected an !mgp.buffer type");
    if (device && device != inBufferType.getDevice())
      return p.emitError(loc, "all !mgp.buffer types must agree on device");
    device = inBufferType.getDevice();
  }

  auto outBufferType = dyn_cast<DeviceBufferType>(outBuffer);
  if (!outBufferType)
    return p.emitError(loc, "expected an !mgp.buffer type");
  if (device && device != outBufferType.getDevice())
    return p.emitError(loc, "all !mgp.buffer types must agree on device");
  device = outBufferType.getDevice();

  chain = DeviceChainType::get(device);
  return success();
}

static void printBufferTypesToDevice(OpAsmPrinter &p, Operation *op,
                                     TypeRange inBuffers, Type outBuffer,
                                     DeviceRefAttr device,
                                     DeviceChainType chain) {
  p.getStream() << "(";
  llvm::interleaveComma(inBuffers, p.getStream(),
                        [&](Type inBuffer) { p.printType(inBuffer); });
  p.getStream() << ") -> ";
  p.printType(outBuffer);
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
  inChain = DeviceChainType::get(srcDevice);
  return success();
}

static void printSrcDstBufferTypesToDevices(OpAsmPrinter &p, Operation *op,
                                            DeviceBufferType srcBuffer,
                                            DeviceBufferType dstBuffer,
                                            DeviceRefAttr srcDevice,
                                            DeviceRefAttr dstDevice,
                                            DeviceChainType inChain) {
  p << srcBuffer;
  p << " -> ";
  p << dstBuffer;
}

/// Parses optionally type-annotated operands into $gatingOperands, using
/// the default type !mgp.chain<$device>.
static ParseResult parseChainCreateParenOperandList(
    OpAsmParser &p, DeviceRefAttr device,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &gatingOperands,
    SmallVectorImpl<Type> &gatingOperandTypes) {
  Type defaultType =
      DeviceChainType::get(deviceOrDefault(p.getContext(), device));
  return parseParenOperandListWithDefaultType(p, gatingOperands,
                                              gatingOperandTypes, defaultType);
}

static void printChainCreateParenOperandList(
    OpAsmPrinter &p, Operation *op, DeviceRefAttr device,
    const OperandRange &gatingOperands, const TypeRange &gatingOperandTypes) {
  Type defaultType = DeviceChainType::get(device);
  return printParenOperandListWithDefaultType(p, gatingOperands, defaultType);
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
// CpuTensorSpecCreateOp
//===----------------------------------------------------------------------===//

void CpuTensorSpecCreateOp::build(OpBuilder &odsBuilder,
                                  OperationState &odsState, TensorSpecType spec,
                                  ValueRange runtimeDims) {
  ASSERT_STREAM(spec.hasRank(),
                << "can only construct an mgp.tensor_spec.create "
                   "for tensor specs of statically known rank");
  CpuTensorSpecCreateOp::build(odsBuilder, odsState,
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
static void printRawDimsAndDType(AsmPrinter &p, const CpuTensorSpecCreateOp &op,
                                 IntArrayElementsAttr rawDims,
                                 IntegerAttr rawDType) {
  p << TensorSpecType::get(rawDims, rawDType);
}

LogicalResult CpuTensorSpecCreateOp::verify() {
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

LogicalResult CpuTensorSpecCreateOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, mlir::OpaqueProperties properties,
    mlir::RegionRange regions, SmallVectorImpl<Type> &inferredReturnTypes) {
  mlir::OperationName name(getOperationName(), context);
  auto rawDimsAttr = dyn_cast_if_present<IntArrayElementsAttr>(
      attributes.get(getARawDimsAttrName(name)));
  auto rawDTypeAttr = dyn_cast_if_present<IntegerAttr>(
      attributes.get(getBRawDTypeAttrName(name)));
  ASSERT_STREAM(rawDimsAttr && rawDTypeAttr, << "mismatched operand type");
  inferredReturnTypes.emplace_back(
      TensorSpecType::get(rawDimsAttr, rawDTypeAttr));
  return success();
}

//===----------------------------------------------------------------------===//
// CpuTensorSpecEqualStaticOp
//===----------------------------------------------------------------------===//

static void printTensorSpecTypeWithRawDims(AsmPrinter &p,
                                           const CpuTensorSpecEqualStaticOp &op,
                                           Type specType,
                                           IntArrayElementsAttr rawDims) {
  p << specType;
}

//===----------------------------------------------------------------------===//
// CpuTensorSpecDimsDynamicOp
//===----------------------------------------------------------------------===//

/// Parses a !mgp.tensor_spec type, however also sets the rawDims to
/// capture it's dimensions in 'raw form'.
static ParseResult
parseTensorSpecTypeWithRawDims(AsmParser &p, Type &specType,
                               IntArrayElementsAttr &rawDims) {
  llvm::SMLoc loc;
  if (p.getCurrentLocation(&loc) || p.parseType(specType))
    return failure();
  auto tensorSpec = dyn_cast<TensorSpecType>(specType);
  if (tensorSpec == nullptr)
    return p.emitError(loc) << "expected !mgp.tensor_spec type.";
  if (!tensorSpec.hasRank())
    return p.emitError(loc) << "expected !mgp.tensor_spec type to have a "
                               "statically known rank.";
  rawDims = tensorSpec.getRawDims();
  return success();
}

static void printTensorSpecTypeWithRawDims(AsmPrinter &p,
                                           const CpuTensorSpecDimsDynamicOp &op,
                                           Type specType,
                                           IntArrayElementsAttr rawDims) {
  p << specType;
}

LogicalResult CpuTensorSpecDimsDynamicOp::verify() {
  TensorSpecType actualSpecType = getSpec().getType();
  TensorSpecType impliedSpecType = TensorSpecType::get(
      getContext(), getRawDims(), actualSpecType.getRawDType().getUInt());
  assert(impliedSpecType.hasRank());
  if (actualSpecType != impliedSpecType)
    return emitOpError() << "expected tensor spec type is " << actualSpecType
                         << ", but static dimensions arguments "
                            "suggest it should be "
                         << impliedSpecType << ".";
  if (getResults().size() != actualSpecType.getNumDynamicDims())
    return emitOpError() << "expected " << actualSpecType.getNumDynamicDims()
                         << " dimension results to match the tensor spec "
                         << actualSpecType << ", but given "
                         << getResults().size();
  return success();
}

LogicalResult CpuTensorSpecDimsDynamicOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, mlir::OpaqueProperties properties,
    mlir::RegionRange regions, SmallVectorImpl<Type> &inferredReturnTypes) {
  ASSERT_STREAM(operands.size() == 1, << "mismatched operand arity");
  auto specType = dyn_cast<TensorSpecType>(operands[0].getType());
  ASSERT_STREAM(specType, << "mismatched operand type");
  for (size_t i = 0, e = specType.getNumDynamicDims(); i < e; ++i)
    inferredReturnTypes.emplace_back(IndexType::get(context));
  return success();
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

LogicalResult ListGetOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, mlir::OpaqueProperties properties,
    mlir::RegionRange regions, SmallVectorImpl<Type> &inferredReturnTypes) {
  Type inputListElemTy = cast<ListType>(operands[0].getType()).getElementType();
  inferredReturnTypes.emplace_back(inputListElemTy);
  return success();
}

LogicalResult ListGetOp::verify() {
  // Check index tensor type.
  // TODO: Should be IndexType.
  DeviceTensorType indexTensorTy = getIndex().getType();
  if (indexTensorTy.getDevice().getLabel() != "cpu") {
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
  if (indexTensorTy.getDevice().getLabel() != "cpu") {
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

LogicalResult GuardOp::inferReturnTypes(
    ::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location,
    ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes,
    ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes) {

  for (Value val : llvm::drop_begin(operands))
    inferredReturnTypes.push_back(val.getType());

  return success();
}

bool GuardOp::mayAlias(size_t operandIndex, size_t resultIndex) { return true; }

//===----------------------------------------------------------------------===//
// ModelOp
//===----------------------------------------------------------------------===//

/// Returns the implicit signature of the 'init' entry point.
static FunctionType implicitInitSignature(Builder &builder, size_t devices) {
  SmallVector<Type> outputs = {builder.getType<MEF::ChainType>(),
                               builder.getType<MGP::ContextType>()};
  for (size_t i = 0; i < devices; i++)
    outputs.push_back(OpaqueType::get(builder.getContext(), {}));
  return builder.getFunctionType({}, outputs);
}

/// Returns the implicit signature of the 'execute' entry point, before
/// any explicit inputs and results have been included.
static FunctionType implicitExecuteSignature(Builder &builder, size_t devices) {
  SmallVector<Type> inputs = {builder.getType<MEF::ChainType>(),
                              builder.getType<MGP::ContextType>()};
  for (size_t i = 0; i < devices; i++)
    inputs.push_back(OpaqueType::get(builder.getContext(), {}));
  return builder.getFunctionType(inputs, {builder.getType<MEF::ChainType>()});
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
                                      ArrayAttr &devices) {
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
  FunctionType implicitExecSig =
      implicitExecuteSignature(p.getBuilder(), devices.size());
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
                    ArrayRef<Location> inputArgLocs,
                    ArrayRef<NamedAttribute> extraAttrs) {
  DictionaryAttr extraAttrsDict =
      extraAttrs.empty() ? nullptr
                         : DictionaryAttr::get(state.getContext(), extraAttrs);
  auto deviceAttrs = llvm::map_to_vector(devices, [&state](DeviceRef ref) {
    return dyn_cast<Attribute>(
        DeviceRefAttr::get(state.getContext(), ref.label, ref.id));
  });
  build(builder, state, ArrayAttr::get(state.getContext(), deviceAttrs),
        StringAttr::get(state.getContext(), frameworkName),
        TypeAttr::get(builder.getFunctionType(inputTypes, outputTypes)), name,
        extraAttrsDict);

  assert(implicitInitSignature(builder, devices.size()).getInputs().empty());
  Region *initRegion = state.regions[0].get();
  initRegion->emplaceBlock();
  for (auto dev : devices) {
    initRegion->addArgument(OpaqueType::get(builder.getContext(), {}),
                            state.location);
  }

  // The 'execute' entry point has both implicit and explicit arguments.
  Region *executeRegion = state.regions[1].get();
  executeRegion->emplaceBlock();
  for (auto type :
       implicitExecuteSignature(builder, devices.size()).getInputs())
    executeRegion->addArgument(type, state.location);
  for (auto [type, loc] : llvm::zip(inputTypes, inputArgLocs))
    executeRegion->addArgument(type, loc);
}

/// Verifies the init and execute regions of the model.
LogicalResult ModelOp::verifyRegions() {
  OpBuilder builder(getContext());

  // The type of the init region is
  //  () -> (!mef.chain, !mgp.context)
  Block *initBlock = getInitBlock();
  if (auto output = dyn_cast<MEF::OutputOp>(initBlock->getTerminator())) {
    FunctionType implicitInitSig =
        implicitInitSignature(builder, getDevices().size());
    assert(implicitInitSig.getInputs().empty());
    if (initBlock->getArgumentTypes().size() != getDevices().size()) {
      return emitOpError() << "Expected the number of arguments to the 'init' "
                              "body to match the number of devices, but got "
                           << initBlock->getArgumentTypes().size() << " vs "
                           << getDevices().size() << ".";
    }
    for (auto t : initBlock->getArgumentTypes()) {
      if (!mlir::isa<OpaqueType>(t)) {
        return emitOpError()
               << "init block arguments must be !mgp.opaque, but got " << t;
      }
    }

    if (failed(checkOperandTypes(output, implicitInitSig.getResults())))
      return failure();
  }

  // The type of the execute region is:
  //   (!mef.chain, !mgp.context, arg_1, ... arg_n) ->
  //   (!mef.chain, res_1, ... res_m)
  // where the ModelOp's inputOutputTypes are:
  //   (arg_1, ... arg_n) -> (res_1, ... res_m)
  Block *execBlock = getExecuteBlock();
  if (auto output = dyn_cast<MEF::OutputOp>(execBlock->getTerminator())) {
    FunctionType implicitExecSig =
        implicitExecuteSignature(builder, getDevices().size());
    ArrayRef<Type> modelArgTypes = getInputOutputTypes().getInputs();
    SmallVector<Type> expectedArgTypes(implicitExecSig.getInputs());
    expectedArgTypes.append(modelArgTypes.begin(), modelArgTypes.end());
    if (execBlock->getArgumentTypes() != expectedArgTypes)
      return emitOpError() << "expected the arguments of the "
                              "'execute' body to be "
                           << expectedArgTypes << ", but got "
                           << execBlock->getArgumentTypes() << ".";

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
