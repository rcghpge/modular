//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/GraphCompiler/Interfaces/EmitMojoState.h"
#include "GraphCompiler/GraphCompiler/MGPDialect/Dialect/MGPOps.h"
#include "GraphCompiler/GraphCompiler/MGPDialect/Dialect/MGPTypes.h"
#include "GraphCompiler/GraphCompiler/MOGGDialect/Analysis/MojoRegistry.h"
#include "GraphCompiler/GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GraphCompiler/GraphCompiler/MOGGDialect/Support/MOToMOGGLowering.h"
#include "GraphCompiler/GraphCompiler/MOGGDialect/Support/MojoBuilder.h"
#include "GraphCompiler/Support/Diagnostic.h"
#include "KGEN/LITDialect/LITOps.h"
#include "Support/AssertStream.h"
#include "Support/Buffer.h"
#include "Support/Compiler/VerifyUtils.h"
#include "Support/ErrorOr.h"
#include "Support/MDialect/MAttrs.h"
#include "Support/MDialect/ParserUtils.h"
#include "Support/ML/CompiledFrameworkLabel.h"
#include "Support/ML/Permutation.h"
#include "Support/ML/SizeUtils.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/STLExtras.h"
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
static uint64_t computeRawAlign(ElementsAttr value) {
  ErrorOr<std::optional<uint64_t>> errOrOptAlign = getImpliedAlignment(value);
  if (errOrOptAlign.isError())
    return 0;

  // If error we'll report it when the result is validated.
  return optSizeToRawSize(*errOrOptAlign);
}

static llvm::StringRef primitiveMojoName(MOGG::PrimitiveDeclOp decl) {
  // sym_name is set to the fully-qualified Mojo source name by
  // ImportPackages, so it doubles as the identifier to emit in Mojo source.
  return decl.getSymNameAttr().strref();
}

// Helper to emit a typed comptime InlineArray[Int, N] declaration from
// attribute values and return the variable name.
template <typename T>
static MOGG::MojoValue inlineArray(MOGG::EmitMojoState &state,
                                   ArrayRef<T> values, bool isComptime = true) {
  auto mojoValues = llvm::map_to_vector(
      values, [](T value) { return MOGG::MojoValue(value); });

  std::string type;
  llvm::raw_string_ostream os(type);

  os << MOGG::kStructInlineArray << "[" << "Int" << ", " << values.size()
     << "]";

  auto mojoType = MOGG::MojoValue(std::move(type));

  if (isComptime)
    return state.builder.comptime(state.builder.listLiteral(mojoValues),
                                  /*type=*/&mojoType);

  auto var = state.builder.varDecl(mojoType);
  state.builder.assign(var, state.builder.listLiteral(mojoValues));
  return var;
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
  uint64_t align = getBAlign();
  if (!isValidOptAlign(align))
    return emitOpError("invalid alignment ") << align << ".";

  return success();
}

LogicalResult BufferConstantExternalOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, Adaptor adaptor,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  DeviceRefAttr device = adaptor.getCDeviceAttr();
  getDeviceAttrOrDefault(context, device);
  inferredReturnTypes.emplace_back(DeviceBufferType::get(context, device));
  return success();
}

//===----------------------------------------------------------------------===//
// BufferConstantOp
//===----------------------------------------------------------------------===//

void BufferConstantOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                             ElementsAttr value) {
  return build(odsBuilder, odsState, value, computeRawAlign(value));
}

/// Verifies the attribute and result types are consistent, and the attribute
/// is supported.
LogicalResult BufferConstantOp::verify() {
  // We can only construct from attributes which are dense arrays (ie rank-1)
  // of uint8_ts.
  ElementsAttr value = getValue();
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
  return rawSizeToOptSize(getRawAlign());
}

//===----------------------------------------------------------------------===//
// BufferAllocOp
//===----------------------------------------------------------------------===//
bool BufferAllocOp::isThrows() { return true; }

LogicalResult BufferAllocOp::emitMojo(MOGG::EmitMojoState &state) {
  using namespace MOGG;

  auto primDecl = state.registry.lookupPrimitive(getOperationName());
  MojoValue function(primitiveMojoName(primDecl));

  MojoValue byteSizeVar = state.getMojoValueFor(getRuntimeSize());
  MojoValue deviceCtxVar = state.getMojoValueFor(getDeviceContext());

  MojoValue result =
      state.builder.var(function({}, {byteSizeVar, deviceCtxVar}));
  state.setMojoValueFor(getResult(), result);

  return success();
}

LogicalResult BufferAllocOp::verify() {
  if (getDevice() != getOutBuffer().getType().getDevice()) {
    return emitOpError("expected result buffer device does not match actual "
                       "result buffer device (")
           << getDevice() << " vs " << getOutBuffer().getType().getDevice()
           << ").";
  }
  if (!isValidRawAlign(getRawAlign()))
    return emitOpError("invalid alignment ") << getRawAlign() << ".";
  return verifyConsistentDevices(getOperation(), {getDevice()});
}

LogicalResult BufferAllocOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, Adaptor adaptor,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  DeviceRefAttr device = adaptor.getDevice();
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
bool BufferHostToDeviceOp::isThrows() { return true; }

LogicalResult BufferHostToDeviceOp::emitMojo(MOGG::EmitMojoState &state) {
  using namespace MOGG;

  auto primDecl = state.registry.lookupPrimitive(getOperationName());
  MojoValue function(primitiveMojoName(primDecl));

  MojoValue hostDeviceParam =
      MojoValue(getHostDevice() ? getHostDevice().getLabel() : "cpu").quote();
  MojoValue otherDeviceParam =
      MojoValue(getOtherDevice() ? getOtherDevice().getLabel() : "gpu").quote();

  state.builder.call(function, {hostDeviceParam, otherDeviceParam},
                     state.getMojoValuesFor(getOperands().drop_front()));

  return success();
}

LogicalResult BufferHostToDeviceOp::verify() {
  if (getHostBuffer().getType().getDevice() != getHostDevice())
    return emitOpError("expected host buffer to be on 'host' device ")
           << getHostDevice() << ".";
  if (getOtherBuffer().getType().getDevice() != getOtherDevice())
    return emitOpError("expected other buffer to be on 'other' device ")
           << getOtherDevice() << ".";
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

bool BufferDeviceToHostOp::isThrows() { return true; }

LogicalResult BufferDeviceToHostOp::emitMojo(MOGG::EmitMojoState &state) {
  using namespace MOGG;

  auto primDecl = state.registry.lookupPrimitive(getOperationName());
  MojoValue function(primitiveMojoName(primDecl));

  MojoValue otherDeviceParam =
      MojoValue(getOtherDevice() ? getOtherDevice().getLabel() : "gpu").quote();
  MojoValue hostDeviceParam =
      MojoValue(getHostDevice() ? getHostDevice().getLabel() : "cpu").quote();

  state.builder.call(function, {otherDeviceParam, hostDeviceParam},
                     state.getMojoValuesFor(getOperands().drop_front()));

  return success();
}

LogicalResult BufferDeviceToHostOp::verify() {
  if (getHostBuffer().getType().getDevice() != getHostDevice())
    return emitOpError("expected host buffer to be on 'host' device ")
           << getHostDevice() << ".";
  if (getOtherBuffer().getType().getDevice() != getOtherDevice())
    return emitOpError("expected other buffer to be on 'other' device ")
           << getOtherDevice() << ".";
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

bool BufferDeviceToDeviceOp::isThrows() { return true; }

LogicalResult BufferDeviceToDeviceOp::emitMojo(MOGG::EmitMojoState &state) {
  using namespace MOGG;

  auto primDecl = state.registry.lookupPrimitive(getOperationName());
  MojoValue function(primitiveMojoName(primDecl));

  MojoValue srcDeviceParam = MojoValue(getSrcDevice().getLabel()).quote();
  MojoValue destDeviceParam = MojoValue(getDstDevice().getLabel()).quote();

  state.builder.call(function, {srcDeviceParam, destDeviceParam},
                     state.getMojoValuesFor(getOperands().drop_front()));

  return success();
}

LogicalResult BufferDeviceToDeviceOp::verify() {
  if (getSrcDevice().getLabel() != getDstDevice().getLabel()) {
    return emitOpError("expecting source and destination devices to have the "
                       "same device label.");
  }
  if (getSrcBuffer().getType().getDevice() != getSrcDevice() ||
      getDstBuffer().getType().getDevice() != getDstDevice()) {
    return emitOpError(
               "expecting input chain and source buffer to be on device '")
           << getSrcDevice()
           << "', and output chain and destination buffer to be on device '"
           << getDstDevice() << "'.";
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

bool BufferConcatOp::isThrows() { return true; }

LogicalResult BufferConcatOp::emitMojo(MOGG::EmitMojoState &state) {
  using namespace MOGG;

  auto primDecl = state.registry.lookupPrimitive(getOperationName());
  MojoValue function(primitiveMojoName(primDecl));

  MojoValue bDevice = getBDevice().getLabel();
  MojoValue output = state.getMojoValueFor(getOutBuffer());
  MojoValue ctx = state.getMojoValueFor(getCtx());

  MojoValue inputBuffers = state.builder.var(MojoValue(
      kStructStaticTuple)[{kComptimeMutByteBuffer, getInBuffers().size()}]());
  // The StaticTuple holds register-passable `MutByteBuffer` views; unwrap each
  // input buffer composite's `.view` (an `OwnedByteBuffer` cannot live in a
  // register-passable container).
  for (auto &&[i, buffer] : llvm::enumerate(getInBuffers()))
    state.builder.assign(
        inputBuffers[i],
        OwnedByteBufferRef(state.getMojoValueFor(buffer)).view());

  state.builder.call(function, bDevice.quote(), {output, inputBuffers, ctx});

  return success();
}

//===----------------------------------------------------------------------===//
// BufferSliceOp
//===----------------------------------------------------------------------===//
LogicalResult BufferSliceOp::emitMojo(MOGG::EmitMojoState &state) {
  using namespace MOGG;

  auto primDecl = state.registry.lookupPrimitive(getOperationName());
  MojoValue function(primitiveMojoName(primDecl));

  MojoValue res =
      state.builder.var(function({}, state.getMojoValuesFor(getOperands())));
  state.setMojoValueFor(getOutBuffer(), res);

  return success();
}

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
// BufferBulkSliceOp
//===----------------------------------------------------------------------===//

LogicalResult BufferBulkSliceOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, Adaptor adaptor,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  auto baseType = dyn_cast<DeviceBufferType>(adaptor.getBaseBuffer().getType());
  if (!baseType)
    return failure();

  auto resultType = DeviceBufferType::get(context, baseType.getDevice());
  inferredReturnTypes.append(adaptor.getOffsets().size(), resultType);
  return success();
}

LogicalResult BufferBulkSliceOp::verify() {
  if (!llvm::all_equal(
          {getOffsets().size(), getSizes().size(), getOutBuffers().size()}))
    return emitOpError("offsets, sizes, and results must have the same count");
  return success();
}

bool BufferBulkSliceOp::extendsLifetime(size_t operandIndex,
                                        size_t /*resultIndex*/) {
  return operandIndex == getBaseBufferMutable().getOperandNumber();
}

LogicalResult BufferBulkSliceOp::emitMojo(MOGG::EmitMojoState &state) {
  using namespace MOGG;

  auto primDecl = state.registry.lookupPrimitive(getOperationName());
  MojoValue function(primitiveMojoName(primDecl));

  auto offsets = llvm::map_to_vector(
      getOffsets(), [&](Value v) { return state.getMojoValueFor(v); });

  auto sizes = llvm::map_to_vector(
      getSizes(), [&](Value v) { return state.getMojoValueFor(v); });

  MojoValue offsetsArray =
      inlineArray(state, ArrayRef(offsets), /*isComptime=*/false);
  MojoValue sizesArray =
      inlineArray(state, ArrayRef(sizes), /*isComptime=*/false);
  MojoValue baseArg = state.getMojoValueFor(getBaseBuffer());

  // Call: mgp_buffer_bulk_slice(base, offsets, sizes)
  // Returns InlineArray[MutByteBuffer, N]; extract each result by index.
  MojoValue result =
      state.builder.var(function({}, {baseArg, offsetsArray, sizesArray}));

  for (auto &&[i, outBuf] : llvm::enumerate(getOutBuffers()))
    state.setMojoValueFor(outBuf, result[i]);

  return success();
}

//===----------------------------------------------------------------------===//
// BufferToIndexOp
//===----------------------------------------------------------------------===//
bool BufferToIndexOp::isThrows() { return true; }

LogicalResult BufferToIndexOp::emitMojo(MOGG::EmitMojoState &state) {
  using namespace MOGG;

  auto primDecl = state.registry.lookupPrimitive(getOperationName());
  MojoValue function(primitiveMojoName(primDecl));

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
  auto primDecl = state.registry.lookupPrimitive(getOperationName());
  MojoValue function(primitiveMojoName(primDecl));

  DeviceBufferType bufferType = getBuffer().getType();
  MojoValue deviceParam = MojoValue(bufferType.getDevice().getLabel()).quote();

  state.setMojoValueFor(
      getResult(), state.builder.var(function(
                       {deviceParam}, {state.getMojoValueFor(getBuffer())})));
  return success();
}

LogicalResult BufferToBoolOp::verify() {
  return verifyConsistentDevices(getOperation(), {getDevice()});
}

//===----------------------------------------------------------------------===//
// BufferSetWithIndexOp
//===----------------------------------------------------------------------===//
bool BufferSetWithIndexOp::isThrows() { return true; }

LogicalResult BufferSetWithIndexOp::emitMojo(MOGG::EmitMojoState &state) {
  using namespace MOGG;

  auto primDecl = state.registry.lookupPrimitive(getOperationName());
  MojoValue function(primitiveMojoName(primDecl));

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
  return verifyConsistentDevices(getOperation(), {getDevice()});
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

LogicalResult BufferPlanOp::emitMojo(MOGG::EmitMojoState &state) {
  using namespace MOGG;

  auto primDecl = state.registry.lookupPrimitive(getOperationName());
  MojoValue function(primitiveMojoName(primDecl));

  size_t numRuntimeSizes = getRuntimeSizes().size();

  // Emit compile-time InlineArray parameters.
  MojoValue alignmentsParam = inlineArray(state, getAlignments());
  MojoValue canShareParam = inlineArray(state, getCanShare());

  // Emit compile-time InlineArray for optional staticSizes attribute.
  MojoValue staticSizesParam;
  if (auto staticSizes = getStaticSizes()) {
    staticSizesParam = inlineArray(state, *staticSizes);
  } else {
    // Empty InlineArray for no static sizes.
    staticSizesParam = inlineArray(state, ArrayRef<int64_t>{});
  }

  // Collect all runtime sizes as MojoValue arguments.
  SmallVector<MojoValue> runtimeSizeArgs;
  for (Value size : getRuntimeSizes())
    runtimeSizeArgs.push_back(state.getMojoValueFor(size));

  // Emit a typed var for the runtime sizes: InlineArray[Int, N] = [v0, v1, ...]
  MojoValue runtimeSizesArray;
  {
    MojoValue name = state.builder.newVar("var");
    std::string str;
    llvm::raw_string_ostream stream(str);
    stream << "var " << name << ": " << kStructInlineArray << "[Int, "
           << numRuntimeSizes << "] = [";
    llvm::interleaveComma(runtimeSizeArgs, stream,
                          [&](const MojoValue &v) { stream << v; });
    stream << "]";
    state.builder.verbatim(MojoValue(str));
    runtimeSizesArray = name;
  }

  // Call the function with compile-time parameters and runtime arguments.
  // Function signature: [alignments, canShare, staticSizes](runtimeSizes)
  MojoValue result = state.builder.var(function(
      {alignmentsParam, canShareParam, staticSizesParam}, {runtimeSizesArray}));

  // Extract results: result is a tuple (highWatermark, offsets)
  state.setMojoValueFor(getHighWatermark(), result[0]);

  // Extract offsets from result[1] which is a StaticTuple
  for (auto &&[i, offset] : llvm::enumerate(getOffsets()))
    state.setMojoValueFor(offset, result[1][i]);

  return success();
}

namespace {

struct StaticizeAllocSizes final : public OpRewritePattern<BufferPlanOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(BufferPlanOp op,
                                PatternRewriter &rewriter) const override {
    int64_t numExistingStatic = static_cast<int64_t>(
        op.getStaticSizes().value_or(ArrayRef<int64_t>()).size());

    // Identify which runtime inputs are statically known
    SmallVector<int64_t> staticValues;

    llvm::SmallBitVector staticIndices(op.getRuntimeSizes().size());

    for (auto [i, runtimeSize] : llvm::enumerate(op.getRuntimeSizes())) {
      IntegerAttr constantAttr;
      if (matchPattern(runtimeSize, m_Constant(&constantAttr))) {
        // This runtime operand is actually a constant
        staticIndices.set(i);
        staticValues.push_back(constantAttr.getInt());
      }
    }

    // If we have no new static values, nothing to canonicalize
    if (staticIndices.none())
      return failure();

    llvm::SmallBitVector dynamicIndices =
        llvm::SmallBitVector(staticIndices).flip();

    // Build new runtime sizes operands (only the dynamic ones)
    SmallVector<Value> newRuntimeSizes;
    for (size_t idx : dynamicIndices.set_bits())
      newRuntimeSizes.push_back(op.getRuntimeSizes()[idx]);

    // Build permutation map for attributes
    // New order: [existing static..., newly discovered static..., dynamic...]
    // Permutation maps new index -> old index in the full allocation array
    // Existing static allocations stay in their positions
    SmallVector<int64_t> permutation =
        llvm::to_vector(llvm::seq(numExistingStatic));

    // Newly discovered static allocations (from runtime operands)
    for (int64_t idx : staticIndices.set_bits())
      permutation.push_back(numExistingStatic + idx);

    // Remaining dynamic allocations (from runtime operands)
    for (int64_t idx : dynamicIndices.set_bits())
      permutation.push_back(numExistingStatic + idx);

    // Permute the attributes according to the new order
    auto permuteAttr = [&](ArrayRef<uint64_t> attr) {
      return M::permute(attr, permutation);
    };

    // Permute the NxN canShare matrix: both rows and columns must be reordered.
    // permutation[newIdx] = oldIdx, so newMat[ni][nj] = oldMat[p[ni]][p[nj]].
    size_t n = permutation.size();
    ArrayRef<uint8_t> oldCanShare = op.getCanShare();
    SmallVector<uint8_t> newCanShareVec(n * n);
    for (size_t ni = 0; ni < n; ++ni)
      for (size_t nj = 0; nj < n; ++nj)
        newCanShareVec[ni * n + nj] =
            oldCanShare[permutation[ni] * n + permutation[nj]];

    // Build new staticSizes attribute
    SmallVector<int64_t> newStaticSizes =
        llvm::to_vector(op.getStaticSizes().value_or(ArrayRef<int64_t>{}));

    newStaticSizes.append(staticValues.begin(), staticValues.end());

    auto newStaticSizesAttr = rewriter.getAttr<IntArrayElementsAttr>(
        ArrayRef<int64_t>(newStaticSizes), IntegerType::Signed);

    // Create new operation with permuted attributes
    auto newOp =
        BufferPlanOp::create(rewriter, op.getLoc(),
                             /*runtimeSizes=*/newRuntimeSizes,
                             /*alignments=*/permuteAttr(op.getAlignments()),
                             /*canShare=*/newCanShareVec, newStaticSizesAttr);

    SmallVector<Value> newOffsets =
        llvm::to_vector(ValueRange(newOp.getOffsets()));

    // Permute the results back to match original order
    SmallVector<Value> newResults;
    newResults.push_back(newOp.getHighWatermark());
    newResults.append(M::permuteReverse(newOffsets, permutation));

    // Create inverse permutation for results
    rewriter.replaceOp(op, newResults);
    return success();
  }
};

} // namespace

void BufferPlanOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                               MLIRContext *ctx) {
  patterns.add<StaticizeAllocSizes>(ctx);
}

LogicalResult BufferPlanOp::verify() {
  size_t numStaticSizes = getStaticSizes() ? getStaticSizes()->size() : 0;
  size_t numRuntimeSizes = getRuntimeSizes().size();
  size_t totalAllocs = numStaticSizes + numRuntimeSizes;

  // The total number of offsets should equal staticSizes + runtimeSizes
  if (getOffsets().size() != totalAllocs) {
    return emitOpError("expected number of offsets (")
           << getOffsets().size() << ") to equal staticSizes size ("
           << numStaticSizes << ") + runtimeSizes size (" << numRuntimeSizes
           << ") = " << totalAllocs;
  }

  if (getAlignments().size() != totalAllocs) {
    return emitOpError("expected alignments to have size equal to total number "
                       "of allocations (")
           << totalAllocs << ")";
  }

  // canShare must be an NxN matrix
  if (getCanShare().size() != totalAllocs * totalAllocs) {
    return emitOpError("expected canShare to have size N*N = ")
           << totalAllocs * totalAllocs << " for N=" << totalAllocs
           << " allocations, but got " << getCanShare().size();
  }

  // Validate symmetry and zero diagonal
  ArrayRef<uint8_t> mat = getCanShare();
  for (size_t i : llvm::seq(totalAllocs)) {
    if (mat[i * totalAllocs + i] != 0)
      return emitOpError(
                 "expected canShare diagonal to be zero (allocation cannot "
                 "share with itself) at index ")
             << i;
    for (size_t j : llvm::seq(totalAllocs)) {
      if (mat[i * totalAllocs + j] != mat[j * totalAllocs + i])
        return emitOpError("expected canShare to be symmetric: canShare[")
               << i << "][" << j << "]=" << mat[i * totalAllocs + j]
               << " != canShare[" << j << "][" << i
               << "]=" << mat[j * totalAllocs + i];
    }
  }

  return success();
}

LogicalResult BufferPlanOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, Adaptor adaptor,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  ValueRange operands = adaptor.getOperands();
  size_t numStaticSizes =
      adaptor.getStaticSizes() ? adaptor.getStaticSizes()->size() : 0;
  size_t numRuntimeSizes = operands.size();
  size_t totalAllocations = numStaticSizes + numRuntimeSizes;

  // All outputs are index: 1 high watermark + N offsets
  inferredReturnTypes.clear();
  inferredReturnTypes.resize(totalAllocations + 1, IndexType::get(context));

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
bool DebugTensorPrintOp::isThrows() { return true; }

LogicalResult DebugTensorPrintOp::emitMojo(MOGG::EmitMojoState &state) {
  auto primDecl = state.registry.lookupPrimitive(getOperationName());
  MOGG::MojoValue function(primitiveMojoName(primDecl));

  auto tensorSpec = getSpec().getType();
  MOGG::MojoValue specRankParam = MOGG::MojoValue(int(tensorSpec.getRank()));
  MOGG::MojoValue dtypeParam =
      state.mlirDTypeToMojoType(tensorSpec.getOptDType().value());

  MOGG::MojoValue tensorSpecAsyncVar = state.getMojoValueFor(getSpec());
  MOGG::MojoValue bufferVar = state.getMojoValueFor(getBuffer());
  MOGG::MojoValue labelLiteral = MOGG::MojoValue(getLabel()).quote();
  MOGG::MojoValue labelString = state.builder.var(
      MOGG::MojoValue(MOGG::kStructString)({}, {labelLiteral}));
  MOGG::MojoStringRef label(labelString);
  MOGG::MojoValue labelPtrVar = label.unsafePtr();
  MOGG::MojoValue labelSizeVar = label.byteLength();

  state.builder.call(
      function, {specRankParam, dtypeParam},
      {bufferVar, tensorSpecAsyncVar, labelPtrVar, labelSizeVar});

  return success();
}

//===----------------------------------------------------------------------===//
// SyncOp
//===----------------------------------------------------------------------===//

bool SyncOp::isThrows() { return true; }

LogicalResult SyncOp::emitMojo(MOGG::EmitMojoState &state) {
  using namespace MOGG;

  auto primDecl = state.registry.lookupPrimitive(getOperationName());
  MojoValue function(primitiveMojoName(primDecl));

  state.builder.call(function, {},
                     state.getMojoValuesFor(getOperands().drop_front()));

  return success();
}

//===----------------------------------------------------------------------===//
// DeviceWaitOp
//===----------------------------------------------------------------------===//

bool DeviceWaitOp::isThrows() { return true; }

LogicalResult DeviceWaitOp::emitMojo(MOGG::EmitMojoState &state) {
  using namespace MOGG;

  auto primDecl = state.registry.lookupPrimitive(getOperationName());
  MojoValue function(primitiveMojoName(primDecl));

  // Drops the in-chain operand; passes (theContext, waitingDeviceContext,
  // signalingDeviceContext) to the primitive.
  state.builder.call(function, {},
                     state.getMojoValuesFor(getOperands().drop_front()));

  return success();
}
