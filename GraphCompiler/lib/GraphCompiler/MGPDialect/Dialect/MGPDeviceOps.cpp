//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/GraphCompiler/Interfaces/EmitMojoState.h"
#include "GraphCompiler/GraphCompiler/MGPDialect/Dialect/MGPOps.h"
#include "GraphCompiler/GraphCompiler/MGPDialect/Dialect/MGPTypes.h"
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
bool BufferAllocOp::isThrows() { return true; }

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
bool BufferHostToDeviceOp::isThrows() { return true; }

LogicalResult BufferHostToDeviceOp::emitMojo(MOGG::EmitMojoState &state) {
  using namespace MOGG;

  KGEN::LIT::FnOp litFunction =
      state.analysis.getRegisteredLITMojoKernelFor(getOperationName());
  MojoValue function = state.getOriginalMojoNameFor(litFunction);

  MojoValue hostDeviceParam =
      MojoValue(getCHostDevice() ? getCHostDevice().getLabel() : "cpu").quote();
  MojoValue otherDeviceParam =
      MojoValue(getDOtherDevice() ? getDOtherDevice().getLabel() : "gpu")
          .quote();

  state.builder.call(function, {hostDeviceParam, otherDeviceParam},
                     state.getMojoValuesFor(getOperands().drop_front()));

  return success();
}

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

bool BufferDeviceToHostOp::isThrows() { return true; }

LogicalResult BufferDeviceToHostOp::emitMojo(MOGG::EmitMojoState &state) {
  using namespace MOGG;

  KGEN::LIT::FnOp litFunction =
      state.analysis.getRegisteredLITMojoKernelFor(getOperationName());
  MojoValue function = state.getOriginalMojoNameFor(litFunction);

  MojoValue otherDeviceParam =
      MojoValue(getCOtherDevice() ? getCOtherDevice().getLabel() : "gpu")
          .quote();
  MojoValue hostDeviceParam =
      MojoValue(getDHostDevice() ? getDHostDevice().getLabel() : "cpu").quote();

  state.builder.call(function, {otherDeviceParam, hostDeviceParam},
                     state.getMojoValuesFor(getOperands().drop_front()));

  return success();
}

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

bool BufferDeviceToDeviceOp::isThrows() { return true; }

LogicalResult BufferDeviceToDeviceOp::emitMojo(MOGG::EmitMojoState &state) {
  using namespace MOGG;

  KGEN::LIT::FnOp litFunction =
      state.analysis.getRegisteredLITMojoKernelFor(getOperationName());
  MojoValue function = state.getOriginalMojoNameFor(litFunction);

  MojoValue srcDeviceParam = MojoValue(getCSrcDevice().getLabel()).quote();
  MojoValue destDeviceParam = MojoValue(getDDstDevice().getLabel()).quote();

  state.builder.call(function, {srcDeviceParam, destDeviceParam},
                     state.getMojoValuesFor(getOperands().drop_front()));

  return success();
}

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

bool BufferConcatOp::isThrows() { return true; }

LogicalResult BufferConcatOp::emitMojo(MOGG::EmitMojoState &state) {
  using namespace MOGG;

  KGEN::LIT::FnOp litFunction =
      state.analysis.getRegisteredLITMojoKernelFor(getOperationName());
  MojoValue function = state.getOriginalMojoNameFor(litFunction);

  MojoValue bDevice = getBDevice().getLabel();
  MojoValue output = state.getMojoValueFor(getOutBuffer());
  MojoValue ctx = state.getMojoValueFor(getCtx());

  MojoValue inputBuffers = state.builder.var(MojoValue(
      HardcodedMojoSymbols::struct_StaticTuple)[{
      HardcodedMojoSymbols::comptime_MutByteBuffer, getInBuffers().size()}]());
  for (auto &&[i, buffer] : llvm::enumerate(getInBuffers()))
    state.builder.assign(inputBuffers[i], state.getMojoValueFor(buffer));

  state.builder.call(function, bDevice.quote(), {output, inputBuffers, ctx});

  return success();
}

//===----------------------------------------------------------------------===//
// BufferSliceOp
//===----------------------------------------------------------------------===//
mlir::BitVector BufferSliceOp::resultsToPack() {
  return mlir::BitVector{1, false};
}

LogicalResult BufferSliceOp::emitMojo(MOGG::EmitMojoState &state) {
  using namespace MOGG;

  KGEN::LIT::FnOp litFunction =
      state.analysis.getRegisteredLITMojoKernelFor(getOperationName());
  MojoValue function = state.getOriginalMojoNameFor(litFunction);

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
// BufferToIndexOp
//===----------------------------------------------------------------------===//
bool BufferToIndexOp::isThrows() { return true; }

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
      getResult(), state.builder.var(function(
                       {deviceParam}, {state.getMojoValueFor(getBuffer())})));
  return success();
}

LogicalResult BufferToBoolOp::verify() {
  return verifyConsistentDevices(getOperation(), {getBDevice()});
}

//===----------------------------------------------------------------------===//
// BufferSetWithIndexOp
//===----------------------------------------------------------------------===//
bool BufferSetWithIndexOp::isThrows() { return true; }

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

// Helper to emit a typed comptime InlineArray[Int, N] declaration from
// attribute values and return the variable name.
template <typename T>
static MOGG::MojoValue inlineArray(MOGG::EmitMojoState &state,
                                   ArrayRef<T> values) {
  static_assert(std::is_integral_v<T>, "Must be an integer type");

  auto mojoValues = llvm::map_to_vector(
      values, [](T value) { return MOGG::MojoValue(value); });

  std::string type;
  llvm::raw_string_ostream os(type);

  os << MOGG::HardcodedMojoSymbols::struct_InlineArray << "[" << "Int" << ", "
     << values.size() << "]";

  auto mojoType = MOGG::MojoValue(std::move(type));

  MOGG::MojoValue name =
      state.builder.comptime(state.builder.listLiteral(mojoValues),
                             /*type=*/&mojoType);
  return name;
}

LogicalResult BufferPlanOp::emitMojo(MOGG::EmitMojoState &state) {
  using namespace MOGG;

  KGEN::LIT::FnOp litFunction =
      state.analysis.getRegisteredLITMojoKernelFor(getOperationName());
  MojoValue function = state.getOriginalMojoNameFor(litFunction);

  // Calculate sizes: total allocations = staticSizes + runtimeSizes
  size_t numStaticSizes = getStaticSizes() ? getStaticSizes()->size() : 0;
  size_t numRuntimeSizes = getRuntimeSizes().size();

  MojoValue numStaticSizesParam = numStaticSizes;
  MojoValue numRuntimeSizesParam = numRuntimeSizes;

  // Emit compile-time InlineArray parameters for all lifetime arrays.
  MojoValue alignmentsParam = inlineArray(state, getAlignments());
  MojoValue minPreParam = inlineArray(state, getMinPre());
  MojoValue minPostParam = inlineArray(state, getMinPost());
  MojoValue maxPreParam = inlineArray(state, getMaxPre());
  MojoValue maxPostParam = inlineArray(state, getMaxPost());

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
    stream << "var " << name << ": " << HardcodedMojoSymbols::struct_InlineArray
           << "[Int, " << numRuntimeSizes << "] = [";
    llvm::interleaveComma(runtimeSizeArgs, stream,
                          [&](const MojoValue &v) { stream << v; });
    stream << "]";
    state.builder.verbatim(MojoValue(str));
    runtimeSizesArray = name;
  }

  // Call the function with compile-time parameters and runtime arguments.
  // Function signature: [num_static_sizes, num_runtime_sizes, alignments,
  // minPre, minPost, maxPre, maxPost, staticSizes](runtimeSizes)
  MojoValue result =
      state.builder.var(function({alignmentsParam, minPreParam, minPostParam,
                                  maxPreParam, maxPostParam, staticSizesParam},
                                 {runtimeSizesArray}));

  // Extract results: result is a tuple (highWatermark, offsets)
  state.setMojoValueFor(getHighWatermark(), result[0]);

  // Extract offsets from result[1] which is a StaticTuple
  for (auto &&[i, offset] : llvm::enumerate(getOffsets()))
    state.setMojoValueFor(offset, result[1][i]);

  return success();
}

namespace {

/// Expand and contract each lifetime bound as far as the reuse constraints
/// permit, reducing the number of distinct values.  The reuse condition is:
///   canReuse(i, j) = (maxPre[i] < minPre[j]) AND (minPost[i] > maxPost[j])
///
/// Pre-order steps (applied in order):
///   a) Expand maxPre[i] upward: upper bound is
///      min(minPre[j] : canReuse(i,j)) - 1, or the global pre-value max.
///   b) Contract minPre[i] downward: lower bound is
///      max(maxPre_new[j] : canReuse(j,i)) + 1, or the global pre-value min.
///
/// Post-order steps (direction reversed because the post condition uses >):
///   c) Contract minPost[i] downward: lower bound is
///      max(maxPost[j] : canReuse(i,j)) + 1, or the global post-value min.
///   d) Expand maxPost[i] upward: upper bound is
///      min(minPost_new[j] : canReuse(j,i)) - 1, or the global post-value max.
///
/// After this pass, allocations with no reuse relationship collapse to the same
/// global min/max values, while strictly-ordered pairs keep their relative
/// ordering.
struct ExpandContractBufferPlanLifetimes final
    : public OpRewritePattern<BufferPlanOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(BufferPlanOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getNumOperands() == 0)
      return failure();

    ArrayRef<uint64_t> minPre = op.getMinPre();
    ArrayRef<uint64_t> maxPre = op.getMaxPre();
    ArrayRef<uint64_t> minPost = op.getMinPost();
    ArrayRef<uint64_t> maxPost = op.getMaxPost();
    unsigned n = minPre.size();

    auto canReuse = [&](unsigned i, unsigned j) {
      return i != j && maxPre[i] < minPre[j] && minPost[i] > maxPost[j];
    };

    uint64_t globalMinPre =
        std::min(*llvm::min_element(minPre), *llvm::min_element(maxPre));
    uint64_t globalMaxPre =
        std::max(*llvm::max_element(minPre), *llvm::max_element(maxPre));
    uint64_t globalMinPost =
        std::min(*llvm::min_element(minPost), *llvm::min_element(maxPost));
    uint64_t globalMaxPost =
        std::max(*llvm::max_element(minPost), *llvm::max_element(maxPost));

    // Expand arr[i] upward toward globalMax, stopped by fence(j) for each j
    // where pred(i, j) is true.
    auto expand = [&](MutableArrayRef<uint64_t> arr, uint64_t globalMax,
                      auto pred, auto fence) {
      for (unsigned i : llvm::seq(n)) {
        uint64_t upper = globalMax;
        for (unsigned j : llvm::seq(n))
          if (pred(i, j))
            upper = std::min(upper, fence(j));
        arr[i] = upper;
      }
    };

    // Contract arr[i] downward toward globalMin, floored by floor(j) for each
    // j where pred(j, i) is true.
    auto contract = [&](MutableArrayRef<uint64_t> arr, uint64_t globalMin,
                        auto pred, auto floor) {
      for (unsigned i : llvm::seq(n)) {
        uint64_t lower = globalMin;
        for (unsigned j : llvm::seq(n))
          if (pred(j, i))
            lower = std::max(lower, floor(j));
        arr[i] = lower;
      }
    };

    // (a) canReuse(i, j) → maxPre[i] < minPre[j], so expand calls pred(i, j).
    auto newMaxPre = llvm::to_vector(maxPre);
    expand(newMaxPre, globalMaxPre, canReuse,
           [&](unsigned j) { return minPre[j] - 1; });

    // (b) canReuse(j, i) → maxPre[j] < minPre[i], so contract calls pred(j, i).
    auto newMinPre = llvm::to_vector(minPre);
    contract(newMinPre, globalMinPre, canReuse,
             [&](unsigned j) { return newMaxPre[j] + 1; });

    // (c) canReuse(i, j) → minPost[i] > maxPost[j], so j constrains i when
    // canReuse(i, j).  contract calls pred(j, i), so flip.
    auto newMinPost = llvm::to_vector(minPost);
    contract(
        newMinPost, globalMinPost,
        [&](unsigned a, unsigned b) { return canReuse(b, a); },
        [&](unsigned j) { return maxPost[j] + 1; });

    // (d) canReuse(j, i) → minPost[j] > maxPost[i], so j constrains i when
    // canReuse(j, i).  expand calls pred(i, j), so flip.
    auto newMaxPost = llvm::to_vector(maxPost);
    expand(
        newMaxPost, globalMaxPost,
        [&](unsigned a, unsigned b) { return canReuse(b, a); },
        [&](unsigned j) { return newMinPost[j] - 1; });

    if (ArrayRef<uint64_t>(newMinPre) == minPre &&
        ArrayRef<uint64_t>(newMaxPre) == maxPre &&
        ArrayRef<uint64_t>(newMinPost) == minPost &&
        ArrayRef<uint64_t>(newMaxPost) == maxPost)
      return failure();

    rewriter.replaceOpWithNewOp<BufferPlanOp>(
        op, op.getRuntimeSizes(), op.getAlignments(), newMinPre, newMinPost,
        newMaxPre, newMaxPost, op.getStaticSizesAttr());
    return success();
  }
};

/// Replace each lifetime value with its rank in the joint sorted set of all
/// pre values (minPre ∪ maxPre), and similarly for post values.  This
/// eliminates differences in absolute DFS index values while preserving all
/// strict-less-than relationships required by the reuse condition.
struct NormalizeBufferPlanLifetimeRanks final
    : public OpRewritePattern<BufferPlanOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(BufferPlanOp op,
                                PatternRewriter &rewriter) const override {
    ArrayRef<uint64_t> minPre = op.getMinPre();
    ArrayRef<uint64_t> maxPre = op.getMaxPre();
    ArrayRef<uint64_t> minPost = op.getMinPost();
    ArrayRef<uint64_t> maxPost = op.getMaxPost();

    auto computeRanks = [](ArrayRef<uint64_t> a, ArrayRef<uint64_t> b,
                           SmallVectorImpl<uint64_t> &ra,
                           SmallVectorImpl<uint64_t> &rb) {
      SmallVector<uint64_t> all(a.begin(), a.end());
      all.append(b.begin(), b.end());

      llvm::sort(all);
      all.erase(llvm::unique(all), all.end());

      auto rank = [&](uint64_t v) -> uint64_t {
        return llvm::lower_bound(all, v) - all.begin();
      };

      ra = llvm::map_to_vector(a, rank);
      rb = llvm::map_to_vector(b, rank);
    };

    SmallVector<uint64_t> cMinPre, cMaxPre, cMinPost, cMaxPost;
    computeRanks(minPre, maxPre, cMinPre, cMaxPre);
    computeRanks(minPost, maxPost, cMinPost, cMaxPost);

    if (ArrayRef<uint64_t>(cMinPre) == minPre &&
        ArrayRef<uint64_t>(cMaxPre) == maxPre &&
        ArrayRef<uint64_t>(cMinPost) == minPost &&
        ArrayRef<uint64_t>(cMaxPost) == maxPost)
      return failure();

    rewriter.replaceOpWithNewOp<BufferPlanOp>(
        op, op.getRuntimeSizes(), op.getAlignments(), cMinPre, cMinPost,
        cMaxPre, cMaxPost, op.getStaticSizesAttr());
    return success();
  }
};

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

    // Build new staticSizes attribute
    SmallVector<int64_t> newStaticSizes =
        llvm::to_vector(op.getStaticSizes().value_or(ArrayRef<int64_t>{}));

    newStaticSizes.append(staticValues.begin(), staticValues.end());

    auto newStaticSizesAttr = rewriter.getAttr<IntArrayElementsAttr>(
        ArrayRef<int64_t>(newStaticSizes), IntegerType::Signed);

    // Create new operation with permuted attributes
    auto newOp = BufferPlanOp::create(
        rewriter, op.getLoc(),
        /*runtimeSizes=*/newRuntimeSizes,
        /*alignments=*/permuteAttr(op.getAlignments()),
        /*minPre=*/permuteAttr(op.getMinPre()),
        /*minPost=*/permuteAttr(op.getMinPost()),
        /*maxPre=*/permuteAttr(op.getMaxPre()),
        /*maxPost*/ permuteAttr(op.getMaxPost()), newStaticSizesAttr);

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
  patterns.add<ExpandContractBufferPlanLifetimes,
               NormalizeBufferPlanLifetimeRanks, StaticizeAllocSizes>(ctx);
}

LogicalResult BufferPlanOp::verify() {
  size_t numStaticSizes = getStaticSizes() ? getStaticSizes()->size() : 0;
  size_t numRuntimeSizes = getRuntimeSizes().size();
  size_t totalSizes = numStaticSizes + numRuntimeSizes;

  // The total number of offsets should equal staticSizes + runtimeSizes
  if (getOffsets().size() != totalSizes) {
    return emitOpError("expected number of offsets (")
           << getOffsets().size() << ") to equal staticSizes size ("
           << numStaticSizes << ") + runtimeSizes size (" << numRuntimeSizes
           << ") = " << totalSizes;
  }

  // All lifetime and alignment attributes should match the total number of
  // sizes
  if (!llvm::all_equal({totalSizes, getAlignments().size(), getMinPre().size(),
                        getMinPost().size(), getMaxPre().size(),
                        getMaxPost().size()})) {
    return emitOpError("expected alignments and lifetime info attributes to "
                       "have size equal to total number of sizes (")
           << totalSizes << ")";
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
  MOGG::MojoValue labelSizeVar = labelString.get("byte_length")();

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

  KGEN::LIT::FnOp litFunction =
      state.analysis.getRegisteredLITMojoKernelFor(getOperationName());
  MojoValue function = state.getOriginalMojoNameFor(litFunction);

  state.builder.call(function, {},
                     state.getMojoValuesFor(getOperands().drop_front()));

  return success();
}
