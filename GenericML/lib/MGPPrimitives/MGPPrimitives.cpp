//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/MGPPrimitives/MGPPrimitives.h"
#include "GenericML/MOPrimitives/MOPrimitives.h"
#include "GenericML/Support/Context.h"
#include "GenericML/Support/HostDeviceSpecs.h"
#include "GenericML/Support/StateSlot.h"
#include "GenericML/Support/Tensor.h"
#include "GraphRT/MEFExecutor/PrimitiveRegistry.h"
#include "Kernels/Misc.h"
#include "LLCL/Runtime/Algorithms.h"
#include "LLCL/Runtime/AsyncValue.h"
#include "LLCL/Runtime/AsyncValueRef.h"
#include "LLCL/Support/SpinWaiter.h"
#include "Support/DeviceSpecs.h"
#include "Support/MArchTarget/MArchTarget.h"
#include "Support/ML/SizeUtils.h"
#include "Support/ML/TensorShape.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"

#include "./CASStateSlot.h"
#include "./CompiledKernelsContext.h"

#define DEBUG_TYPE "mgp"

using namespace M;
using namespace M::GML;

using MGPProfilerEntry = LLCL::WorkProfilerEntry;

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

static std::optional<size_t> getRuntimeSize(uint64_t rawSize) {
  return hasRawSize(rawSize)
             ? std::optional<size_t>(rawSizeToRuntimeSize(rawSize))
             : std::nullopt;
}

static std::optional<size_t> getRuntimeAlign(uint64_t rawAlign) {
  return getRuntimeSize(rawAlign);
}

//===----------------------------------------------------------------------===//
// mgp.cpu.buffer.constant
//===----------------------------------------------------------------------===//

/// Fulfills the result with the borrowed BufferRef whose data is backed by
/// the given constant.
static void mgp_cpu_buffer_constant(Result<BufferRef> outBuffer,
                                    MEFAttr<ArrayRef<uint8_t>> aValue,
                                    MEFAttr<uint64_t> bRawAlign,
                                    Runtime &runtime) {
  TimeTraceScope scope(MGPProfilerEntry::create(
      "mgp.cpu.buffer.constant", [aValue, bRawAlign]() {
        return ("size:" + Twine(aValue.size()) +
                "B, align:" + Twine(*bRawAlign) + "B")
            .str();
      }));
  std::optional<size_t> optAlign;
  if (hasRawSize(*bRawAlign))
    optAlign = rawSizeToRuntimeSize(*bRawAlign);
  std::move(outBuffer).emplace(GML::BufferRef::createBorrowed(
      aValue->data(), aValue->size(), optAlign, BufferRef::kConstant));
}

//===----------------------------------------------------------------------===//
// mgp.cpu.buffer.cache
//===----------------------------------------------------------------------===//

static void mgp_cpu_buffer_cache(Chain inChain, Context &ctx,
                                 AsyncValueRef<BufferRef> buffer,
                                 Result<Chain> outChain, MEFAttr<uint64_t> slot,
                                 Runtime &runtime, EncodedLocation loc) {
  TimeTraceScope scope(MGPProfilerEntry::create("mgp.cpu.buffer.cache"));
  ctx.emplaceStateSlot<DefaultStateSlot>(*slot, /*numBuffers=*/1);
  auto &defaultStateSlot = ctx.getStateSlot(*slot);
  ErrorOrSuccess errOr;

  if (buffer.getPointer()->isUnique()) {
    // Since this is the only remaining reference to the buffer, so we can just
    // move it into the cache slot.
    errOr = defaultStateSlot.setBuffer(
        /*bufferIndex=*/0, std::move(buffer.get()));
  } else if (buffer.get().getMutability() == BufferRef::kImmutable) {
    // Since the buffer is immutable we can just borrow it. It must have at
    // least 'execute' lifetime.
    errOr = defaultStateSlot.setBuffer(
        /*bufferIndex=*/0, buffer.get().borrow(BufferRef::kExecute));
  } else {
    // Since the buffer is mutable we must protect against mutations from
    // other references by taking a deep copy.
    errOr = defaultStateSlot.setBuffer(
        /*bufferIndex*/ 0, buffer.get().deepCopy(runtime));
  }

  if (errOr.isError())
    return std::move(outChain).setToError({errOr.takeError(), std::move(loc)});
  std::move(outChain).emplace();
}

//===----------------------------------------------------------------------===//
// mgp.cpu.buffer.get_cached
//===----------------------------------------------------------------------===//

static void mgp_cpu_buffer_get_cached(Chain inChain, Context &ctx,
                                      Result<BufferRef> outBuffer,
                                      MEFAttr<uint64_t> aSlot,
                                      MEFAttr<uint64_t> bRawSize,
                                      Runtime &runtime, EncodedLocation loc) {
  TimeTraceScope scope(MGPProfilerEntry::create("mgp.cpu.buffer.get_cached"));
  auto &stateSlot = ctx.getStateSlot(*aSlot);
  ErrorOr<BufferRef> errOr = stateSlot.getBuffer(/*bufferIndex=*/0);
  if (errOr.isError())
    return std::move(outBuffer).setToError({errOr.takeError(), std::move(loc)});
  std::optional<size_t> expectedSize = getRuntimeSize(*bRawSize);
  if (expectedSize && errOr->getSize() != *expectedSize)
    return std::move(outBuffer).setToError(
        {Error(Twine("buffer cached at slot ") + Twine(*aSlot) +
               " has different size than expected (" + Twine(errOr->getSize()) +
               " vs " + Twine(*expectedSize) + ")"),
         std::move(loc)});
  std::move(outBuffer).emplace(std::move(*errOr));
}

//===----------------------------------------------------------------------===//
// mgp.cpu.buffer.alloc.*
//===----------------------------------------------------------------------===//

/// Returns in result a freshly allocated BufferRef. The size can be provided
/// statically, via an runtime size argument, or via a runtime TensorSpec.
/// Any alignment attribute is respected.
static void mgp_cpu_buffer_alloc_static(Chain inChain, Result<BufferRef> result,
                                        MEFAttr<uint64_t> aSize,
                                        MEFAttr<uint64_t> bRawAlign,
                                        Runtime &runtime) {
  TimeTraceScope scope(MGPProfilerEntry::create(
      "mgp.cpu.buffer.alloc.static", [aSize, bRawAlign]() {
        return ("size:" + Twine(*aSize) + "B, align:" + Twine(*bRawAlign) + "B")
            .str();
      }));
  if (!isValidPlainSize(*aSize))
    return std::move(result).setToError("invalid size attribute");
  if (!isValidRawAlign(*bRawAlign))
    return std::move(result).setToError("invalid align attribute");
  size_t runtimeSize = plainSizeToRuntimeSize(*aSize);
  std::move(result).emplace(BufferRef::createOwned(
      runtime, runtimeSize, getRuntimeAlign(*bRawAlign)));
}

static void mgp_cpu_buffer_alloc_dynamic(Chain inChain, size_t runtimeSize,
                                         Result<BufferRef> result,
                                         MEFAttr<uint64_t> rawAlign,
                                         Runtime &runtime) {
  TimeTraceScope scope(MGPProfilerEntry::create(
      "mgp.cpu.buffer.alloc.dynamic", [runtimeSize, rawAlign]() {
        return ("size:" + Twine(runtimeSize) + "B, align:" + Twine(*rawAlign) +
                "B")
            .str();
      }));
  if (!isValidRawAlign(*rawAlign))
    return std::move(result).setToError("invalid align attribute");
  std::move(result).emplace(
      BufferRef::createOwned(runtime, runtimeSize, getRuntimeAlign(*rawAlign)));
}

static void mgp_cpu_buffer_alloc_spec(Chain inChain, const TensorSpec &spec,
                                      Result<BufferRef> result,
                                      MEFAttr<uint64_t> rawAlign,
                                      Runtime &runtime) {
  TimeTraceScope scope(MGPProfilerEntry::create("mgp.cpu.buffer.alloc.spec"));
  if (!isValidRawAlign(*rawAlign))
    return std::move(result).setToError("invalid align attribute");
  std::move(result).emplace(BufferRef::createOwned(
      runtime, spec.getSizeInBytes(), getRuntimeAlign(*rawAlign)));
}

//===----------------------------------------------------------------------===//
// mgp.cpu.buffer.realloc.*
//===----------------------------------------------------------------------===//

/// Returns a mutable borrow BufferRef of a cached buffer. The buffer is cached
/// at the buffer index given by bBuffer in the state slot index given by aSlot
/// in ctx. The size can be provided statically, via a runtime size argument, or
/// via a runtime TensorSpec. If the cache is empty a fresh buffer is allocated
/// and cached. If the cache has a buffer of different size then it is freed
/// and a fresh buffer is allocated. Otherwise the existing cached buffer is
/// reused. The result BufferRef will borrow from the cached buffer.
///
/// Any alignment attribute is respected on allocation, however alignment
/// is ignored when considering whether an existing cached buffer matches the
/// requested size.
static void mgp_cpu_buffer_realloc_static(
    Chain inChain, Context &ctx, Result<BufferRef> result,
    MEFAttr<uint64_t> aSlot, MEFAttr<uint64_t> bBuffer, MEFAttr<uint64_t> cSize,
    MEFAttr<uint64_t> dRawAlign, Runtime &runtime) {
  TimeTraceScope scope(
      MGPProfilerEntry::create("mgp.cpu.buffer.realloc.static"));
  if (!isValidPlainSize(*cSize))
    return std::move(result).setToError("invalid size attribute");
  if (!isValidRawAlign(*dRawAlign))
    return std::move(result).setToError("invalid align attribute");
  size_t runtimeSize = plainSizeToRuntimeSize(*cSize);
  StateSlot &stateSlot = ctx.getStateSlot(*aSlot);
  std::move(result).emplace(stateSlot.getOrCreateBuffer(
      *bBuffer, runtime, runtimeSize, getRuntimeAlign(*dRawAlign)));
}

static void mgp_cpu_buffer_realloc_dynamic(
    Chain inChain, Context &ctx, size_t runtimeSize, Result<BufferRef> result,
    MEFAttr<uint64_t> aSlot, MEFAttr<uint64_t> bBuffer,
    MEFAttr<uint64_t> cRawAlign, Runtime &runtime) {
  TimeTraceScope scope(
      MGPProfilerEntry::create("mgp.cpu.buffer.realloc.dynamic"));
  if (!isValidRawAlign(*cRawAlign))
    return std::move(result).setToError("invalid align attribute");
  StateSlot &stateSlot = ctx.getStateSlot(*aSlot);
  std::move(result).emplace(stateSlot.getOrCreateBuffer(
      *bBuffer, runtime, runtimeSize, getRuntimeAlign(*cRawAlign)));
}

static void
mgp_cpu_buffer_realloc_spec(Chain inChain, Context &ctx, const TensorSpec &spec,
                            Result<BufferRef> result, MEFAttr<uint64_t> aSlot,
                            MEFAttr<uint64_t> bBuffer,
                            MEFAttr<uint64_t> cRawAlign, Runtime &runtime) {
  TimeTraceScope scope(MGPProfilerEntry::create("mgp.cpu.buffer.realloc.spec"));
  if (!isValidRawAlign(*cRawAlign))
    return std::move(result).setToError("invalid align attribute");
  StateSlot &stateSlot = ctx.getStateSlot(*aSlot);
  std::move(result).emplace(stateSlot.getOrCreateBuffer(
      *bBuffer, runtime, spec.getSizeInBytes(), getRuntimeAlign(*cRawAlign)));
}

//===----------------------------------------------------------------------===//
// mgp.cpu.buffer.set_with_index
//===----------------------------------------------------------------------===//

/// Helper to fill the buffer with the given index values.
template <typename T>
static void fillBuffer(BufferRef &buffer, RemainingArguments idxValues) {
  T *target = buffer.getMutableBuffer<T>();
  for (const auto &asyncIdx : idxValues)
    *(target++) = static_cast<T>(asyncIdx.get<size_t>());
}

static ErrorOr<Chain>
mgp_cpu_buffer_set_with_index(BufferRef &buffer, RemainingArguments idxValues) {
  size_t bufSize = buffer.getSize();
  size_t numArgs = idxValues.size();

  if (bufSize % numArgs != 0) {
    return Error("buffer size " + Twine(bufSize) +
                 " not divisible by number of index arguments ( " +
                 Twine(numArgs) + " )");
  }

  size_t elSize = bufSize / numArgs;
  switch (elSize) {
  case 4:
    fillBuffer<int32_t>(buffer, idxValues);
    break;
  case 8:
    fillBuffer<int64_t>(buffer, idxValues);
    break;
  default:
    return Error("unsupported element size for index buffer: " + Twine(elSize));
  }
  return Chain();
}

//===----------------------------------------------------------------------===//
// mgp.cpu.buffer.copy
//===----------------------------------------------------------------------===//

static void mgp_cpu_buffer_copy(Context &ctx, AsyncValueRef<BufferRef> inBuffer,
                                Result<BufferRef> outBuffer, Runtime &runtime) {
  TimeTraceScope scope(MGPProfilerEntry::create("mgp.cpu.buffer.copy"));
  std::move(outBuffer).emplace(inBuffer->deepCopy(runtime));
}

//===----------------------------------------------------------------------===//
// mgp.cpu.buffer.concat
//===----------------------------------------------------------------------===//

static void mgp_cpu_buffer_concat(Chain inChain,
                                  AsyncValueRef<BufferRef> outBuffer,
                                  RemainingArguments inBuffers,
                                  Result<Chain> outChain, Runtime &runtime) {
  TimeTraceScope scope(MGPProfilerEntry::create("mgp.cpu.buffer.concat"));
  uint8_t *const out = outBuffer->getMutableBuffer<uint8_t>();
  const size_t outSize = outBuffer->getSize();
  size_t totalSize = 0; // total size of all input buffers processed so far
  for (const auto &inBuffer : inBuffers) {
    const BufferRef &ref = inBuffer.get<BufferRef>();
    const size_t thisSize = ref.getSize();
    if (totalSize + thisSize <= outSize)
      // Just stop copying if it turns out the sizes don't line up.
      // We'll report failure below.
      memcpy(out + totalSize, ref.getBuffer(), thisSize);
    totalSize += thisSize;
  }
  if (totalSize != outBuffer->getSize())
    return std::move(outChain).setToError(
        Twine("output buffer size must be sum of input buffer sizes (") +
        Twine(outBuffer->getSize()) + " vs " + Twine(totalSize));
  std::move(outChain).emplace();
}

//===----------------------------------------------------------------------===//
// mgp.cpu.buffer.section.borrow.static
//===----------------------------------------------------------------------===//

static void mgp_cpu_buffer_section_borrow_static(
    Chain inChain, AsyncValueRef<BufferRef> inBuffer,
    Result<BufferRef> outBuffer, MEFAttr<uint64_t> aOffset,
    MEFAttr<uint64_t> bSize, Runtime &runtime, EncodedLocation loc) {
  TimeTraceScope scope(
      MGPProfilerEntry::create("mgp.cpu.buffer.section.borrow.static"));
  const size_t outOffset = static_cast<size_t>(*aOffset);
  const size_t outSize = static_cast<size_t>(*bSize);

  // createSafeBorrowedSlice will assert the offset and size are valid.
  // This is guaranteed by op verify and type system, so no need to guard
  // with error checking.
  std::move(outBuffer).emplace(BufferRef::createSafeBorrowedSlice(
      inBuffer.copy(), loc, outOffset, outSize));
}

//===----------------------------------------------------------------------===//
// mgp.cpu.buffer.section.borrow.dynamic
//===----------------------------------------------------------------------===//

static void mgp_cpu_buffer_section_borrow_dynamic(
    Chain inChain, AsyncValueRef<BufferRef> inBuffer, size_t scale,
    Result<BufferRef> outBuffer, MEFAttr<uint64_t> aUnscaledOffset,
    MEFAttr<uint64_t> bUnscaledSize, Runtime &runtime, EncodedLocation loc) {
  TimeTraceScope scope(
      MGPProfilerEntry::create("mgp.cpu.buffer.section.borrow.dynamic"));
  const size_t outOffset = static_cast<size_t>(*aUnscaledOffset) * scale;
  const size_t outSize = static_cast<size_t>(*bUnscaledSize) * scale;
  const size_t outEndOffset = outOffset + outSize;

  // Unlike for mgp_cpu_buffer_section_borrow_static we're on the hook for
  // checking the offset and size are sensible w.r.t. the input buffer.
  // Note we'll not check for scale being sensible, but will watch for overflow
  // of the offsets.
  if (outEndOffset < outOffset || outEndOffset > inBuffer->getSize()) {
    return std::move(outBuffer).setToError(
        Twine("mgp.cpu.buffer.section.borrow.dynamic is "
              "requesting a section of ") +
        Twine(outSize) + " bytes, starting at offset " + Twine(outOffset) +
        ", which is outside the range of the input buffer of " +
        Twine(inBuffer->getSize()) + " bytes");
  }

  std::move(outBuffer).emplace(BufferRef::createSafeBorrowedSlice(
      inBuffer.copy(), loc, outOffset, outSize));
}

//===----------------------------------------------------------------------===//
// mgp.cpu.buffer.split.borrow
//===----------------------------------------------------------------------===//

static void
mgp_cpu_buffer_split_borrow(Chain inChain, AsyncValueRef<BufferRef> inBuffer,
                            RemainingArguments runtimeSizes,
                            RemainingResults outBuffers,
                            MEFAttr<ArrayRef<uint64_t>> outBufferRawSizes,
                            Runtime &runtime, EncodedLocation loc) {
  TimeTraceScope scope(MGPProfilerEntry::create("mgp.cpu.buffer.split.borrow"));
  const size_t inSize = inBuffer->getSize();
  size_t totalSize = 0; // total size of all output buffers processed so far
  size_t runtimeSizeIndex = 0; // index of next runtimeSizes operand to use
  SmallVector<BufferRef> newBuffers;
  // We'll have an entry in outBufferRawSizes for each output buffer, which
  // will be either the output buffer's compile time size or the distinguished
  // 'no value' value.
  newBuffers.reserve(outBufferRawSizes.size());
  for (uint64_t outBufferRawSize : *outBufferRawSizes) {
    size_t outBufferRuntimeSize = rawSizeToRuntimeSize(outBufferRawSize);
    size_t thisSize;
    if (hasRuntimeSize(outBufferRuntimeSize)) {
      // The size of this output buffer is known at compile time.
      thisSize = outBufferRuntimeSize;
    } else {
      // The size of this output buffer in unknown at compile time, so
      // must be taken from a runtimeSize operand value.
      assert(runtimeSizeIndex < runtimeSizes.size() &&
             "mismatched runtimeSizes arity");
      thisSize = runtimeSizes[runtimeSizeIndex].get<size_t>();
      runtimeSizeIndex++;
    }

    // Just stop creating results if it turns out the sizes don't line up.
    // We'll report failure below.
    // We'll create a 'safe' borrow from inBuffer, with the same mutability.
    // Each borrow will take a reference to inBuffer, thus guaranteeing its
    // lifetime will exceed each borrow.
    if (totalSize + thisSize <= inSize)
      newBuffers.emplace_back(BufferRef::createSafeBorrowedSlice(
          inBuffer.copy(), loc, totalSize, thisSize));
    totalSize += thisSize;
  }

  if (totalSize != inBuffer->getSize()) {
    return std::move(outBuffers)
        .setAllToError(Twine("mgp.cpu.buffer.split.borrow input buffer "
                             "size must be sum of output buffer sizes (") +
                           Twine(inBuffer->getSize()) + " vs " +
                           Twine(totalSize) + ")",
                       loc);
  }

  for (auto [outBuffer, newBuffer] : llvm::zip(outBuffers.get(), newBuffers))
    outBuffer =
        AnyAsyncValueRef::createReady<BufferRef>(runtime, std::move(newBuffer));
}

//===----------------------------------------------------------------------===//
// mgp.cpu.buffer.set
//===----------------------------------------------------------------------===//

/// Set each byte of the given inBuffer with the given Value, returns chain.
static Chain mgp_cpu_buffer_set(BufferRef &buffer, MEFAttr<uint8_t> value) {
  TimeTraceScope scope(MGPProfilerEntry::create("mgp.cpu.buffer.set"));
  memset(buffer.getMutableBuffer(), *value, buffer.getSize());
  return {};
}

//===----------------------------------------------------------------------===//
// mgp.context.create
//===----------------------------------------------------------------------===//

/// Fulfills the result with a Context initialized with `numPrimContexts`
/// primitive contexts.
static void mgp_context_create(Result<Context> result,
                               MEFAttr<uint64_t> numSlots) {
  TimeTraceScope scope(MGPProfilerEntry::create("mgp.context.create"));
  std::move(result).emplace(*numSlots);
}

//===----------------------------------------------------------------------===//
// mgp.state_slot.create
//===----------------------------------------------------------------------===//

/// Constructs the default state slot with the given number of cached buffers
/// under the given slot in the mgp context.
static Chain mgp_state_slot_create(Chain inChain, Context &ctx,
                                   MEFAttr<uint64_t> numBuffers,
                                   MEFAttr<uint64_t> slot) {
  TimeTraceScope scope(MGPProfilerEntry::create("mgp.state_slot.create"));
  ctx.emplaceStateSlot<DefaultStateSlot>(*slot, *numBuffers);
  return {};
}

//===----------------------------------------------------------------------===//
// mgp.create_runtime
//===----------------------------------------------------------------------===//

static void mgp_check_runtime(Chain inChain, Context &ctx,
                              Result<Chain> outChain, MEFAttr<uint64_t> aSlot,
                              MEFAttr<StringRef> bDevicesJSON,
                              Runtime &runtime) {
  TimeTraceScope scope(MGPProfilerEntry::create("mgp.check_runtime"));

  auto providedDeviceSpecsOr =
      runtime.createContextIfMissing<DeviceSpecCollection>(
          [&runtime]() { return GML::getHostDeviceSpecs(runtime); });
  if (providedDeviceSpecsOr)
    return std::move(outChain).setToError(providedDeviceSpecsOr.takeError());

  ErrorOr<DeviceSpecCollection> requiredDeviceSpecsOr =
      DeviceSpecCollection::deserializeFromJSON(*bDevicesJSON);
  if (requiredDeviceSpecsOr)
    return std::move(outChain).setToError(requiredDeviceSpecsOr.takeError());

  // Reconcile the model's required devices against the runtime's provided
  // devices.
  ErrorOr<DeviceSpecMap> deviceSpecMapOr =
      (*providedDeviceSpecsOr)->reconcileDeviceSpecs(*requiredDeviceSpecsOr);
  if (deviceSpecMapOr) {
    return std::move(outChain).setToError(
        Error(Twine("model cannot be executed in the current environment: ") +
              deviceSpecMapOr.getError()));
  }

  ctx.emplaceStateSlot<DeviceSpecMap>(*aSlot, /*numBuffers=*/0,
                                      std::move(*deviceSpecMapOr));
  std::move(outChain).emplace();
}

//===----------------------------------------------------------------------===//
// mgp.cpu.tensor_spec.create
//===----------------------------------------------------------------------===//

/// Creates a TensorSpec by combining the runtime and statically known
/// shape dimensions and the statically known DType. Only runtime dimensions
/// corresponding to unknown dimensions should be passed in runtimeDims.
static void mgp_cpu_tensor_spec_create(RemainingArguments runtimeDims,
                                       Result<TensorSpec> outSpec,
                                       MEFAttr<ArrayRef<uint64_t>> aRawDims,
                                       MEFAttr<uint8_t> bRawDType,
                                       Runtime &runtime) {
  TimeTraceScope scope(
      MGPProfilerEntry::create("mgp.cpu.tensor_spec.create", [aRawDims]() {
        return ("rank:" + Twine(aRawDims.size())).str();
      }));
  DType dtype(*bRawDType);
  if (dtype.isInvalid())
    return std::move(outSpec).setToError("invalid dtype");
  auto runtimeDimsVec = runtimeDims.takeAsVector();
  auto nextRuntimeDim = runtimeDimsVec.begin();
  SmallVector<uint64_t> shape;
  shape.reserve(aRawDims.size());
  for (uint64_t dim : *aRawDims) {
    if (hasRawSize(dim))
      shape.push_back(rawSizeToRuntimeSize(dim));
    else {
      size_t runtimeDim = (*nextRuntimeDim++).get<size_t>();
      shape.push_back(runtimeDim);
    }
  }
  std::move(outSpec).emplace(TensorSpec(shape, dtype));
}

//===----------------------------------------------------------------------===//
// mgp.cpu.tensor_spec.create.dynamic
//===----------------------------------------------------------------------===//

/// Creates a TensorSpec by combining the runtime and statically known
/// shape dimensions and a runtime DType. Only runtime dimensions corresponding
/// to unknown dimensions should be passed in runtimeDims.
static void mgp_cpu_tensor_spec_create_dynamic(
    DType runtimeDType, RemainingArguments runtimeDims,
    Result<TensorSpec> outSpec, MEFAttr<ArrayRef<uint64_t>> rawDims,
    Runtime &runtime) {
  TimeTraceScope scope(
      MGPProfilerEntry::create("mgp.cpu.tensor_spec.create.dynamic"));
  if (runtimeDType.isInvalid())
    return std::move(outSpec).setToError("invalid dtype");
  auto runtimeDimsVec = runtimeDims.takeAsVector();
  auto nextRuntimeDim = runtimeDimsVec.begin();
  SmallVector<uint64_t> shape;
  for (auto dim : *rawDims) {
    if (hasRawSize(dim))
      shape.push_back(rawSizeToRuntimeSize(dim));
    else {
      size_t runtimeDim = (*nextRuntimeDim++).get<size_t>();
      shape.push_back(runtimeDim);
    }
  }
  std::move(outSpec).emplace(TensorSpec(shape, runtimeDType));
}

//===----------------------------------------------------------------------===//
// mgp.cpu.tensor_spec.with_dtype
//===----------------------------------------------------------------------===//

static void mgp_cpu_tensor_spec_with_dtype(const TensorSpec &tensorSpec,
                                           Result<TensorSpec> outSpec,
                                           MEFAttr<uint8_t> rawDType,
                                           Runtime &runtime) {
  TimeTraceScope scope(
      MGPProfilerEntry::create("mgp.cpu.tensor_spec.with_dtype"));
  DType dtype(*rawDType);
  if (dtype.isInvalid())
    return std::move(outSpec).setToError("invalid dtype");
  std::move(outSpec).emplace(TensorSpec(tensorSpec.getDimsCopy(), dtype));
}

//===----------------------------------------------------------------------===//
// mgp.cpu.tensor_spec.dims.dynamic
//===----------------------------------------------------------------------===//

/// Fulfills the result with dimensions of tensor spec which were unknown
/// at compile time.
static void mgp_cpu_tensor_spec_dims_dynamic(
    const TensorSpec &tensorSpec, RemainingResults results,
    MEFAttr<ArrayRef<uint64_t>> rawDims, Runtime &runtime) {
  TimeTraceScope scope(
      MGPProfilerEntry::create("mgp.cpu.tensor_spec.dims.dynamic"));
  // All of the results are created as size_t.
  for (auto &result : results)
    result = AsyncValueRef<size_t>::allocate(runtime);

  unsigned next = 0;
  for (size_t i = 0, e = rawDims->size(); i < e; ++i)
    if (!hasRawSize((*rawDims)[i]))
      results[next++].copy().emplace<size_t>(tensorSpec[i]);
}

//===----------------------------------------------------------------------===//
// mgp.cpu.tensor_spec.size
//===----------------------------------------------------------------------===//

static void mgp_cpu_tensor_spec_size(const TensorSpec &tensorSpec,
                                     Result<size_t> result) {
  TimeTraceScope scope(MGPProfilerEntry::create("mgp.cpu.tensor_spec.size"));
  std::move(result).emplace(tensorSpec.getSizeInBytes());
}

//===----------------------------------------------------------------------===//
// mgp.cpu.tensor_spec.num_elems
//===----------------------------------------------------------------------===//

static void mgp_cpu_tensor_spec_num_elems(const TensorSpec &tensorSpec,
                                          Result<size_t> result) {
  TimeTraceScope scope(
      MGPProfilerEntry::create("mgp.cpu.tensor_spec.num_elems"));
  std::move(result).emplace(tensorSpec.getNumElements());
}

//===----------------------------------------------------------------------===//
// mgp.cpu.tensor_spec.dtype
//===----------------------------------------------------------------------===//

static void mgp_cpu_tensor_spec_dtype(const TensorSpec &tensorSpec,
                                      Result<DType> result) {
  TimeTraceScope scope(MGPProfilerEntry::create("mgp.cpu.tensor_spec.dtype"));
  std::move(result).emplace(tensorSpec.getEltType());
}

//===----------------------------------------------------------------------===//
// mgp.cpu.tensor.create
//===----------------------------------------------------------------------===//

/// Fulfills the outTensor by creating a tensor from the given tensor spec and
/// the buffer.
static void mgp_cpu_tensor_create(Chain inChain,
                                  AsyncValueRef<BufferRef> bufferRef,
                                  const TensorSpec &tensorSpec,
                                  Result<Tensor> outTensor, Runtime &runtime,
                                  EncodedLocation loc) {
  TimeTraceScope scope(MGPProfilerEntry::create("mgp.cpu.tensor.create"));
  size_t expectedSize = tensorSpec.getSizeInBytes();
  size_t actualSize = bufferRef.get().getSize();
  if (expectedSize != actualSize)
    return std::move(outTensor).setToError(
        Twine("buffer size should match spec size (") + Twine(actualSize) +
        " vs " + Twine(expectedSize) + ")");
  if (bufferRef.getPointer()->isUnique()) {
    // If this is the only reference to the input buffer then it cannot be
    // further mutated, so we can move it into the result tensor.
    LLVM_DEBUG(llvm::dbgs() << "mgp.cpu.tensor.create: Creating M::Tensor, "
                               "taking ownership of buffer\n");
    std::move(outTensor).emplace(std::move(*bufferRef), tensorSpec);
  } else {
    // Otherwise we must take a copy of the buffer to protect the result
    // tensor from side-effects.
    LLVM_DEBUG(llvm::dbgs()
               << "mgp.cpu.tensor.create: Creating M::Tensor from copy of "
                  "buffer since still has "
               << bufferRef.getPointer()->getRefCountForDebugging()
               << " users\n");
    std::move(outTensor).emplace(bufferRef->deepCopy(runtime), tensorSpec);
  }
}

//===----------------------------------------------------------------------===//
// mgp.cpu.tensor.extract
//===----------------------------------------------------------------------===//

/// fulfills the outTensorSpec and outBuffer with the corresponding tensor spec
/// and buffer from the input tensor.
static void mgp_cpu_tensor_extract(AsyncValueRef<Tensor> tensor,
                                   Result<BufferRef> outBuffer,
                                   Result<TensorSpec> outTensorSpec,
                                   Runtime &runtime) {
  TimeTraceScope scope(MGPProfilerEntry::create("mgp.cpu.tensor.extract"));
  std::move(outTensorSpec).emplace(tensor->getSpec());
  if (tensor.getPointer()->isUnique())
    // If this is the only reference to the input tensor then it cannot
    // be further observed, so we can move its buffer into the result.
    std::move(outBuffer).emplace(tensor->releaseBufferRef());
  else
    // Otherwise we must take a copy of the buffer to protect the input tensor
    // from side-effects.
    std::move(outBuffer).emplace(tensor->deepCopyBuffer(runtime));
}

//===----------------------------------------------------------------------===//
// mgp.cpu.tensor.extract.shape
//===----------------------------------------------------------------------===//

static void mgp_cpu_tensor_extract_shape(AsyncValueRef<Tensor> tensor,
                                         Result<TensorSpec> outTensorSpec,
                                         Runtime &runtime) {
  TimeTraceScope scope(
      MGPProfilerEntry::create("mgp.cpu.tensor.extract.shape"));
  std::move(outTensorSpec).emplace(tensor->getSpec().getDimsCopy(), DType());
}

//===----------------------------------------------------------------------===//
// mgp.debug.cpu.tensor.print
//===----------------------------------------------------------------------===//

static void mgp_debug_cpu_tensor_print(Chain inChain, const Tensor &tensor,
                                       Result<Chain> outChain,
                                       MEFAttr<StringRef> label,
                                       Runtime &runtime) {
  TimeTraceScope scope(MGPProfilerEntry::create("mgp.debug.cpu.tensor.print"));
  if (label->empty()) {
    llvm::outs() << tensor << "\n";
  } else {
    llvm::outs() << *label << " = ";
    if (auto errOr = Kernels::printTensor(tensor.getBuffer(), tensor.getSpec(),
                                          Kernels::ResultOutputStyle::kCompact))
      return std::move(outChain).setToError(errOr.takeError());
    llvm::outs() << "\n";
  }
  llvm::outs().flush();
  std::move(outChain).emplace();
}

//===----------------------------------------------------------------------===//
// mgp.debug.cpu.buffer.print
//===----------------------------------------------------------------------===//

static void mgp_debug_cpu_buffer_print(Chain inChain, const BufferRef &buffer,
                                       Result<Chain> outChain,
                                       MEFAttr<StringRef> label,
                                       Runtime &runtime) {
  TimeTraceScope scope(MGPProfilerEntry::create("mgp.debug.cpu.buffer.print"));
  if (label->empty()) {
    llvm::outs() << buffer << "\n";
  } else {
    llvm::outs() << *label << " = ";
    TensorSpec spec;
    if (buffer.getSize() % sizeof(uint64_t) == 0) {
      spec = TensorSpec(
          {buffer.getSize() / sizeof(uint64_t)},
          DType::getInt(sizeof(uint64_t) * 8, /*isSigned=*/false).value());
    } else if (buffer.getSize() % sizeof(uint32_t) == 0) {
      spec = TensorSpec(
          {buffer.getSize() / sizeof(uint32_t)},
          DType::getInt(sizeof(uint32_t) * 8, /*isSigned=*/false).value());
    } else {
      spec = TensorSpec({buffer.getSize()},
                        DType::getInt(8, /*isSigned=*/false).value());
    }
    if (auto errOr = Kernels::printTensor(buffer.getBuffer(), spec,
                                          Kernels::ResultOutputStyle::kCompact))
      return std::move(outChain).setToError(errOr.takeError());
  }
  llvm::outs().flush();
  std::move(outChain).emplace();
}

//===----------------------------------------------------------------------===//
// mgp.debug.bool.print
//===----------------------------------------------------------------------===//

static void mgp_debug_bool_print(Chain inChain, const bool &cond,
                                 Result<Chain> outChain, Runtime &runtime) {
  TimeTraceScope scope(MGPProfilerEntry::create("mgp.debug.bool.print"));
  llvm::outs() << cond << "\n";
  llvm::outs().flush();
  std::move(outChain).emplace();
}

//===----------------------------------------------------------------------===//
// mgp.debug.cpu.spin_wait
//===----------------------------------------------------------------------===//

static void mgp_debug_cpu_spin_wait(Chain inChain, Result<Chain> outChain,
                                    MEFAttr<uint64_t> delayUs,
                                    Runtime &runtime) {
  std::chrono::microseconds delay(*delayUs);
  auto profilerEntry =
      MGPProfilerEntry::create("mgp.debug.cpu.spin_wait", [delay]() {
        return Twine("delay:").concat(Twine(delay.count())).concat("us").str();
      });
  outChain.copy().andThenSync(
      [profilerEntry = std::move(profilerEntry)]() mutable {
        std::move(profilerEntry).record();
      });

  addTask(runtime, [delay, outChain = outChain.copy()]() mutable {
    size_t numLoops = 0;
    LLCL::BusyWaitSpinWaiter spinner(delay);
    while (!spinner.wait())
      ++numLoops;
    (void)numLoops;
    std::move(outChain).emplace();
  });
}

//===----------------------------------------------------------------------===//
// mgp.cas.create_instance
//===----------------------------------------------------------------------===//

static void mgp_cas_create_instance(Chain inChain, Context &ctx,
                                    Result<Chain> outChain,
                                    MEFAttr<uint64_t> aSlot,
                                    MEFAttr<StringRef> bCacheDir,
                                    Runtime &runtime) {
  TimeTraceScope scope(MGPProfilerEntry::create("mgp.cas.create_instance"));
  auto &stateSlot = ctx.emplaceStateSlot<CASStateSlot>(
      *aSlot, /*numBuffers=*/0, std::string(*bCacheDir), runtime);
  // The CASStateSlot must be setup.
  if (auto err = stateSlot.setup())
    return std::move(outChain).setToError(err.takeError());
  std::move(outChain).emplace();
}

//===----------------------------------------------------------------------===//
// mgp.cas.lookup
//===----------------------------------------------------------------------===//

static void mgp_cas_lookup(Chain inChain, Context &ctx,
                           Result<BufferRef> result, MEFAttr<uint64_t> aSlot,
                           MEFAttr<StringRef> bKey, MEFAttr<uint64_t> cRawSize,
                           MEFAttr<uint64_t> dRawAlign, EncodedLocation loc) {
  TimeTraceScope scope(MGPProfilerEntry::create("mgp.cas.lookup"));
  auto &stateSlot = ctx.getStateSlotPayload<CASStateSlot>(*aSlot);
  if (!stateSlot.isValid())
    return std::move(result).setToError(
        {"CAS instance has not been established", std::move(loc)});
  std::optional<size_t> expectedSize = getRuntimeSize(*cRawSize);
  std::optional<size_t> alignment = getRuntimeAlign(*dRawAlign);
  AsyncValueRef<BufferRef> lookupResult =
      stateSlot.lookup(*bKey, expectedSize, alignment, loc.copy());
  std::move(lookupResult)
      .andThenSync([expectedSize, slot = *aSlot, result = result.copy(),
                    key = *bKey, loc = std::move(loc)](
                       AsyncValueRef<BufferRef> &&lookupResult) mutable {
        if (lookupResult.isError()) {
          // We'll construct our own error so that we can report the key
          // in base64 form.
          return std::move(result).setToError(
              {"unable to lookup buffer with key '" + key +
                   "': " + lookupResult.takeDiagnostic().getMessage().get(),
               std::move(loc)});
        }
        if (expectedSize && lookupResult.get().getSize() != *expectedSize)
          return std::move(result).setToError(
              {Twine("buffer returned from CAS at slot ") + Twine(slot) +
                   " has different size than expected (" +
                   Twine(lookupResult.get().getSize()) + " vs " +
                   Twine(*expectedSize) + ")",
               std::move(loc)});
        std::move(result).emplace(std::move(lookupResult.get()));
      });
}

//===----------------------------------------------------------------------===//
// mgp.cas.store
//===----------------------------------------------------------------------===//

static AsyncValueRef<Chain>
mgp_cas_store(Chain inChain, Context &ctx, const BufferRef &bufferRef,
              MEFAttr<uint64_t> aSlot, MEFAttr<StringRef> bKey,
              EncodedLocation loc, Runtime &runtime) {
  TimeTraceScope scope(MGPProfilerEntry::create("mgp.cas.store"));
  auto &stateSlot = ctx.getStateSlotPayload<CASStateSlot>(*aSlot);
  if (!stateSlot.isValid())
    return AsyncValueRef<Chain>::createError(
        runtime, {"CAS instance has not been established", std::move(loc)});
  return stateSlot.store(*bKey, bufferRef, std::move(loc));
}

//===----------------------------------------------------------------------===//
// mgp.to_grt_bool
//===----------------------------------------------------------------------===//

static void mgp_to_grt_bool(Chain inChain, AsyncValueRef<BufferRef> bufferRef,
                            Result<bool> result, Runtime &runtime) {
  TimeTraceScope scope(MGPProfilerEntry::create("mgp.to_grt_bool"));
  std::move(result).emplace(
      *static_cast<const bool *>(bufferRef.get().getBuffer()) != 0);
}

//===----------------------------------------------------------------------===//
// mgp.assert
//===----------------------------------------------------------------------===//

static ErrorOr<Chain> mgp_assert(bool cond, MEFAttr<StringRef> message) {
  if (cond)
    return Chain();
  return Error(*message);
}

//===----------------------------------------------------------------------===//
// mgp.cpu.kernel.create_context
//===----------------------------------------------------------------------===//

static void mgp_cpu_kernel_create_context(Chain inChain, Context &ctx,
                                          Result<Chain> outChain,
                                          MEFAttr<uint64_t> slot,
                                          Runtime &runtime) {
  TimeTraceScope scope(
      MGPProfilerEntry::create("mgp.cpu.kernel.create_context"));

  // Establish an execution engine, to be shared amongst all loaded models.
  ErrorOr<KGEN::ExecutionEngine *> engineOr =
      runtime.createContextIfMissing<KGEN::ExecutionEngine>(
          []() -> ErrorOr<std::unique_ptr<KGEN::ExecutionEngine>> {
            // Initialize the host target.
            llvm::InitializeNativeTarget();
            llvm::InitializeNativeTargetAsmParser();
            llvm::InitializeNativeTargetAsmPrinter();

            // Create the execution engine.
            auto targetMachineOr = getTargetMachineForHost();
            if (targetMachineOr.isError())
              return targetMachineOr.takeError();

            return KGEN::ExecutionEngine::createWithStandardLayers(
                /*options=*/{}, **targetMachineOr);
          });

  if (engineOr.isError())
    return std::move(outChain).setToError(engineOr.takeError());

  // Establish the context for this executing model.
  ctx.emplaceStateSlot<CompiledKernelsContext>(
      /*slotIndex=*/static_cast<size_t>(*slot),
      /*numBuffers=*/0, *engineOr);

  std::move(outChain).emplace();
}

//===----------------------------------------------------------------------===//
// mgp.cpu.kernel.load_archive
//===----------------------------------------------------------------------===//

static void mgp_cpu_kernel_load_archive(Chain inChain, Context &ctx,
                                        Result<Chain> outChain,
                                        MEFAttr<uint64_t> aSlot,
                                        MEFAttr<ArrayRef<uint8_t>> bArchive,
                                        MEFAttr<StringRef> cKernelNames) {
  TimeTraceScope scope(MGPProfilerEntry::create("mgp.cpu.kernel.load_archive"));
  CompiledKernelsContext &kernelsCtx =
      ctx.getStateSlotPayload<CompiledKernelsContext>(
          static_cast<size_t>(*aSlot));

  // Split the combined kernel names into a list of individual kernel names.
  // Remember that M.multiline strings are explicitly null-terminated when
  // encoded into MEF.
  SmallVector<StringRef> kernelNames;
  StringRef combined = *cKernelNames;
  size_t start = 0;
  size_t end = combined.size();
  while (start < end) {
    size_t next = combined.find('\n', start);
    if (next == std::string::npos) {
      assert(start + 1 == end && combined[start] == '\0');
      break;
    } else {
      size_t len = next - start;
      if (len)
        kernelNames.emplace_back(combined.data() + start, len);
      start = next + 1;
    }
  }
  if (kernelNames.empty())
    return std::move(outChain).setToError(
        "expecting at least one kernel symbol to load");

  // Load the binary.
  ErrorOrSuccess errOr = kernelsCtx.load(*bArchive, kernelNames);
  if (errOr)
    return std::move(outChain).setToError(errOr.takeError());

  std::move(outChain).emplace();
}

//===----------------------------------------------------------------------===//
// mgp.cpu.kernel.execute
//===----------------------------------------------------------------------===//

static void
mgp_cpu_kernel_execute(Chain inChain, Context &ctx, RemainingArguments operands,
                       Result<Chain> outChain, RemainingResults outDimensions,
                       MEFAttr<uint64_t> aSlot, MEFAttr<uint64_t> bKernelIndex,
                       MEFAttr<StringRef> cInOutSignature,
                       MEFAttr<StringRef> dKernelName,
                       MEFAttr<uint64_t> eNumOutDimensions, Runtime &runtime,
                       EncodedLocation loc) {
  TimeTraceScope scope(MGPProfilerEntry::create("mgp.cpu.kernel.execute"));
  CompiledKernelsContext &kernelsCtx =
      ctx.getStateSlotPayload<CompiledKernelsContext>(
          static_cast<size_t>(*aSlot));
  size_t kernelIndex = static_cast<size_t>(*bKernelIndex);
  CompiledKernelFn *kernelPtr = kernelsCtx.getKernel(kernelIndex);

#if MODULAR_PARANOID
  // The kernel name is redundant since it can be retrieved using the
  // kernel index, however we retain it for debuggability and unit testing.
  // At least check it is consistent.
  assert(*dKernelName == kernelsCtx.getKernelName(kernelIndex) &&
         "kernel name is inconsistent with kernel index");
#endif

  // The operands are:
  //  [buffer_1, ..., inputBuffer_numBufferArgs,
  //   inDimension_1, ..., inDimension_numInDimension]
  StringRef inOutSignature = *cInOutSignature;
  size_t numBuffers = inOutSignature.size();
#if MODULAR_PARANOID
  assert(numBuffers <= operands.size() && "invalid buffer signature arity");
#endif
  size_t numInDimensions = operands.size() - numBuffers;

  // The outDimensions are:
  //  [outDimension_1, ..., outDimension_numOutDimensions]
  size_t numOutDimensions = static_cast<size_t>(*eNumOutDimensions);
#if MODULAR_PARANOID
  assert(numOutDimensions == outDimensions.size() &&
         "invalid out dimensions count attribute");
#endif
  for (auto &outDimension : outDimensions)
    outDimension = AsyncValueRef<size_t>::allocate(runtime);

  // Create an OutputChain to manage the kernel's asynchronous execution.
  KGEN::OutputChain outputChain(outChain.copy(), outChain.copyLocation());

  // Hold on to all the operands until the kernel completes.
  outputChain.transfer(operands.copyAsVector());

  // Put the buffers into the ABI void* array. This will mix the input
  // and output buffers.
  SmallVector<void *, 8> bufferPointers;
  bufferPointers.reserve(numBuffers);
  for (size_t i = 0; i < numBuffers; ++i) {
#if MODULAR_PARANOID
    switch (inOutSignature[i]) {
    case 'i': {
      auto &buffer = operands[i].get<GML::BufferRef>();
      // Forget the const here since the ABI mixes input and output.
      bufferPointers.push_back(const_cast<void *>(buffer.getBuffer()));
      // Record that we'll be reading this buffer while the kernel is in-flight.
      buffer.collectAccessUses(
          outputChain.uses,
          (Twine("kernel(line=") + Twine(loc.decode().line) + ")").str(),
          LLCL::kReadingResourceUse);
      break;
    }
    case 'o': {
      auto &buffer = operands[i].get<GML::BufferRef>();
      bufferPointers.push_back(buffer.getMutableBuffer());
      // Record that we'll be writing this buffers while the kernel is
      // in-flight.
      buffer.collectAccessUses(
          outputChain.uses,
          (Twine("kernel(line=") + Twine(loc.decode().line) + ")").str(),
          LLCL::kWritingResourceUse);
      break;
    }
    default:
      llvm::llvm_unreachable_internal("invalid in/out signature");
    }
#else // ! MODULAR_PARANOID
    const void *ptr = operands[i].get<GML::BufferRef>().getBuffer();
    bufferPointers.push_back(const_cast<void *>(ptr));
#endif
  }

  // Put the runtime dimensions, corresponding to each kernel 'parameter',
  // into the ABI ssize_t array. This will mix the input and output
  // dimensions.
  SmallVector<ssize_t, 16> dimensions;
  dimensions.resize(numInDimensions + numOutDimensions);
  for (size_t i = 0; i < numInDimensions; ++i)
    dimensions[i] = operands[numBuffers + i].get<size_t>();
#if MODULAR_PARANOID
  for (size_t i = 0; i < numOutDimensions; ++i)
    dimensions[numInDimensions + i] = -1;
#endif

  // Since we have the kernel name available, and since it's tedious to
  // plumb it into the compiled kernel, start the tracing now. This profiling
  // entry may be augmented by additional details on the Mojo side.
  outputChain.trace(*dKernelName);

  // Execute the kernel:
  //  - Asynchronous kernels will have copied everything out of the three
  //    pointer arguments by the time this call returns.
  //  - Synchronous kernels wil have completed upon return.
  //  - The dimensions of the bufferPointers and dimensions vectors are known
  //    at kernel compile time and do not need to be passed at runtime.
  kernelPtr(bufferPointers.data(), dimensions.data(), &outputChain);

  // If the kernel has output dimensions then it must be synchronous and those
  // values can be copied out now.
  for (size_t i = 0; i < numOutDimensions; ++i) {
    // Note we must copy rather than move the result reference since it is
    // needed by the MEF executor.
    outDimensions[i].copy().emplace<size_t>(dimensions[numInDimensions + i]);
  }

#if MODULAR_PARANOID
  // If the kernel is asynchronous but accidentally depends on the stack
  // at the time sub-tasks were launched, deliberately clobber that.
  // Pick a modest size to clobber to avoid stack overflow.
  const size_t s = 256;
  void *p = alloca(s);
  memset(p, 0xef, s);
#endif

  // If the kernel is asynchronous then it will have 'forked' the
  // output chain and launched sub-tasks to do the actual work. We can
  // thus stop profiling the 'main' kernel.
  // If the kernel is synchronous then profiling is already finished and
  // this call will be a no-op.
  std::move(outputChain).recordProfilerEntry();
}

//===----------------------------------------------------------------------===//
// mgp.model
//===----------------------------------------------------------------------===//

static void mgp_model() {
  assert(false && "mgp.model should never be invoked");
}

//===----------------------------------------------------------------------===//
// registerMGPPrimitives
//===----------------------------------------------------------------------===//

void M::registerMGPPrimitives(PrimitiveRegistry &registry) {
  ADD_PRIMITIVE(registry, "mgp.cpu.buffer.constant", mgp_cpu_buffer_constant);
  ADD_PRIMITIVE(registry, "mgp.cpu.buffer.cache", mgp_cpu_buffer_cache);
  ADD_PRIMITIVE(registry, "mgp.cpu.buffer.get_cached",
                mgp_cpu_buffer_get_cached);
  ADD_PRIMITIVE(registry, "mgp.cpu.buffer.alloc.static",
                mgp_cpu_buffer_alloc_static);
  ADD_PRIMITIVE(registry, "mgp.cpu.buffer.alloc.dynamic",
                mgp_cpu_buffer_alloc_dynamic);
  ADD_PRIMITIVE(registry, "mgp.cpu.buffer.alloc.spec",
                mgp_cpu_buffer_alloc_spec);
  ADD_PRIMITIVE(registry, "mgp.cpu.buffer.realloc.static",
                mgp_cpu_buffer_realloc_static);
  ADD_PRIMITIVE(registry, "mgp.cpu.buffer.realloc.dynamic",
                mgp_cpu_buffer_realloc_dynamic);
  ADD_PRIMITIVE(registry, "mgp.cpu.buffer.realloc.spec",
                mgp_cpu_buffer_realloc_spec);
  ADD_PRIMITIVE(registry, "mgp.cpu.buffer.set_with_index",
                mgp_cpu_buffer_set_with_index);
  ADD_PRIMITIVE(registry, "mgp.cpu.buffer.copy", mgp_cpu_buffer_copy);
  ADD_PRIMITIVE(registry, "mgp.cpu.buffer.concat", mgp_cpu_buffer_concat);
  ADD_PRIMITIVE(registry, "mgp.cpu.buffer.section.borrow.static",
                mgp_cpu_buffer_section_borrow_static);
  ADD_PRIMITIVE(registry, "mgp.cpu.buffer.section.borrow.dynamic",
                mgp_cpu_buffer_section_borrow_dynamic);
  ADD_PRIMITIVE(registry, "mgp.cpu.buffer.split.borrow",
                mgp_cpu_buffer_split_borrow);
  ADD_PRIMITIVE(registry, "mgp.cpu.buffer.set", mgp_cpu_buffer_set);
  ADD_PRIMITIVE(registry, "mgp.context.create", mgp_context_create);
  ADD_PRIMITIVE(registry, "mgp.state_slot.create", mgp_state_slot_create);
  ADD_PRIMITIVE(registry, "mgp.check_runtime", mgp_check_runtime);
  ADD_PRIMITIVE(registry, "mgp.cpu.tensor_spec.create",
                mgp_cpu_tensor_spec_create);
  ADD_PRIMITIVE(registry, "mgp.cpu.tensor_spec.create.dynamic",
                mgp_cpu_tensor_spec_create_dynamic);
  ADD_PRIMITIVE(registry, "mgp.cpu.tensor_spec.with_dtype",
                mgp_cpu_tensor_spec_with_dtype);
  ADD_PRIMITIVE(registry, "mgp.cpu.tensor_spec.dims.dynamic",
                mgp_cpu_tensor_spec_dims_dynamic);
  ADD_PRIMITIVE(registry, "mgp.cpu.tensor_spec.size", mgp_cpu_tensor_spec_size);
  ADD_PRIMITIVE(registry, "mgp.cpu.tensor_spec.num_elems",
                mgp_cpu_tensor_spec_num_elems);
  ADD_PRIMITIVE(registry, "mgp.cpu.tensor_spec.dtype",
                mgp_cpu_tensor_spec_dtype);
  ADD_PRIMITIVE(registry, "mgp.cpu.tensor.create", mgp_cpu_tensor_create);
  ADD_PRIMITIVE(registry, "mgp.cpu.tensor.extract", mgp_cpu_tensor_extract);
  ADD_PRIMITIVE(registry, "mgp.cpu.tensor.extract.shape",
                mgp_cpu_tensor_extract_shape);
  ADD_PRIMITIVE(registry, "mgp.debug.cpu.tensor.print",
                mgp_debug_cpu_tensor_print);
  ADD_PRIMITIVE(registry, "mgp.debug.cpu.buffer.print",
                mgp_debug_cpu_buffer_print);
  ADD_PRIMITIVE(registry, "mgp.debug.bool.print", mgp_debug_bool_print);
  ADD_PRIMITIVE(registry, "mgp.debug.cpu.spin_wait", mgp_debug_cpu_spin_wait);
  ADD_PRIMITIVE(registry, "mgp.cas.create_instance", mgp_cas_create_instance);
  ADD_PRIMITIVE(registry, "mgp.cas.lookup", mgp_cas_lookup);
  ADD_PRIMITIVE(registry, "mgp.cas.store", mgp_cas_store);
  ADD_PRIMITIVE(registry, "mgp.to_grt_bool", mgp_to_grt_bool);
  ADD_PRIMITIVE(registry, "mgp.assert", mgp_assert);
  ADD_PRIMITIVE(registry, "mgp.cpu.kernel.create_context",
                mgp_cpu_kernel_create_context);
  ADD_PRIMITIVE(registry, "mgp.cpu.kernel.load_archive",
                mgp_cpu_kernel_load_archive);
  ADD_PRIMITIVE(registry, "mgp.cpu.kernel.execute", mgp_cpu_kernel_execute);
  ADD_PRIMITIVE(registry, "mgp.model", mgp_model);

  // Have the MO primitives be registered wherever MGP is registered.
  // CAUTION: To be removed.
  registerMOPrimitives(registry);
}
