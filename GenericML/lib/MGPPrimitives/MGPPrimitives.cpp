//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/MGPPrimitives/MGPPrimitives.h"
#include "CASStateSlot.h"
#include "GenericML/MOPrimitives/MOPrimitives.h"
#include "GenericML/Support/Context.h"
#include "GenericML/Support/StateSlot.h"
#include "GenericML/Support/Tensor.h"
#include "GraphRT/MEFExecutor/PrimitiveRegistry.h"
#include "LLCL/Runtime/Algorithms.h"
#include "LLCL/Runtime/AsyncValue.h"
#include "LLCL/Runtime/AsyncValueRef.h"
#include "LLCL/Support/SpinWaiter.h"
#include "Support/ML/SizeUtils.h"
#include "Support/ML/TensorShape.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "mgp"

using namespace M;
using namespace M::GML;

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
  TimeTraceScope scope(
      LLCL::WorkProfilerEntry::create("mgp.cpu.buffer.constant"));
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
  TimeTraceScope scope(LLCL::WorkProfilerEntry::create("mgp.cpu.buffer.cache"));
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
  TimeTraceScope scope(
      LLCL::WorkProfilerEntry::create("mgp.cpu.buffer.get_cached"));
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
  TimeTraceScope scope(
      LLCL::WorkProfilerEntry::create("mgp.cpu.buffer.alloc.static"));
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
  TimeTraceScope scope(
      LLCL::WorkProfilerEntry::create("mgp.cpu.buffer.alloc.dynamic"));
  if (!isValidRawAlign(*rawAlign))
    return std::move(result).setToError("invalid align attribute");
  std::move(result).emplace(
      BufferRef::createOwned(runtime, runtimeSize, getRuntimeAlign(*rawAlign)));
}

static void mgp_cpu_buffer_alloc_spec(Chain inChain, const TensorSpec &spec,
                                      Result<BufferRef> result,
                                      MEFAttr<uint64_t> rawAlign,
                                      Runtime &runtime) {
  TimeTraceScope scope(
      LLCL::WorkProfilerEntry::create("mgp.cpu.buffer.alloc.spec"));
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
      LLCL::WorkProfilerEntry::create("mgp.cpu.buffer.realloc.static"));
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
      LLCL::WorkProfilerEntry::create("mgp.cpu.buffer.realloc.dynamic"));
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
  TimeTraceScope scope(
      LLCL::WorkProfilerEntry::create("mgp.cpu.buffer.realloc.spec"));
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
  TimeTraceScope scope(LLCL::WorkProfilerEntry::create("mgp.cpu.buffer.copy"));
  std::move(outBuffer).emplace(inBuffer->deepCopy(runtime));
}

//===----------------------------------------------------------------------===//
// mgp.cpu.buffer.concat
//===----------------------------------------------------------------------===//

static void mgp_cpu_buffer_concat(Chain inChain,
                                  AsyncValueRef<BufferRef> outBuffer,
                                  RemainingArguments inBuffers,
                                  Result<Chain> outChain, Runtime &runtime) {
  TimeTraceScope scope(
      LLCL::WorkProfilerEntry::create("mgp.cpu.buffer.concat"));
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
// mgp.cpu.buffer.split.borrow
//===----------------------------------------------------------------------===//

static void
mgp_cpu_buffer_split_borrow(Chain inChain, AsyncValueRef<BufferRef> inBuffer,
                            RemainingArguments runtimeSizes,
                            RemainingResults outBuffers,
                            MEFAttr<ArrayRef<uint64_t>> outBufferRawSizes,
                            Runtime &runtime, EncodedLocation loc) {
  TimeTraceScope scope(
      LLCL::WorkProfilerEntry::create("mgp.cpu.buffer.split.borrow"));
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
          inBuffer.copy(), totalSize, thisSize));
    totalSize += thisSize;
  }
  if (totalSize != inBuffer->getSize())
    return std::move(outBuffers)
        .setAllToError(Twine("mgp.cpu.buffer.split.borrow input buffer "
                             "size must be sum of output buffer sizes (") +
                           Twine(inBuffer->getSize()) + " vs " +
                           Twine(totalSize) + ")",
                       loc);
  for (auto [outBuffer, newBuffer] : llvm::zip(outBuffers.get(), newBuffers))
    outBuffer =
        AnyAsyncValueRef::createReady<BufferRef>(runtime, std::move(newBuffer));
}

//===----------------------------------------------------------------------===//
// mgp.cpu.buffer.set
//===----------------------------------------------------------------------===//

/// Set each byte of the given inBuffer with the given Value, returns chain.
static Chain mgp_cpu_buffer_set(BufferRef &buffer, MEFAttr<uint8_t> value) {
  TimeTraceScope scope(LLCL::WorkProfilerEntry::create("mgp.cpu.buffer.set"));
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
  TimeTraceScope scope(LLCL::WorkProfilerEntry::create("mgp.context.create"));
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
  TimeTraceScope scope(
      LLCL::WorkProfilerEntry::create("mgp.state_slot.create"));
  ctx.emplaceStateSlot<DefaultStateSlot>(*slot, *numBuffers);
  return {};
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
      LLCL::WorkProfilerEntry::create("mgp.cpu.tensor_spec.create"));
  DType dtype(*bRawDType);
  if (dtype.isInvalid())
    return std::move(outSpec).setToError("invalid dtype");
  auto runtimeDimsVec = runtimeDims.takeAsVector();
  auto nextRuntimeDim = runtimeDimsVec.begin();
  SmallVector<uint64_t> shape;
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
      LLCL::WorkProfilerEntry::create("mgp.cpu.tensor_spec.create.dynamic"));
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
// mgp.cpu.tensor_spec.dims.dynamic
//===----------------------------------------------------------------------===//

/// Fulfills the result with dimensions of tensor spec which were unknown
/// at compile time.
static void mgp_cpu_tensor_spec_dims_dynamic(
    const TensorSpec &tensorSpec, RemainingResults results,
    MEFAttr<ArrayRef<uint64_t>> rawDims, Runtime &runtime) {
  TimeTraceScope scope(
      LLCL::WorkProfilerEntry::create("mgp.cpu.tensor_spec.dims.dynamic"));
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
  TimeTraceScope scope(
      LLCL::WorkProfilerEntry::create("mgp.cpu.tensor_spec.size"));
  std::move(result).emplace(tensorSpec.getSizeInBytes());
}

//===----------------------------------------------------------------------===//
// mgp.cpu.tensor_spec.num_elems
//===----------------------------------------------------------------------===//

static void mgp_cpu_tensor_spec_num_elems(const TensorSpec &tensorSpec,
                                          Result<size_t> result) {
  TimeTraceScope scope(
      LLCL::WorkProfilerEntry::create("mgp.cpu.tensor_spec.num_elems"));
  std::move(result).emplace(tensorSpec.getNumElements());
}

//===----------------------------------------------------------------------===//
// mgp.cpu.tensor_spec.dtype
//===----------------------------------------------------------------------===//

static void mgp_cpu_tensor_spec_dtype(const TensorSpec &tensorSpec,
                                      Result<DType> result) {
  TimeTraceScope scope(
      LLCL::WorkProfilerEntry::create("mgp.cpu.tensor_spec.dtype"));
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
  TimeTraceScope scope(
      LLCL::WorkProfilerEntry::create("mgp.cpu.tensor.create"));
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
  TimeTraceScope scope(
      LLCL::WorkProfilerEntry::create("mgp.cpu.tensor.extract"));
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
// mgp.debug.cpu.tensor.print
//===----------------------------------------------------------------------===//

static void mgp_debug_cpu_tensor_print(Chain inChain, const Tensor &tensor,
                                       Result<Chain> outChain,
                                       Runtime &runtime) {
  TimeTraceScope scope(
      LLCL::WorkProfilerEntry::create("mgp.debug.cpu.tensor.print"));
  llvm::outs() << tensor << "\n";
  llvm::outs().flush();
  std::move(outChain).emplace();
}

//===----------------------------------------------------------------------===//
// mgp.debug.cpu.buffer.print
//===----------------------------------------------------------------------===//

static void mgp_debug_cpu_buffer_print(Chain inChain, const BufferRef &buffer,
                                       Result<Chain> outChain,
                                       Runtime &runtime) {
  TimeTraceScope scope(
      LLCL::WorkProfilerEntry::create("mgp.debug.cpu.buffer.print"));
  llvm::outs() << buffer << "\n";
  llvm::outs().flush();
  std::move(outChain).emplace();
}

//===----------------------------------------------------------------------===//
// mgp.debug.bool.print
//===----------------------------------------------------------------------===//

static void mgp_debug_bool_print(Chain inChain, const bool &cond,
                                 Result<Chain> outChain, Runtime &runtime) {
  TimeTraceScope scope(LLCL::WorkProfilerEntry::create("mgp.debug.bool.print"));
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
      LLCL::WorkProfilerEntry::create("mgp.debug.cpu.spin_wait", [delay]() {
        return Twine("delay:").concat(Twine(delay.count())).concat("us").str();
      });
  addTask(runtime, std::move(profilerEntry),
          [delay, outChain = outChain.copy()]() mutable {
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
  TimeTraceScope scope(
      LLCL::WorkProfilerEntry::create("mgp.cas.create_instance"));
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
  TimeTraceScope scope(LLCL::WorkProfilerEntry::create("mgp.cas.lookup"));
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
  TimeTraceScope scope(LLCL::WorkProfilerEntry::create("mgp.cas.store"));
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
  TimeTraceScope scope(LLCL::WorkProfilerEntry::create("mgp.to_grt_bool"));
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
// mgp.model
//===----------------------------------------------------------------------===//

static void mgp_model() {
  assert(false && "mgp.model should never be invoked");
}

//===----------------------------------------------------------------------===//
// Primitive registration
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
  ADD_PRIMITIVE(registry, "mgp.cpu.buffer.split.borrow",
                mgp_cpu_buffer_split_borrow);
  ADD_PRIMITIVE(registry, "mgp.cpu.buffer.set", mgp_cpu_buffer_set);
  ADD_PRIMITIVE(registry, "mgp.context.create", mgp_context_create);
  ADD_PRIMITIVE(registry, "mgp.state_slot.create", mgp_state_slot_create);
  ADD_PRIMITIVE(registry, "mgp.cpu.tensor_spec.create",
                mgp_cpu_tensor_spec_create);
  ADD_PRIMITIVE(registry, "mgp.cpu.tensor_spec.create.dynamic",
                mgp_cpu_tensor_spec_create_dynamic);
  ADD_PRIMITIVE(registry, "mgp.cpu.tensor_spec.dims.dynamic",
                mgp_cpu_tensor_spec_dims_dynamic);
  ADD_PRIMITIVE(registry, "mgp.cpu.tensor_spec.size", mgp_cpu_tensor_spec_size);
  ADD_PRIMITIVE(registry, "mgp.cpu.tensor_spec.num_elems",
                mgp_cpu_tensor_spec_num_elems);
  ADD_PRIMITIVE(registry, "mgp.cpu.tensor_spec.dtype",
                mgp_cpu_tensor_spec_dtype);
  ADD_PRIMITIVE(registry, "mgp.cpu.tensor.create", mgp_cpu_tensor_create);
  ADD_PRIMITIVE(registry, "mgp.cpu.tensor.extract", mgp_cpu_tensor_extract);
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
  ADD_PRIMITIVE(registry, "mgp.model", mgp_model);

  // Have the MO primitives be registered wherever MGP is registered.
  registerMOPrimitives(registry);
}
