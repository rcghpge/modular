//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/MOPPrimitives/MOPPrimitives.h"
#include "GenericML/MOP/Context.h"
#include "GenericML/MOP/StateSlot.h"
#include "GenericML/Support/Tensor.h"
#include "GraphRT/BEFExecutor/PrimitiveRegistry.h"
#include "Support/ML/SizeUtils.h"
#include "Support/ML/TensorShape.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace M;
using namespace M::MOP;
using M::LLCL::BufferRef;

//===----------------------------------------------------------------------===//
// mop.cpu.buffer.constant
//===----------------------------------------------------------------------===//

/// Fulfills the result with the borrowed BufferRef whose data is backed by
/// the given constant.
static void mop_cpu_buffer_constant(Result<BufferRef> outBuffer,
                                    BEFDynamicAttr values, Runtime &runtime) {
  // Create unowned BufferRef as the BEF outlives the BufferRef.
  // TODO(#5369): Need to capture any alignment constraint on attribute so
  // it is preserved if the result needs to be deep copied.
  if (auto buf = dyn_cast<ArrayRef<uint8_t>>(values))
    outBuffer.emplace(BufferRef::createBorrowed(
        const_cast<void *>(static_cast<const void *>((*buf)->data())),
        (*buf)->size()));
  else
    outBuffer.setToError(
        "unsupported type, only u8 is supported for attribute elements");
}

//===----------------------------------------------------------------------===//
// mop.cpu.buffer.alloc.*
//===----------------------------------------------------------------------===//

/// Returns in result a freshly allocated BufferRef. The size can be provided
/// statically, via an runtime size argument, or via a runtime TensorSpec.
/// Any alignment attribute is respected.
static void mop_cpu_buffer_alloc_static(Context &ctx, Result<BufferRef> result,
                                        BEFAttr<uint64_t> aSize,
                                        BEFAttr<uint64_t> bRawAlign,
                                        Runtime &runtime) {
  if (!isValidPlainSize(*aSize)) {
    result.setToError("invalid size attribute");
    return;
  }
  if (!isValidRawAlign(*bRawAlign)) {
    result.setToError("invalid align attribute");
    return;
  }
  size_t runtimeSize = plainSizeToRuntimeSize(*aSize);
  result.emplace(BufferRef::createOwned(
      runtime.getAllocator(), runtimeSize,
      rawSizeToRuntimeSizeOrDefault(*bRawAlign, kPreferredMemoryAlignment)));
}

static void mop_cpu_buffer_alloc_dynamic(Context &ctx, size_t runtimeSize,
                                         Result<BufferRef> result,
                                         BEFAttr<uint64_t> rawAlign,
                                         Runtime &runtime) {
  if (!isValidRuntimeSize(runtimeSize)) {
    result.setToError("invalid size operand");
    return;
  }
  if (!isValidRawAlign(*rawAlign)) {
    result.setToError("invalid align attribute");
    return;
  }
  result.emplace(BufferRef::createOwned(
      runtime.getAllocator(), runtimeSize,
      rawSizeToRuntimeSizeOrDefault(*rawAlign, kPreferredMemoryAlignment)));
}

static void mop_cpu_buffer_alloc_spec(Context &ctx, const TensorSpec &spec,
                                      Result<BufferRef> result,
                                      BEFAttr<uint64_t> rawAlign,
                                      Runtime &runtime) {
  if (!isValidRawAlign(*rawAlign)) {
    result.setToError("invalid align attribute");
    return;
  }
  result.emplace(BufferRef::createOwned(
      runtime.getAllocator(), spec.getSizeInBytes(),
      rawSizeToRuntimeSizeOrDefault(*rawAlign, kPreferredMemoryAlignment)));
}

//===----------------------------------------------------------------------===//
// mop.cpu.buffer.realloc.*
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
static void mop_cpu_buffer_realloc_static(
    Chain chain, Context &ctx, Result<BufferRef> result,
    BEFAttr<uint64_t> aSlot, BEFAttr<uint64_t> bBuffer, BEFAttr<uint64_t> cSize,
    BEFAttr<uint64_t> dRawAlign, Runtime &runtime) {
  if (!isValidPlainSize(*cSize)) {
    result.setToError("invalid size attribute");
    return;
  }
  if (!isValidRawAlign(*dRawAlign)) {
    result.setToError("invalid align attribute");
    return;
  }
  size_t runtimeSize = plainSizeToRuntimeSize(*cSize);
  StateSlot &stateSlot = ctx.getStateSlot(*aSlot);
  result.emplace(stateSlot.getOrCreateBuffer(
      *bBuffer, runtime.getAllocator(), runtimeSize,
      rawSizeToRuntimeSizeOrDefault(*dRawAlign, kPreferredMemoryAlignment)));
}

static void mop_cpu_buffer_realloc_dynamic(
    Chain chain, Context &ctx, size_t runtimeSize, Result<BufferRef> result,
    BEFAttr<uint64_t> aSlot, BEFAttr<uint64_t> bBuffer,
    BEFAttr<uint64_t> cRawAlign, Runtime &runtime) {
  if (!isValidRuntimeSize(runtimeSize)) {
    result.setToError("invalid size operand");
    return;
  }
  if (!isValidRawAlign(*cRawAlign)) {
    result.setToError("invalid align attribute");
    return;
  }
  StateSlot &stateSlot = ctx.getStateSlot(*aSlot);
  result.emplace(stateSlot.getOrCreateBuffer(
      *bBuffer, runtime.getAllocator(), runtimeSize,
      rawSizeToRuntimeSizeOrDefault(*cRawAlign, kPreferredMemoryAlignment)));
}

static void
mop_cpu_buffer_realloc_spec(Chain chain, Context &ctx, const TensorSpec &spec,
                            Result<BufferRef> result, BEFAttr<uint64_t> aSlot,
                            BEFAttr<uint64_t> bBuffer,
                            BEFAttr<uint64_t> cRawAlign, Runtime &runtime) {
  if (!isValidRawAlign(*cRawAlign)) {
    result.setToError("invalid align attribute");
    return;
  }
  StateSlot &stateSlot = ctx.getStateSlot(*aSlot);
  result.emplace(stateSlot.getOrCreateBuffer(
      *bBuffer, runtime.getAllocator(), spec.getSizeInBytes(),
      rawSizeToRuntimeSizeOrDefault(*cRawAlign, kPreferredMemoryAlignment)));
}

//===----------------------------------------------------------------------===//
// mop.cpu.buffer.copy
//===----------------------------------------------------------------------===//

/// Fulfills the result with the owning reference to the deep copy of inBuffer.
static void mop_cpu_buffer_copy(Context &ctx, AsyncValueRef<BufferRef> inBuffer,
                                Result<BufferRef> outBuffer, Runtime &runtime) {
  outBuffer.emplace(inBuffer->deepCopy(runtime.getAllocator()));
}

//===----------------------------------------------------------------------===//
// mop.cpu.buffer.set
//===----------------------------------------------------------------------===//

/// Set each byte of the given inBuffer with the given Value, returns chain.
static Chain mop_cpu_buffer_set(BufferRef &buffer, BEFAttr<uint8_t> value) {
  memset(buffer.getData(), *value, buffer.getSize());
  return {};
}

//===----------------------------------------------------------------------===//
// mop.context.create
//===----------------------------------------------------------------------===//

/// Fulfills the result with a Context initialized with `numPrimContexts`
/// primitive contexts.
static void mop_context_create(Result<Context> result,
                               BEFAttr<uint64_t> numSlots) {
  result->emplace<Context>(*numSlots);
}

//===----------------------------------------------------------------------===//
// mop.state_slot.create
//===----------------------------------------------------------------------===//

/// Constructs the default state slot with the given number of cached buffers
/// under the given slot in the mop context.
static Chain mop_state_slot_create(Context &ctx, BEFAttr<uint64_t> numBuffers,
                                   BEFAttr<uint64_t> slot) {
  ctx.emplaceStateSlot<DefaultStateSlot>(*slot, *numBuffers);
  return {};
}

//===----------------------------------------------------------------------===//
// mop.cpu.tensor_spec.create
//===----------------------------------------------------------------------===//

/// Creates a TensorSpec by combining the runtime and statically known
/// shape dimensions and the statically known DType. Only runtime dimensions
/// corresponding to unknown dimensions should be passed in runtimeDims.
static void mop_cpu_tensor_spec_create(RemainingArguments runtimeDims,
                                       Result<TensorSpec> outSpec,
                                       BEFAttr<ArrayRef<uint64_t>> aRawDims,
                                       BEFAttr<uint8_t> bRawDType,
                                       Runtime &runtime) {
  DType dtype(*bRawDType);
  if (dtype.isInvalid()) {
    outSpec.setToError("invalid dtype");
    return;
  }
  auto runtimeDimsVec = runtimeDims.takeAsVector();
  auto nextRuntimeDim = runtimeDimsVec.begin();
  SmallVector<uint64_t> shape;
  for (uint64_t dim : *aRawDims) {
    if (hasRawSize(dim))
      shape.push_back(rawSizeToRuntimeSize(dim));
    else {
      size_t runtimeDim = (*nextRuntimeDim++)->get<size_t>();
      if (!isValidRuntimeSize(runtimeDim)) {
        outSpec.setToError("invalid dimension");
        return;
      }
      shape.push_back(runtimeDim);
    }
  }
  outSpec.emplace(TensorSpec(shape, dtype));
}

//===----------------------------------------------------------------------===//
// mop.cpu.tensor_spec.create.dynamic
//===----------------------------------------------------------------------===//

/// Creates a TensorSpec by combining the runtime and statically known
/// shape dimensions and a runtime DType. Only runtime dimensions corresponding
/// to unknown dimensions should be passed in runtimeDims.
static void mop_cpu_tensor_spec_create_dynamic(
    DType runtimeDType, RemainingArguments runtimeDims,
    Result<TensorSpec> outSpec, BEFAttr<ArrayRef<uint64_t>> rawDims,
    Runtime &runtime) {
  if (runtimeDType.isInvalid()) {
    outSpec.setToError("invalid dtype");
    return;
  }
  auto runtimeDimsVec = runtimeDims.takeAsVector();
  auto nextRuntimeDim = runtimeDimsVec.begin();
  SmallVector<uint64_t> shape;
  for (auto dim : *rawDims) {
    if (hasRawSize(dim))
      shape.push_back(rawSizeToRuntimeSize(dim));
    else {
      size_t runtimeDim = (*nextRuntimeDim++)->get<size_t>();
      if (!isValidRuntimeSize(runtimeDim)) {
        outSpec.setToError("invalid dimension");
        return;
      }
      shape.push_back(runtimeDim);
    }
  }
  outSpec.emplace(TensorSpec(shape, runtimeDType));
}

//===----------------------------------------------------------------------===//
// mop.cpu.tensor_spec.dims.dynamic
//===----------------------------------------------------------------------===//

/// Fulfills the result with dimensions of tensor spec which were unknown
/// at compile time.
static void mop_cpu_tensor_spec_dims_dynamic(
    const TensorSpec &tensorSpec, RemainingResults results,
    BEFAttr<ArrayRef<uint64_t>> rawDims, Runtime &runtime) {
  // All of the results are created as size_t.
  for (auto &result : results)
    result = AsyncValueRef<size_t>::allocate(runtime);

  unsigned next = 0;
  for (size_t i = 0, e = rawDims->size(); i < e; ++i)
    if (!hasRawSize((*rawDims)[i]))
      results[next++]->emplace<size_t>(tensorSpec[i]);
}

//===----------------------------------------------------------------------===//
// mop.cpu.tensor_spec.size
//===----------------------------------------------------------------------===//

static void mop_cpu_tensor_spec_size(const TensorSpec &tensorSpec,
                                     Result<size_t> result) {
  result.emplace(tensorSpec.getSizeInBytes());
}

//===----------------------------------------------------------------------===//
// mop.cpu.tensor_spec.num_elems
//===----------------------------------------------------------------------===//

static void mop_cpu_tensor_spec_num_elems(const TensorSpec &tensorSpec,
                                          Result<size_t> result) {
  result.emplace(tensorSpec.getNumElements());
}

//===----------------------------------------------------------------------===//
// mop.cpu.tensor_spec.dtype
//===----------------------------------------------------------------------===//

static void mop_cpu_tensor_spec_dtype(const TensorSpec &tensorSpec,
                                      Result<DType> result) {
  result.emplace(tensorSpec.getEltType());
}

//===----------------------------------------------------------------------===//
// mop.cpu.tensor.create
//===----------------------------------------------------------------------===//

/// Fulfills the outTensor by creating a tensor from the given tensor spec and
/// the buffer.
static void mop_cpu_tensor_create(Chain inChain, const TensorSpec &tensorSpec,
                                  AsyncValueRef<BufferRef> bufferRef,
                                  Result<Tensor> outTensor, Runtime &runtime) {
  // If this is the last user of the given buffer, let Tensor take the
  // ownership of the underlying buffer if the buffer is owned. If not, we will
  // clone the buffer and create a tensor from it.
  if (bufferRef.getPointer()->isUnique())
    outTensor.emplace(
        Tensor::create(std::move(*bufferRef), tensorSpec, runtime));
  else
    outTensor.emplace(Tensor::create(
        bufferRef->deepCopy(runtime.getAllocator()), tensorSpec, runtime));
}

//===----------------------------------------------------------------------===//
// mop.cpu.tensor.extract
//===----------------------------------------------------------------------===//

/// fulfills the outTensorSpec and outBuffer with the corresponding tensor spec
/// and buffer from the input tensor.
static void mop_cpu_tensor_extract(AsyncValueRef<Tensor> tensor,
                                   Result<TensorSpec> outTensorSpec,
                                   Result<BufferRef> outBuffer,
                                   Runtime &runtime) {
  outTensorSpec.emplace(tensor->getSpec());
  // If this is the last user of the given tensor, move the bufferRef to the
  // output. Otherwise, clone the underlying buffer.
  if (tensor.getPointer()->isUnique())
    outBuffer.emplace(tensor->releaseBuffer());
  else
    outBuffer.emplace(tensor->deepCopyBuffer());
}

//===----------------------------------------------------------------------===//
// mop.model
//===----------------------------------------------------------------------===//

void mop_model() { assert(false && "mop.model should never be invoked"); }

//===----------------------------------------------------------------------===//
// Primitive registration
//===----------------------------------------------------------------------===//

void M::registerMOPPrimitives(PrimitiveRegistry &registry) {
  StateSlot::registerType<DefaultStateSlot>();

  ADD_PRIMITIVE(registry, "mop.cpu.buffer.constant", mop_cpu_buffer_constant);
  ADD_PRIMITIVE(registry, "mop.cpu.buffer.alloc.static",
                mop_cpu_buffer_alloc_static);
  ADD_PRIMITIVE(registry, "mop.cpu.buffer.alloc.dynamic",
                mop_cpu_buffer_alloc_dynamic);
  ADD_PRIMITIVE(registry, "mop.cpu.buffer.alloc.spec",
                mop_cpu_buffer_alloc_spec);
  ADD_PRIMITIVE(registry, "mop.cpu.buffer.realloc.static",
                mop_cpu_buffer_realloc_static);
  ADD_PRIMITIVE(registry, "mop.cpu.buffer.realloc.dynamic",
                mop_cpu_buffer_realloc_dynamic);
  ADD_PRIMITIVE(registry, "mop.cpu.buffer.realloc.spec",
                mop_cpu_buffer_realloc_spec);
  ADD_PRIMITIVE(registry, "mop.cpu.buffer.copy", mop_cpu_buffer_copy);
  ADD_PRIMITIVE(registry, "mop.cpu.buffer.set", mop_cpu_buffer_set);
  ADD_PRIMITIVE(registry, "mop.context.create", mop_context_create);
  ADD_PRIMITIVE(registry, "mop.state_slot.create", mop_state_slot_create);
  ADD_PRIMITIVE(registry, "mop.cpu.tensor_spec.create",
                mop_cpu_tensor_spec_create);
  ADD_PRIMITIVE(registry, "mop.cpu.tensor_spec.create.dynamic",
                mop_cpu_tensor_spec_create_dynamic);
  ADD_PRIMITIVE(registry, "mop.cpu.tensor_spec.dims.dynamic",
                mop_cpu_tensor_spec_dims_dynamic);
  ADD_PRIMITIVE(registry, "mop.cpu.tensor_spec.size", mop_cpu_tensor_spec_size);
  ADD_PRIMITIVE(registry, "mop.cpu.tensor_spec.num_elems",
                mop_cpu_tensor_spec_num_elems);
  ADD_PRIMITIVE(registry, "mop.cpu.tensor_spec.dtype",
                mop_cpu_tensor_spec_dtype);
  ADD_PRIMITIVE(registry, "mop.cpu.tensor.create", mop_cpu_tensor_create);
  ADD_PRIMITIVE(registry, "mop.cpu.tensor.extract", mop_cpu_tensor_extract);
  ADD_PRIMITIVE(registry, "mop.model", mop_model);
}
