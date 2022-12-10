//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/MGPPrimitives/MGPPrimitives.h"
#include "GenericML/Support/Context.h"
#include "GenericML/Support/StateSlot.h"
#include "GenericML/Support/Tensor.h"
#include "GraphRT/BEFExecutor/PrimitiveRegistry.h"
#include "LLCL/Runtime/AsyncValue.h"
#include "LLCL/Runtime/AsyncValueRef.h"
#include "Support/ML/SizeUtils.h"
#include "Support/ML/TensorShape.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "mgp_primitives"

using namespace M;
using namespace M::GML;

static Optional<size_t> getAlign(uint64_t rawAlign) {
  return hasRawSize(rawAlign) ? Optional<size_t>(rawSizeToRuntimeSize(rawAlign))
                              : llvm::None;
}

//===----------------------------------------------------------------------===//
// mgp.cpu.buffer.constant
//===----------------------------------------------------------------------===//

/// Fulfills the result with the borrowed BufferRef whose data is backed by
/// the given constant.
static void mgp_cpu_buffer_constant(Result<BufferRef> outBuffer,
                                    BEFDynamicAttr aValue,
                                    BEFAttr<uint64_t> bRawAlign,
                                    Runtime &runtime) {
  Optional<size_t> optAlign;
  if (hasRawSize(*bRawAlign))
    optAlign = rawSizeToRuntimeSize(*bRawAlign);

  // Create unowned BufferRef as the BEF outlives the BufferRef.
  // TODO(#5369): Remove need for const cast.
  if (auto buf = dyn_cast<ArrayRef<uint8_t>>(aValue))
    outBuffer.emplace(GML::BufferRef::createBorrowed(
        const_cast<void *>(static_cast<const void *>((*buf)->data())),
        (*buf)->size(), optAlign, BufferRef::kConstant));
  else
    outBuffer.setToError(
        "unsupported type, only u8 is supported for attribute elements");
}

//===----------------------------------------------------------------------===//
// mgp.cpu.buffer.alloc.*
//===----------------------------------------------------------------------===//

/// Returns in result a freshly allocated BufferRef. The size can be provided
/// statically, via an runtime size argument, or via a runtime TensorSpec.
/// Any alignment attribute is respected.
static void mgp_cpu_buffer_alloc_static(Context &ctx, Result<BufferRef> result,
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
  result.emplace(
      BufferRef::createOwned(runtime, runtimeSize, getAlign(*bRawAlign)));
}

static void mgp_cpu_buffer_alloc_dynamic(Context &ctx, size_t runtimeSize,
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
  result.emplace(
      BufferRef::createOwned(runtime, runtimeSize, getAlign(*rawAlign)));
}

static void mgp_cpu_buffer_alloc_spec(Context &ctx, const TensorSpec &spec,
                                      Result<BufferRef> result,
                                      BEFAttr<uint64_t> rawAlign,
                                      Runtime &runtime) {
  if (!isValidRawAlign(*rawAlign)) {
    result.setToError("invalid align attribute");
    return;
  }
  result.emplace(BufferRef::createOwned(runtime, spec.getSizeInBytes(),
                                        getAlign(*rawAlign)));
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
  result.emplace(stateSlot.getOrCreateBuffer(*bBuffer, runtime, runtimeSize,
                                             getAlign(*dRawAlign)));
}

static void mgp_cpu_buffer_realloc_dynamic(
    Chain inChain, Context &ctx, size_t runtimeSize, Result<BufferRef> result,
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
  result.emplace(stateSlot.getOrCreateBuffer(*bBuffer, runtime, runtimeSize,
                                             getAlign(*cRawAlign)));
}

static void
mgp_cpu_buffer_realloc_spec(Chain inChain, Context &ctx, const TensorSpec &spec,
                            Result<BufferRef> result, BEFAttr<uint64_t> aSlot,
                            BEFAttr<uint64_t> bBuffer,
                            BEFAttr<uint64_t> cRawAlign, Runtime &runtime) {
  if (!isValidRawAlign(*cRawAlign)) {
    result.setToError("invalid align attribute");
    return;
  }
  StateSlot &stateSlot = ctx.getStateSlot(*aSlot);
  result.emplace(stateSlot.getOrCreateBuffer(
      *bBuffer, runtime, spec.getSizeInBytes(), getAlign(*cRawAlign)));
}

//===----------------------------------------------------------------------===//
// mgp.cpu.buffer.copy
//===----------------------------------------------------------------------===//

/// Fulfills the result with the owning reference to the deep copy of inBuffer.
static void mgp_cpu_buffer_copy(Context &ctx, AsyncValueRef<BufferRef> inBuffer,
                                Result<BufferRef> outBuffer, Runtime &runtime) {
  outBuffer.emplace(inBuffer->deepCopy(runtime));
}

//===----------------------------------------------------------------------===//
// mgp.cpu.buffer.set
//===----------------------------------------------------------------------===//

/// Set each byte of the given inBuffer with the given Value, returns chain.
static Chain mgp_cpu_buffer_set(BufferRef &buffer, BEFAttr<uint8_t> value) {
  memset(buffer.getMutableBuffer(), *value, buffer.getSize());
  return {};
}

//===----------------------------------------------------------------------===//
// mgp.context.create
//===----------------------------------------------------------------------===//

/// Fulfills the result with a Context initialized with `numPrimContexts`
/// primitive contexts.
static void mgp_context_create(Result<Context> result,
                               BEFAttr<uint64_t> numSlots) {
  result->emplace<Context>(*numSlots);
}

//===----------------------------------------------------------------------===//
// mgp.state_slot.create
//===----------------------------------------------------------------------===//

/// Constructs the default state slot with the given number of cached buffers
/// under the given slot in the mgp context.
static Chain mgp_state_slot_create(Chain inChain, Context &ctx,
                                   BEFAttr<uint64_t> numBuffers,
                                   BEFAttr<uint64_t> slot) {
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
// mgp.cpu.tensor_spec.create.dynamic
//===----------------------------------------------------------------------===//

/// Creates a TensorSpec by combining the runtime and statically known
/// shape dimensions and a runtime DType. Only runtime dimensions corresponding
/// to unknown dimensions should be passed in runtimeDims.
static void mgp_cpu_tensor_spec_create_dynamic(
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
// mgp.cpu.tensor_spec.dims.dynamic
//===----------------------------------------------------------------------===//

/// Fulfills the result with dimensions of tensor spec which were unknown
/// at compile time.
static void mgp_cpu_tensor_spec_dims_dynamic(
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
// mgp.cpu.tensor_spec.size
//===----------------------------------------------------------------------===//

static void mgp_cpu_tensor_spec_size(const TensorSpec &tensorSpec,
                                     Result<size_t> result) {
  result.emplace(tensorSpec.getSizeInBytes());
}

//===----------------------------------------------------------------------===//
// mgp.cpu.tensor_spec.num_elems
//===----------------------------------------------------------------------===//

static void mgp_cpu_tensor_spec_num_elems(const TensorSpec &tensorSpec,
                                          Result<size_t> result) {
  result.emplace(tensorSpec.getNumElements());
}

//===----------------------------------------------------------------------===//
// mgp.cpu.tensor_spec.dtype
//===----------------------------------------------------------------------===//

static void mgp_cpu_tensor_spec_dtype(const TensorSpec &tensorSpec,
                                      Result<DType> result) {
  result.emplace(tensorSpec.getEltType());
}

//===----------------------------------------------------------------------===//
// mgp.cpu.tensor.create
//===----------------------------------------------------------------------===//

/// Fulfills the outTensor by creating a tensor from the given tensor spec and
/// the buffer.
static void mgp_cpu_tensor_create(Chain inChain,
                                  AsyncValueRef<BufferRef> bufferRef,
                                  const TensorSpec &tensorSpec,
                                  Result<Tensor> outTensor, Runtime &runtime) {
  if (bufferRef.getPointer()->isUnique()) {
    // If this is the only reference to the input buffer then it cannot be
    // further mutated, so we can move it into the result tensor.
    LLVM_DEBUG(llvm::dbgs() << "mgp.cpu.tensor.create: Creating M::Tensor "
                               "taking ownership of buffer\n");
    outTensor.emplace(Tensor::create(std::move(*bufferRef), tensorSpec));
  } else {
    // Otherwise we must take a copy of the buffer to protect the result
    // tensor from side-effects.
    LLVM_DEBUG(llvm::dbgs()
               << "mgp.cpu.tensor.create: Creating M::Tensor from copy of "
                  "buffer since still has "
               << std::to_string(bufferRef.getPointer()->getRefCount())
               << " users\n");
    outTensor.emplace(Tensor::create(bufferRef->deepCopy(runtime), tensorSpec));
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
  outTensorSpec.emplace(tensor->getSpec());
  if (tensor.getPointer()->isUnique())
    // If this is the only reference to the input tensor then it cannot
    // be further observed, so we can move its buffer into the result.
    outBuffer.emplace(tensor->releaseBuffer());
  else
    // Otherwise we must take a copy of the buffer to protect the input tensor
    // from side-effects.
    outBuffer.emplace(tensor->deepCopyBuffer(runtime));
}

//===----------------------------------------------------------------------===//
// mgp.model
//===----------------------------------------------------------------------===//

void mgp_model() { assert(false && "mgp.model should never be invoked"); }

//===----------------------------------------------------------------------===//
// Primitive registration
//===----------------------------------------------------------------------===//

void M::registerMGPPrimitives(PrimitiveRegistry &registry) {
  StateSlot::registerType<DefaultStateSlot>();

  ADD_PRIMITIVE(registry, "mgp.cpu.buffer.constant", mgp_cpu_buffer_constant);
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
  ADD_PRIMITIVE(registry, "mgp.cpu.buffer.copy", mgp_cpu_buffer_copy);
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
  ADD_PRIMITIVE(registry, "mgp.model", mgp_model);
}
