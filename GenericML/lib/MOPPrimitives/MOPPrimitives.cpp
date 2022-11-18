//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/MOP/MOPPrimitives.h"
#include "GenericML/MOP/Context.h"
#include "GenericML/MOP/StateSlot.h"
#include "GenericML/Support/Tensor.h"
#include "GraphRT/BEFExecutor/PrimitiveRegistry.h"
#include "Support/ML/SizeUtils.h"
#include "Support/ML/TensorShape.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace M;
using namespace M::LLCL;

/// Returns the size to use when allocating a buffer. If rawStaticSize
/// is the distinguished missing value then the size must come from the
/// unique entry in dynamicSize. Otherwise rawStaticSize is returned.
static ErrorOr<size_t> getSize(const SmallVector<AnyAsyncValueRef> &dynamicSize,
                               const uint64_t rawStaticSize) {
  if (dynamicSize.size() > 1)
    return Error("unexpected argument provided");

  bool hasDynamicSize = !dynamicSize.empty();
  if (hasDynamicSize == hasRawSize(rawStaticSize))
    return Error("expected a runtime size argument or a size attribute");

  return hasDynamicSize ? dynamicSize[0]->get<size_t>()
                        : asRuntimeSize(rawStaticSize);
}

//===----------------------------------------------------------------------===//
// mop.cpu.buffer.constant
//===----------------------------------------------------------------------===//

/// Fulfills the result with the borrowed BufferRef whose data is backed by
/// the given constant.
static void mop_cpu_buffer_constant(Result<BufferRef> outBuffer,
                                    BEFDynamicAttr values, Runtime &runtime) {
  // Create unowned BufferRef as the BEF outlives the BufferRef.
  if (auto buf = dyn_cast<ArrayRef<uint8_t>>(values))
    outBuffer.emplace(BufferRef::createBorrowed(
        const_cast<void *>(static_cast<const void *>((*buf)->data())),
        (*buf)->size()));
  else
    outBuffer.setToError(
        "unsupported type, only u8 is supported for attribute elements");
}

//===----------------------------------------------------------------------===//
// mop.cpu.buffer.alloc
//===----------------------------------------------------------------------===//

/// Fulfills the result with the BufferRef. The size can be provided at run time
/// with the async value or at compile time as an attribute. Alignment can be
/// given as an attribute.
static void mop_cpu_buffer_alloc(MOP::Context &ctx, RemainingArguments args,
                                 Result<BufferRef> outBuffer,
                                 BEFAttr<uint64_t> rawAlign,
                                 BEFAttr<uint64_t> rawSize, Runtime &runtime) {
  ErrorOr<size_t> sizeOr = getSize(args.takeAsVector(), *rawSize);
  if (sizeOr.isError()) {
    outBuffer.setToError(sizeOr.takeError());
    return;
  }
  outBuffer.emplace(BufferRef::createOwned(
      runtime.getAllocator(), *sizeOr,
      asRuntimeSizeOrDefault(*rawAlign, kPreferredMemoryAlignment)));
}

//===----------------------------------------------------------------------===//
// mop.cpu.buffer.realloc
//===----------------------------------------------------------------------===//

/// Fulfills the result with the RCBuffer. The RCBuffer can be the cached one
/// from the given (slot, output). If cache misses, allocates a new RCBuffer,
/// cache it, and fulfills the result with it. The size of the buffer can be
/// given at either run time or compile time.
static void
mop_cpu_buffer_realloc(Chain chain, MOP::Context &ctx, RemainingArguments args,
                       Result<BufferRef> outBuffer, BEFAttr<uint64_t> aSlot,
                       BEFAttr<uint64_t> bBuffer, BEFAttr<uint64_t> cRawAlign,
                       BEFAttr<uint64_t> dRawSize, Runtime &runtime) {
  ErrorOr<size_t> sizeOr = getSize(args.takeAsVector(), *dRawSize);
  if (sizeOr.isError()) {
    outBuffer.setToError(sizeOr.takeError());
    return;
  }
  MOP::StateSlot &stateSlot = ctx.getStateSlot(*aSlot);
  outBuffer.emplace(stateSlot.getOrCreateBuffer(
      *bBuffer, runtime.getAllocator(), *sizeOr,
      asRuntimeSizeOrDefault(*cRawAlign, kPreferredMemoryAlignment)));
}

//===----------------------------------------------------------------------===//
// mop.cpu.buffer.copy
//===----------------------------------------------------------------------===//

/// Fulfills the result with the owning reference to the deep copy of inBuffer.
static void mop_cpu_buffer_copy(MOP::Context &ctx,
                                AsyncValueRef<BufferRef> inBuffer,
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

/// Fulfills the result with a MOP::Context initialized with `numPrimContexts`
/// primitive contexts.
static void mop_context_create(Result<MOP::Context> result,
                               BEFAttr<uint64_t> numSlots) {
  result->emplace<MOP::Context>(*numSlots);
}

//===----------------------------------------------------------------------===//
// mop.state_slot.create
//===----------------------------------------------------------------------===//

/// Constructs the default state slot with the given number of cached buffers
/// under the given slot in the mop context.
static Chain mop_state_slot_create(MOP::Context &ctx,
                                   BEFAttr<uint64_t> numBuffers,
                                   BEFAttr<uint64_t> slot) {
  ctx.emplaceStateSlot<MOP::DefaultStateSlot>(*slot, *numBuffers);
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
                                       BEFAttr<ArrayRef<uint64_t>> fstRawDims,
                                       BEFAttr<uint8_t> sndRawDType,
                                       Runtime &runtime) {
  auto runtimeDimsVec = runtimeDims.takeAsVector();
  auto nextRuntimeDim = runtimeDimsVec.begin();
  SmallVector<uint64_t> shape;
  for (uint64_t dim : *fstRawDims) {
    if (hasRawSize(dim))
      shape.push_back(asRuntimeSize(dim));
    else
      shape.push_back((*nextRuntimeDim++)->get<size_t>());
  }
  outSpec.emplace(TensorSpec(shape, DType(*sndRawDType)));
}

//===----------------------------------------------------------------------===//
// mop.cpu.tensor_spec.create.dyn
//===----------------------------------------------------------------------===//

/// Creates a TensorSpec by combining the runtime and statically known
/// shape dimensions and a runtime DType. Only runtime dimensions corresponding
/// to unknown dimensions should be passed in runtimeDims.
static void mop_cpu_tensor_spec_create_dyn(DType runtimeDType,
                                           RemainingArguments runtimeDims,
                                           Result<TensorSpec> outSpec,
                                           BEFAttr<ArrayRef<uint64_t>> rawDims,
                                           Runtime &runtime) {
  auto runtimeDimsVec = runtimeDims.takeAsVector();
  auto nextRuntimeDim = runtimeDimsVec.begin();
  SmallVector<uint64_t> shape;
  for (auto dim : *rawDims) {
    if (hasRawSize(dim))
      shape.push_back(asRuntimeSize(dim));
    else
      shape.push_back((*nextRuntimeDim++)->get<size_t>());
  }
  outSpec.emplace(TensorSpec(shape, runtimeDType));
}

//===----------------------------------------------------------------------===//
// mop.cpu.tensor_spec.dims.dyn
//===----------------------------------------------------------------------===//

/// Fulfills the result with dimensions of tensor spec which were unknown
/// at compile time.
static void mop_cpu_tensor_spec_dims_dyn(const TensorSpec &tensorSpec,
                                         RemainingResults results,
                                         BEFAttr<ArrayRef<uint64_t>> rawDims,
                                         Runtime &runtime) {
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
    outTensor.emplace(Tensor::get(std::move(*bufferRef), tensorSpec, runtime));
  else
    outTensor.emplace(Tensor::get(bufferRef->deepCopy(runtime.getAllocator()),
                                  tensorSpec, runtime));
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
  MOP::StateSlot::registerType<MOP::DefaultStateSlot>();

  ADD_PRIMITIVE(registry, "mop.cpu.buffer.constant", mop_cpu_buffer_constant);
  ADD_PRIMITIVE(registry, "mop.cpu.buffer.alloc", mop_cpu_buffer_alloc);
  ADD_PRIMITIVE(registry, "mop.cpu.buffer.realloc", mop_cpu_buffer_realloc);
  ADD_PRIMITIVE(registry, "mop.cpu.buffer.copy", mop_cpu_buffer_copy);
  ADD_PRIMITIVE(registry, "mop.cpu.buffer.set", mop_cpu_buffer_set);
  ADD_PRIMITIVE(registry, "mop.context.create", mop_context_create);
  ADD_PRIMITIVE(registry, "mop.state_slot.create", mop_state_slot_create);
  ADD_PRIMITIVE(registry, "mop.cpu.tensor_spec.create",
                mop_cpu_tensor_spec_create);
  ADD_PRIMITIVE(registry, "mop.cpu.tensor_spec.create.dyn",
                mop_cpu_tensor_spec_create_dyn);
  ADD_PRIMITIVE(registry, "mop.cpu.tensor_spec.dims.dyn",
                mop_cpu_tensor_spec_dims_dyn);
  ADD_PRIMITIVE(registry, "mop.cpu.tensor_spec.size", mop_cpu_tensor_spec_size);
  ADD_PRIMITIVE(registry, "mop.cpu.tensor_spec.num_elems",
                mop_cpu_tensor_spec_num_elems);
  ADD_PRIMITIVE(registry, "mop.cpu.tensor_spec.dtype",
                mop_cpu_tensor_spec_dtype);
  ADD_PRIMITIVE(registry, "mop.cpu.tensor.create", mop_cpu_tensor_create);
  ADD_PRIMITIVE(registry, "mop.cpu.tensor.extract", mop_cpu_tensor_extract);
  ADD_PRIMITIVE(registry, "mop.model", mop_model);
}
