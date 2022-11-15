//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/MOP/MOPPrimitives.h"
#include "GenericML/MOP/Context.h"
#include "GenericML/MOP/StateSlot.h"
#include "GenericML/MOP/Utils.h"
#include "GenericML/Support/Tensor.h"
#include "GraphRT/BEFExecutor/PrimitiveRegistry.h"
#include "Support/ML/TensorShape.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace M;
using namespace M::LLCL;

static ErrorOr<size_t> getSize(const SmallVector<AnyAsyncValueRef> &dynamicSize,
                               const uint64_t staticSize) {
  if (dynamicSize.size() > 1)
    return Error("unexpected argument provided");

  bool hasDynamicSize = !dynamicSize.empty();
  if (hasDynamicSize == MOP::haveSize(staticSize))
    return Error("expected a runtime size argument or a size attribute");

  return hasDynamicSize ? dynamicSize[0]->get<size_t>()
                        : static_cast<size_t>(staticSize);
}

//===----------------------------------------------------------------------===//
// mop.buffer.constant
//===----------------------------------------------------------------------===//

/// Fulfills the result with the borrowed BufferRef whose data is backed by
/// the given constant.
static void mop_buffer_constant(Result<BufferRef> outBuffer,
                                BEFDynamicAttr values, Runtime &runtime) {
  // Create unowned BufferRef as the BEF outlives the BufferRef.
  if (auto buf = dyn_cast<ArrayRef<uint8_t>>(values))
    outBuffer.emplace(BufferRef::createBorrowed(
        const_cast<void *>(static_cast<const void *>((*buf)->data())),
        (*buf)->size()));
  else
    outBuffer.setToError("unsupported type, only u8 is supported for buffer");
}

//===----------------------------------------------------------------------===//
// mop.buffer.alloc
//===----------------------------------------------------------------------===//

/// Fulfills the result with the BufferRef. The size can be provided at run time
/// with the async value or at compile time as an attribute. Alignment can be
/// given as an attribute.
static void mop_buffer_alloc(MOP::Context &ctx, RemainingArguments args,
                             Result<BufferRef> outBuffer,
                             BEFAttr<uint64_t> align, BEFAttr<uint64_t> size,
                             Runtime &runtime) {
  ErrorOr<size_t> sizeOr = getSize(args.takeAsVector(), *size);
  if (sizeOr.isError())
    outBuffer.setToError(sizeOr.takeError());

  if (MOP::haveAlign(*align) && MOP::isValidAlign(*align))
    outBuffer.emplace(
        BufferRef::createOwned(runtime.getAllocator(), *sizeOr, *align));
  else
    outBuffer.emplace(BufferRef::createOwned(runtime.getAllocator(), *sizeOr));
}

//===----------------------------------------------------------------------===//
// mop.buffer.realloc
//===----------------------------------------------------------------------===//

/// Fulfills the result with the RCBuffer. The RCBuffer can be the cached one
/// from the given (slot, output). If cache misses, allocates a new RCBuffer,
/// cache it, and fulfills the result with it. The size of the buffer can be
/// given at either run time or compile time.
static void mop_buffer_realloc(Chain chain, MOP::Context &ctx,
                               RemainingArguments args,
                               Result<BufferRef> outBuffer,
                               BEFAttr<uint64_t> align,
                               BEFAttr<uint64_t> output, BEFAttr<uint64_t> size,
                               BEFAttr<uint64_t> slot, Runtime &runtime) {
  ErrorOr<size_t> sizeOr = getSize(args.takeAsVector(), *size);
  if (sizeOr.isError())
    outBuffer.setToError(sizeOr.takeError());

  Allocator *allocator = runtime.getAllocator();
  MOP::StateSlot &stateSlot = ctx.getStateSlot(*slot);
  if (MOP::haveAlign(*align) && MOP::isValidAlign(*align))
    outBuffer.emplace(
        stateSlot.getOrCreateBuffer(*output, allocator, *sizeOr, *align));
  else
    outBuffer.emplace(stateSlot.getOrCreateBuffer(*output, allocator, *sizeOr));
}

//===----------------------------------------------------------------------===//
// mop.buffer.realloc
//===----------------------------------------------------------------------===//

/// Fulfills the result with the owning reference to the deep copy of inBuffer.
static void mop_buffer_copy(MOP::Context &ctx,
                            AsyncValueRef<BufferRef> inBuffer,
                            Result<BufferRef> outBuffer, Runtime &runtime) {
  outBuffer.emplace(inBuffer->deepCopy(runtime.getAllocator()));
}

//===----------------------------------------------------------------------===//
// mop.context.create
//===----------------------------------------------------------------------===//

/// Fulfills the result with a MOP::Context initialized with `numPrimContexts`
/// primitive contexts.
static void mop_context_create(Result<MOP::Context> result,
                               BEFAttr<uint64_t> numPrimContexts) {
  result->emplace<MOP::Context>(*numPrimContexts);
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
// mop.spec.create
//===----------------------------------------------------------------------===//

/// Creates a TensorSpec by combining the runtime and statically known
/// shape dimensions and the statically known DType. Only runtime dimensions
/// corresponding to unknown dimensions should be passed in runtimeDims.
static void mop_spec_create(RemainingArguments runtimeDims,
                            Result<TensorSpec> outSpec,
                            BEFAttr<ArrayRef<int64_t>> fstRawShape,
                            BEFAttr<uint8_t> sndRawDType, Runtime &runtime) {
  auto runtimeDimsVec = runtimeDims.takeAsVector();
  auto nextRuntimeDim = runtimeDimsVec.begin();
  SmallVector<uint64_t> shape;
  for (auto dim : *fstRawShape) {
    if (dim == MOP::kDynamicSize)
      shape.push_back((*nextRuntimeDim++)->get<size_t>());
    else
      shape.push_back(dim);
  }
  outSpec.emplace(TensorSpec(shape, DType(*sndRawDType)));
}

//===----------------------------------------------------------------------===//
// mop.spec.create.dyn
//===----------------------------------------------------------------------===//

/// Creates a TensorSpec by combining the runtime and statically known
/// shape dimensions and a runtime DType. Only runtime dimensions corresponding
/// to unknown dimensions should be passed in runtimeDims.
static void mop_spec_create_dyn(DType runtimeDType,
                                RemainingArguments runtimeDims,
                                Result<TensorSpec> outSpec,
                                BEFAttr<ArrayRef<int64_t>> rawShape,
                                Runtime &runtime) {
  auto runtimeDimsVec = runtimeDims.takeAsVector();
  auto nextRuntimeDim = runtimeDimsVec.begin();
  SmallVector<uint64_t> shape;
  for (auto dim : *rawShape) {
    if (dim == MOP::kDynamicSize)
      shape.push_back((*nextRuntimeDim++)->get<size_t>());
    else
      shape.push_back(dim);
  }
  outSpec.emplace(TensorSpec(shape, runtimeDType));
}

//===----------------------------------------------------------------------===//
// mop.spec.dims.dyn
//===----------------------------------------------------------------------===//

/// Fulfills the result with dimensions of tensor spec which were unknown
/// at compile time.
static void mop_spec_dims_dyn(TensorSpec spec, RemainingResults results,
                              BEFAttr<ArrayRef<int64_t>> rawShape,
                              Runtime &runtime) {
  // All of the results are created as size_t.
  for (auto &result : results)
    result = AsyncValueRef<size_t>::allocate(runtime);

  unsigned next = 0;
  for (unsigned i = 0, e = rawShape->size(); i < e; ++i)
    if ((*rawShape)[i] == MOP::kDynamicSize)
      results[next++]->emplace<size_t>(spec[i]);
}

//===----------------------------------------------------------------------===//
// mop.spec.size
//===----------------------------------------------------------------------===//

static void mop_spec_size(TensorSpec tensorSpec, Result<size_t> result) {
  result.emplace(tensorSpec.getSizeInBytes());
}

//===----------------------------------------------------------------------===//
// mop.spec.num_elems
//===----------------------------------------------------------------------===//

static void mop_spec_num_elems(TensorSpec tensorSpec, Result<size_t> result) {
  result.emplace(tensorSpec.getNumElements());
}

//===----------------------------------------------------------------------===//
// mop.spec.dtype
//===----------------------------------------------------------------------===//

static void mop_spec_dtype(TensorSpec tensorSpec, Result<DType> result) {
  result.emplace(tensorSpec.getEltType());
}

//===----------------------------------------------------------------------===//
// mop.tensor.create
//===----------------------------------------------------------------------===//

/// Fulfills the outTensor by creating a tensor from the given tensor spec and
/// the buffer.
static void mop_tensor_create(Chain inChain, TensorSpec tensorSpec,
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
// mop.tensor.extract
//===----------------------------------------------------------------------===//

/// fulfills the outSpec and outBuffer with the corresponding spec/buffer from
/// the input tensor.
static void mop_tensor_extract(AsyncValueRef<Tensor> tensor,
                               Result<TensorSpec> outSpec,
                               Result<BufferRef> outBuffer, Runtime &runtime) {
  outSpec.emplace(tensor->getSpec());
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

  ADD_PRIMITIVE(registry, "mop.buffer.constant", mop_buffer_constant);
  ADD_PRIMITIVE(registry, "mop.buffer.alloc", mop_buffer_alloc);
  ADD_PRIMITIVE(registry, "mop.buffer.realloc", mop_buffer_realloc);
  ADD_PRIMITIVE(registry, "mop.buffer.copy", mop_buffer_copy);
  ADD_PRIMITIVE(registry, "mop.context.create", mop_context_create);
  ADD_PRIMITIVE(registry, "mop.state_slot.create", mop_state_slot_create);
  ADD_PRIMITIVE(registry, "mop.spec.create", mop_spec_create);
  ADD_PRIMITIVE(registry, "mop.spec.create.dyn", mop_spec_create_dyn);
  ADD_PRIMITIVE(registry, "mop.spec.dims.dyn", mop_spec_dims_dyn);
  ADD_PRIMITIVE(registry, "mop.spec.size", mop_spec_size);
  ADD_PRIMITIVE(registry, "mop.spec.num_elems", mop_spec_num_elems);
  ADD_PRIMITIVE(registry, "mop.spec.dtype", mop_spec_dtype);
  ADD_PRIMITIVE(registry, "mop.tensor.create", mop_tensor_create);
  ADD_PRIMITIVE(registry, "mop.tensor.extract", mop_tensor_extract);
  ADD_PRIMITIVE(registry, "mop.model", mop_model);
}
