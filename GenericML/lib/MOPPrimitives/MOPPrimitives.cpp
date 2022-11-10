//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/MOP/MOPPrimitives.h"
#include "GenericML/MOP/Context.h"
#include "GenericML/MOP/PrimContext.h"
#include "GenericML/Support/Tensor.h"
#include "GraphRT/BEFExecutor/PrimitiveRegistry.h"
#include "Support/ML/TensorShape.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace M;
using namespace M::LLCL;

/// Denotes a dynamic dimension in a static tensor spec shape.
/// This corresponds to the MLIR ShapedType::kDynamicSize, however
/// the -1 literal is used ubiquitously.
constexpr int64_t kDynamicSize = -1;

/// Denotes a missing size or alignment attribute value.
constexpr uint64_t kMissingSizeOrAlign = std::numeric_limits<uint64_t>::max();

static ErrorOr<size_t> getSize(const SmallVector<AnyAsyncValueRef> &dynamicSize,
                               const uint64_t staticSize) {
  if (dynamicSize.size() > 1)
    return Error("unexpected argument provided");

  bool hasDynamicSize = !dynamicSize.empty();
  bool hasStaticSize = (staticSize != kMissingSizeOrAlign);
  if (hasDynamicSize == hasStaticSize)
    return Error("expected a runtime size argument or a size attribute");

  return hasDynamicSize ? dynamicSize[0]->get<size_t>()
                        : static_cast<size_t>(staticSize);
}

// TODO(#4705): Consolidate utility functions in MOPPrimitives and MOPDialect.
Optional<size_t> getAlign(const uint64_t align) {
  if (align != kMissingSizeOrAlign)
    return align;
  return None;
}

//===----------------------------------------------------------------------===//
// mop.buffer.constant
//===----------------------------------------------------------------------===//

/// Fulfills the result with the immoral RCBufferRef whose data is backed by
/// the given constant.
static void mop_buffer_constant(Result<RCBufferRef> outBuffer,
                                BEFDynamicAttr values, Runtime &runtime) {
  // Create immortal RCBuffer as the BEF outlives the RCBuffer.
  if (auto buf = dyn_cast<ArrayRef<uint8_t>>(values))
    outBuffer.emplace(RCBuffer::createImmortal(
        const_cast<void *>(static_cast<const void *>((*buf)->data())),
        (*buf)->size()));
  else
    outBuffer.setToError("unsupported type, only u8 is supported for buffer");
}

//===----------------------------------------------------------------------===//
// mop.buffer.alloc
//===----------------------------------------------------------------------===//

/// Fulfills the result with the RCBuffer. The size can be provided at run time
/// with the async value or at compile time as an attribute. Alignment can be
/// given as an attribute.
static void mop_buffer_alloc(MOP::Context &ctx, RemainingArguments args,
                             Result<RCBufferRef> outBuffer,
                             BEFAttr<uint64_t> align, BEFAttr<uint64_t> size,
                             Runtime &runtime) {
  ErrorOr<size_t> sizeOr = getSize(args.takeAsVector(), *size);
  if (sizeOr.isError())
    outBuffer.setToError(sizeOr.takeError());

  if (auto validAlign = getAlign(*align))
    outBuffer.emplace(
        RCBuffer::createHeap(runtime.getAllocator(), *sizeOr, *validAlign));
  else
    outBuffer.emplace(RCBuffer::createHeap(runtime.getAllocator(), *sizeOr));
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
                               Result<RCBufferRef> outBuffer,
                               BEFAttr<uint64_t> align,
                               BEFAttr<uint64_t> output, BEFAttr<uint64_t> size,
                               BEFAttr<uint64_t> slot, Runtime &runtime) {
  ErrorOr<size_t> sizeOr = getSize(args.takeAsVector(), *size);
  if (sizeOr.isError())
    outBuffer.setToError(sizeOr.takeError());

  Allocator *allocator = runtime.getAllocator();
  MOP::AnyGenericPrimContext &primCtx = ctx.getPrimContext(*slot);
  if (auto validAlign = getAlign(*align))
    outBuffer.emplace(
        primCtx.getOrCreateBuffer(*output, allocator, *sizeOr, *validAlign));
  else
    outBuffer.emplace(primCtx.getOrCreateBuffer(*output, allocator, *sizeOr));
}

//===----------------------------------------------------------------------===//
// mop.buffer.realloc
//===----------------------------------------------------------------------===//

/// Fulfills the result with the owning reference to the deep copy of inBuffer.
static void mop_buffer_copy(MOP::Context &ctx, RCBufferRef &inBuffer,
                            Result<RCBufferRef> outBuffer, Runtime &runtime) {
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

/// Constructs the default prim context with the given outputs under the given
/// slot in the mop context.
static Chain mop_state_slot_create(MOP::Context &ctx,
                                   BEFAttr<uint64_t> numOutputs,
                                   BEFAttr<uint64_t> slot) {
  MOP::AnyGenericPrimContext::registerFramework<MOP::DefaultPrimContext>();
  ctx.emplacePrimContext<MOP::DefaultPrimContext>(*slot, *numOutputs);
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
    if (dim == kDynamicSize)
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
    if (dim == kDynamicSize)
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
    if ((*rawShape)[i] == kDynamicSize)
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
                              RCBufferRef &buffer, Result<Tensor> outTensor,
                              Runtime &runtime) {
  outTensor.emplace(Tensor::get(buffer.copy(), tensorSpec, runtime));
}

//===----------------------------------------------------------------------===//
// mop.tensor.extract
//===----------------------------------------------------------------------===//

/// fulfills the outSpec and outBuffer with the corresponding spec/buffer from
/// the input tensor.
static void mop_tensor_extract(Tensor &tensor, Result<TensorSpec> outSpec,
                               Result<RCBufferRef> outBuffer) {
  outSpec.emplace(tensor.getSpec());
  outBuffer.emplace(tensor.getBuffer());
}

//===----------------------------------------------------------------------===//
// Primitive registration
//===----------------------------------------------------------------------===//

void M::registerMOPPrimitives(PrimitiveRegistry &registry) {
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
}
