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

static ErrorOr<size_t> getSize(const SmallVector<AnyAsyncValueRef> &dynamicSize,
                               const int64_t staticSize) {
  if (dynamicSize.size() > 1)
    return Error("unexpected argument provided");

  bool hasDynamicSize = !dynamicSize.empty();
  bool hasStaticSize = (staticSize != mlir::ShapedType::kDynamicSize);
  if (hasDynamicSize == hasStaticSize)
    return Error("expected a runtime size argument or a size attribute");

  return hasDynamicSize ? dynamicSize[0]->get<size_t>() : staticSize;
}

Optional<size_t> getAlign(const int64_t align) {
  if (align != mlir::ShapedType::kDynamicSize)
    return align;
  return None;
}

//===----------------------------------------------------------------------===//
// mop.buffer.constant
//===----------------------------------------------------------------------===//

// Fulfills the result with the immoral RCBufferRef whose data is backed by
// the given constant.
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
// mop.buffer.create
//===----------------------------------------------------------------------===//

// Fulfills the result with the RCBuffer. The size can be provided at run time
// with the async value or at compile time as an attribute. Alignment can be
// given as an attribute.
static void mop_buffer_create(MOP::Context &ctx, RemainingArguments args,
                              Result<RCBufferRef> outBuffer,
                              BEFAttr<int64_t> align, BEFAttr<int64_t> size,
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
// mop.buffer.recreate
//===----------------------------------------------------------------------===//

// Fulfills the result with the RCBuffer. The RCBuffer can be the cached one
// from the given (slot, output). If cache misses, allocates a new RCBuffer,
// cache it, and fulfills the result with it. The size of the buffer can be
// given at either run time or compile time.
static void mop_buffer_recreate(Chain chain, MOP::Context &ctx,
                                RemainingArguments args,
                                Result<RCBufferRef> outBuffer,
                                BEFAttr<int64_t> align,
                                BEFAttr<uint64_t> output, BEFAttr<int64_t> size,
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
// mop.context.create
//===----------------------------------------------------------------------===//

// Fulfills the result with a MOP::Context initialized with `numPrimContexts`
// primitive contexts.
static void mop_context_create(Result<MOP::Context> result,
                               BEFAttr<uint64_t> numPrimContexts) {
  result->emplace<MOP::Context>(*numPrimContexts);
}

//===----------------------------------------------------------------------===//
// mop.primcontext.create
//===----------------------------------------------------------------------===//

// Constructs the default prim context with the given outputs under the given
// slot in the mop context.
static Chain mop_primcontext_create(MOP::Context &ctx,
                                    BEFAttr<uint64_t> numOutputs,
                                    BEFAttr<uint64_t> slot) {
  MOP::AnyGenericPrimContext::registerFramework<MOP::DefaultPrimContext>();
  ctx.emplacePrimContext<MOP::DefaultPrimContext>(*slot, *numOutputs);
  return {};
}

//===----------------------------------------------------------------------===//
// mop.spec.wrap
//===----------------------------------------------------------------------===//

// Fulfills the result with a tensor spec. Dimensions are given as an
// attribute, while static cally unknown dimensions are given at the runtime.
static void mop_spec_wrap(RemainingArguments runtimeDims,
                          Result<TensorSpec> outSpec,
                          BEFAttr<ArrayRef<int64_t>> dims,
                          BEFAttr<BEFType> elementType, Runtime &runtime) {
  auto runtimeDimsVec = runtimeDims.takeAsVector();
  auto nextRuntimeDim = runtimeDimsVec.begin();
  SmallVector<uint64_t> shape;
  for (auto dim : *dims) {
    if (dim == -1)
      shape.push_back((*nextRuntimeDim++)->get<size_t>());
    else
      shape.push_back(dim);
  }
  outSpec.emplace(TensorSpec(shape, elementType.getKind()));
}

//===----------------------------------------------------------------------===//
// mop.spec.unwrap
//===----------------------------------------------------------------------===//

static void mop_spec_unwrap() {
  // TODO: implement when mop.spec.unwrap op is fixed.
}

//===----------------------------------------------------------------------===//
// mop.spec.size
//===----------------------------------------------------------------------===//

static void mop_spec_size(TensorSpec tensorSpec, Result<size_t> result) {
  result.emplace(tensorSpec.getSizeInBytes());
}

//===----------------------------------------------------------------------===//
// mop.tensor.wrap
//===----------------------------------------------------------------------===//

// Fulfills the outTensor by creating a tensor from the given tensor spec and
// the buffer.
static void mop_tensor_wrap(TensorSpec tensorSpec, RCBufferRef &buffer,
                            Result<Tensor> outTensor, Runtime &runtime) {
  outTensor.emplace(Tensor::get(buffer.copy(), tensorSpec, runtime));
}

//===----------------------------------------------------------------------===//
// mop.tensor.unwrap
//===----------------------------------------------------------------------===//

static void mop_tensor_unwrap() {
  // TODO: implement with Tensor's getBuffer().
}

//===----------------------------------------------------------------------===//
// Primitive registration
//===----------------------------------------------------------------------===//

void M::registerMOPPrimitives(PrimitiveRegistry &registry) {
  ADD_PRIMITIVE(registry, "mop.buffer.constant", mop_buffer_constant);
  ADD_PRIMITIVE(registry, "mop.buffer.create", mop_buffer_create);
  ADD_PRIMITIVE(registry, "mop.buffer.recreate", mop_buffer_recreate);
  ADD_PRIMITIVE(registry, "mop.context.create", mop_context_create);
  ADD_PRIMITIVE(registry, "mop.primcontext.create", mop_primcontext_create);
  ADD_PRIMITIVE(registry, "mop.spec.wrap", mop_spec_wrap);
  ADD_PRIMITIVE(registry, "mop.spec.unwrap", mop_spec_unwrap);
  ADD_PRIMITIVE(registry, "mop.spec.size", mop_spec_size);
  ADD_PRIMITIVE(registry, "mop.tensor.wrap", mop_tensor_wrap);
  ADD_PRIMITIVE(registry, "mop.tensor.unwrap", mop_tensor_unwrap);
}
