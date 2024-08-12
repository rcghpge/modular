//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/MGPPrimitives/MGPPrimitives.h"
#include "ArraySupport/StateContext.h"
#include "ArraySupport/StateSlot.h"
#include "ArraySupport/Tensor.h"
#include "AsyncRT/Runtime/Algorithms.h"
#include "AsyncRT/Runtime/AsyncValue.h"
#include "AsyncRT/Runtime/AsyncValueRef.h"
#include "AsyncRT/Runtime/Runtime.h"
#include "CUDASupport/CUDARuntime.h"
#include "CUDASupport/Globals/Globals.h"
#include "GenericML/Support/CompiledKernelsContext.h"
#include "GenericML/Support/DebugPrint.h"
#include "GenericML/Support/HostDeviceSpecs.h"
#include "GenericML/Support/List.h"
#include "GraphRT/MEFExecutor/PrimitiveRegistry.h"
#include "Support/AssertStream.h"
#include "Support/DeviceSpecs.h"
#include "Support/MArchTarget/MArchTarget.h"
#include "Support/ML/SizeUtils.h"
#include "Support/ML/TensorShape.h"
#include "Support/Threading/SpinWaiter.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include <algorithm>

#define DEBUG_TYPE "mgp"

using namespace M;
using namespace M::GML;
using namespace M::CUDA;

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// Profiler entry for launched CPU kernels.
using MGPProfilerEntry =
    ProfilerEntry<Trace::EnableTrace(Trace::kOther, 1), Trace::kOther>;

//===----------------------------------------------------------------------===//
// List Helpers
//===----------------------------------------------------------------------===//

// TODO(GRA-801): Replace mo.list with opaque type implementation
namespace {
// ABI version of GML::List.
// NOTE the layout must match the mojo `ABI_List`
struct ABIList {
  size_t numElems;
  void *elements; // Currently always points to ABITensors

  /// Constructor
  ABIList(size_t numElems, void *elements)
      : numElems(numElems), elements(elements) {}
};

// ABI version of Tensor.
// NOTE the layout must match the mojo `ABI_Tensor`
// Used only for handling lists of tensors
struct ABITensor {
  ssize_t *dims; // Shape, with implied rank
  void *dataPtr; // The underlying buffer

  // Constructor
  ABITensor(ssize_t *dims, void *dataPtr) : dims(dims), dataPtr(dataPtr) {}
};

// Converts runtime GML::List (used to back mo.list), into a form known to MOGG.
// Allocates in the past `*Memory` vectors
ErrorOr<void *>
listToABIListPtr(GML::List &runtimeList,
                 SmallVector<AnyAsyncValueRef> &listMemory,
                 SmallVector<SmallVector<ABITensor>> &listTensorMemory,
                 SmallVector<SmallVector<SmallVector<ssize_t>>> &listDimMemory,
                 Runtime &runtime) {
  auto &abiTensors = listTensorMemory.emplace_back();
  auto &abiDims = listDimMemory.emplace_back();
  abiDims.reserve(runtimeList.getElements().size());
  abiTensors.reserve(runtimeList.getElements().size());

  // Convert individual elements to their corresponding ABI structures.
  // NOTE: This is currently not closed under all argument types and cannot
  //       be extended by subclasses.
  for (GML::Variant &ref : runtimeList.getElements()) {
    if (ref.getValue().isType<Tensor>()) {
      auto &runtimeTensor = ref.getValue().get<Tensor>();
      const TensorShape &shape = runtimeTensor.getShape();
      ssize_t *dims;
      // If the Tensor is a scalar, need to upscale to 1D so mojo
      // can reason about it
      if (shape.getRank() == 0)
        dims = abiDims.emplace_back(1, 1).data();
      else
        dims = abiDims.emplace_back(shape.begin(), shape.end()).data();
      void *data = const_cast<void *>(runtimeTensor.getBuffer());
      abiTensors.emplace_back(dims, data);
    } else {
      return Error(Twine("Invalid list element type: ") +
                   ref.getValue().getPointer()->getTypeID().getTypeName());
    }
  }

  // Convert an entire list to its ABI structure.
  listMemory.push_back(AnyAsyncValueRef::createReady<ABIList>(
      runtime, runtimeList.getElements().size(), abiTensors.data()));

  return &listMemory.back();
}

/// Marshalls the given async value (mainly for handling lists) and may allocate
/// into the given `*Memory` tensors which backs the returned ptr.
ErrorOr<void *> getAnyAsyncValueRefPtr(
    AnyAsyncValueRef &input, SmallVector<AnyAsyncValueRef> &listMemory,
    SmallVector<SmallVector<ABITensor>> &listTensorMemory,
    SmallVector<SmallVector<SmallVector<ssize_t>>> &listDimMemory,
    Runtime &runtime) {
  if (input.isType<GML::List>()) {
    return listToABIListPtr(input.get<GML::List>(), listMemory,
                            listTensorMemory, listDimMemory, runtime);
  }

  return &input;
}

} // namespace

//===----------------------------------------------------------------------===//
// mgp.context.create
//===----------------------------------------------------------------------===//

/// Fulfills the result with a StateContext initialized with `numPrimContexts`
/// primitive contexts.
static void mgp_context_create(Result<StateContext> result,
                               MEFAttr<uint64_t> numSlots) {
  std::move(result).emplace(*numSlots);
}

//===----------------------------------------------------------------------===//
// mgp.state_slot.create
//===----------------------------------------------------------------------===//

/// Constructs the default state slot with the given number of cached buffers
/// under the given slot in the mgp context.
static Chain mgp_state_slot_create(Chain inChain, StateContext &ctx,
                                   MEFAttr<uint64_t> numBuffers,
                                   MEFAttr<uint64_t> slot) {
  ctx.emplaceStateSlot<DefaultStateSlot>(*slot, *numBuffers);
  return {};
}

//===----------------------------------------------------------------------===//
// mgp.tensor_spec.dims.dynamic
//===----------------------------------------------------------------------===//

/// Fulfills the result with dimensions of tensor spec which were unknown
/// at compile time.
static void mgp_tensor_spec_dims_dynamic(const TensorSpec &tensorSpec,
                                         RemainingResults results,
                                         MEFAttr<ArrayRef<uint64_t>> rawDims,
                                         Runtime &runtime) {
  // All of the results are created as size_t.
  for (auto &result : results)
    result = AsyncValueRef<size_t>::allocate(runtime);

  unsigned next = 0;
  for (size_t i = 0, e = rawDims->size(); i < e; ++i)
    if (!hasRawSize((*rawDims)[i]))
      results[next++].copy().emplace<size_t>(tensorSpec[i]);
}

//===----------------------------------------------------------------------===//
// mgp.generic.execute
//===----------------------------------------------------------------------===//

/// Profiler entry for launched CPU kernels.
using MGPProfilerEntry =
    ProfilerEntry<Trace::EnableTrace(Trace::kOther, 1), Trace::kOther>;

static void mgp_generic_execute(
    StateContext &ctx, Chain inChain, AnyAsyncValueRef deviceContext,
    RemainingArguments operands, Result<Chain> outChain,
    RemainingResults outResults, MEFAttr<uint64_t> aSlot,
    MEFAttr<uint64_t> bModuleIndex, MEFAttr<uint64_t> cFunctionIndex,
    MEFAttr<StringRef> dSummary, MEFAttr<bool> eIsAsync, MEFAttr<bool> fHasCuda,
    MEFAttr<uint64_t> gCudaSlot, Runtime &runtime, EncodedLocation loc) {
  auto profilerEntry = MGPProfilerEntry::create(*dSummary);
  M::MGP::KernelExecutionState &kernelExecutionState =
      ctx.getStateSlotPayload<M::MGP::KernelExecutionState>(
          static_cast<size_t>(*aSlot));
  CompiledKernelAsyncABIFn *kernelPtr = kernelExecutionState.getKernelAsyncABI(
      static_cast<size_t>(*bModuleIndex), static_cast<size_t>(*cFunctionIndex));

  AsyncRT::CompactRuntimePtr runtimePtr(runtime);
  // Allocate indirect results so that the primitive scheduler has something
  // to use. We always have to allocate indirects even if the kernel is not
  // async because we pass a copy of the results.
  outResults.allocateIndirectResults();

  auto runKernel = [kernelPtr, &ctx, &runtime,
                    // outResults _must_ be copied
                    outResults = outResults.copyAsVector(),
                    // std::move(operands) does not work, must use takeAsVector
                    takenOperands = operands.takeAsVector(),
                    outChain = outChain.copyWithLocation(),
                    summary = dSummary->str(), hasCuda = fHasCuda.get(),
                    cudaSlot = gCudaSlot.get(),
                    profilerEntry = std::move(profilerEntry),
                    runtimePtr = std::move(runtimePtr),
                    deviceContext = std::move(deviceContext),
                    loc = std::move(loc)]() mutable {
    CUDARuntime *cudaRuntimePtr = nullptr;
    void *deviceContextPtr = deviceContext.get<void *>();
    std::optional<CurrentCUDAContext> current;
    if (hasCuda) {
      // GRA-594: When Device Context is passed to the primitive directly, we
      // don't need to set the runtime
      cudaRuntimePtr = reinterpret_cast<CUDARuntime *>(
          ctx.getStateSlot(static_cast<size_t>(cudaSlot))
              .getUnderlyingPointer());
      current.emplace(*cudaRuntimePtr);
    }
    std::vector<void *> kernelInputs;
    kernelInputs.reserve(outResults.size() + takenOperands.size());

    // Stack allocation needed to handle marshalling to ABIList.
    // Holds references to ABIList wrapped in AnyAsyncValue.
    SmallVector<AnyAsyncValueRef> listMemory;

    // For each associated ABIList, holds the ABITensors in the list.
    SmallVector<SmallVector<ABITensor>> listTensorMemory;

    // For each associated ABITensor, holds the dimensions of the tensor.
    // listDimMemory[1] --> the storage for the index 1 list
    // listDimMemory[0][1] --> the storage for the index 0 lists's 1st tensor
    // listDimMemory[0][1][2] --> the 2nd index dimension of the above
    SmallVector<SmallVector<SmallVector<ssize_t>>> listDimMemory;

    // Note we must be careful of reference invalidation with our SmallVectors
    // as we typically pass pointers to the data directly. Therefore, reserve
    // space to avoid issues.
    size_t maxNumLists = takenOperands.size() + outResults.size();
    listMemory.reserve(maxNumLists);
    listTensorMemory.reserve(maxNumLists);
    listDimMemory.reserve(maxNumLists);

    for (AnyAsyncValueRef &res :
         llvm::concat<AnyAsyncValueRef>(takenOperands, outResults)) {
      ErrorOr<void *> ptrToMarshalledValue = getAnyAsyncValueRefPtr(
          res, listMemory, listTensorMemory, listDimMemory, runtime);

      if (ptrToMarshalledValue.isError()) {
        std::move(res).setToError(EncodedDiagnostic(
            outChain.copy().getDiagnostic().getMessage().copy(), loc.copy()));
        std::move(outChain).setToError(ptrToMarshalledValue.takeError().copy());
        return;
      }

      kernelInputs.push_back(ptrToMarshalledValue.takeValue());
    }

    MojoCallContext callContext(outChain.copy(), outChain.getLocation().copy(),
                                cudaRuntimePtr, deviceContextPtr);
    kernelPtr(reinterpret_cast<void **>(kernelInputs.data()),
              reinterpret_cast<void *>(runtimePtr.get()),
              reinterpret_cast<void *>(&callContext));

    if (outChain.isError()) {
      for (auto &result : outResults) {
        if (result.isReady())
          continue;
        std::move(result).setToError(EncodedDiagnostic(
            outChain.copy().getDiagnostic().getMessage().copy(), loc.copy()));
      }
      return;
    }

    for ([[maybe_unused]] auto &result : outResults)
      assert(result.isReady());

    std::move(outChain).emplace();
    std::move(profilerEntry).record();

    // Make sure our profiling entry (or any which the Mojo kernel may have
    // designated as 'current') does not escape the scope of the Mojo kernel
    // call, either synchronous or up to first await point for asynchronous.
    MGPProfilerEntry::clearCurrentId();
  };

  if (eIsAsync.get())
    runtime.getWorkQueue()->addTask(std::move(runKernel));
  else
    runKernel();
}

//===----------------------------------------------------------------------===//
// mgp.list.append
//===----------------------------------------------------------------------===//

static void mgp_list_append(List &inputList, AnyAsyncValueRef element,
                            Result<List> output) {
  if (element.isType<Tensor>()) {
    if (!inputList.getElements().empty() &&
        inputList.getElements().front().getValue().getPointer()->getTypeID() !=
            element.getPointer()->getTypeID()) {
      std::move(output).setToError("Tensor must be from the same device");
      return;
    }
  }

  // Copy input list then append element into it.
  List newList = inputList.copy();
  newList.getElements().push_back(element.copy());
  std::move(output).emplace(std::move(newList));
}

//===----------------------------------------------------------------------===//
// mgp.list.concat
//===----------------------------------------------------------------------===//

static void mgp_list_concat(List &lhs, List &rhs, Result<List> output) {
  List resultList = lhs.copy();
  resultList.getElements().insert(resultList.getElements().end(),
                                  rhs.getElements().begin(),
                                  rhs.getElements().end());

  std::move(output).emplace(std::move(resultList));
}

//===----------------------------------------------------------------------===//
// mgp.list.create
//===----------------------------------------------------------------------===//

static void mgp_list_create(RemainingArguments operands,
                            Result<List> outputList) {
  std::move(outputList).emplace(List{operands.copyAsVector()});
}

//===----------------------------------------------------------------------===//
// mgp.list.get
//===----------------------------------------------------------------------===//

static void mgp_list_get(List &inputList, Tensor &idxTensor,
                         RemainingResults outputs, EncodedLocation loc) {
  // We must unbox the index value from the tensor.
  // TODO: Should be ssize_t.
  idxTensor.getEltType()
      .dispatch<void>(idxTensor.getBuffer())
      .template when<DType::si32, DType::si64>([&](auto *idxDataPtr) {
        auto idx = *idxDataPtr;

        // Check index is in bound.
        int64_t listSize = inputList.getElements().size();
        if (idx < -listSize || listSize <= idx) {
          return std::move(outputs).setAllToError(
              Twine("index ") + Twine(idx) +
                  " is out of bound for list of size " + Twine(listSize),
              loc);
        }

        // Normalize index.
        if (idx < 0)
          idx += listSize;

        // We must set the output this way because
        // (1). list can contain arbitray types of values, so we can't put a
        // single type into `Result<...>`> (2). we want to share the
        // underlying value as `AsyncValueRef` to avoid deep copying expensive
        // objects like `Tensor`.
        *outputs.begin() = inputList.getElements()[idx].getValue().copy();
      });
}

//===----------------------------------------------------------------------===//
// mgp.list.insert
//===----------------------------------------------------------------------===//

static void mgp_list_insert(List &inputList, AnyAsyncValueRef element,
                            Tensor &idxTensor, Result<List> output) {
  // We must unbox the index value from the tensor.
  // TODO: Should be ssize_t.
  idxTensor.getEltType()
      .dispatch<void>(idxTensor.getBuffer())
      .template when<DType::si32, DType::si64>([&](auto *idxDataPtr) {
        auto idx = *idxDataPtr;

        // Check index is in bound.
        int64_t listSize = inputList.getElements().size();
        if (idx < -listSize || listSize < idx) {
          return std::move(output).setToError(
              Twine("index ") + Twine(idx) +
              " is out of bound for list of size " + Twine(listSize));
        }

        // Normalize index.
        if (idx < 0)
          idx += listSize;

        // Copy input list then insert element into it.
        List newList = inputList.copy();
        newList.getElements().insert(newList.getElements().begin() + idx,
                                     element.copy());
        std::move(output).emplace(std::move(newList));
      });
}

//===----------------------------------------------------------------------===//
// mgp.list.slice
//===----------------------------------------------------------------------===//

static void mgp_list_slice(List &inputList, Tensor &startTensor,
                           Tensor &endTensor, Tensor &stepTensor,
                           Result<List> output) {

  auto sliceFunctor = [&](auto start, auto end, auto step) {
    List resultList;

    auto &elems = inputList.getElements();

    // start/end/step
    ssize_t length = elems.size();

    if (step == 0) {
      return std::move(output).setToError(
          "The slice step argument must be a non-zero");
    }

    const bool stepIsPositive = step > 0;

    // normalize negative start and end values
    if (start < 0) {
      start += length;
    }
    if (end < 0) {
      end += length;
    }

    if (stepIsPositive) {
      start = std::clamp<decltype(start)>(start, 0, length - 1);
      end = std::clamp<decltype(end)>(end, 0, length);
    } else {
      start = std::clamp<decltype(start)>(start, 0, length - 1);
      end = std::clamp<decltype(end)>(end, -1, length);
    }

    if (stepIsPositive) {
      for (ssize_t i = start; i < end && i < length && i >= 0; i += step) {
        resultList.getElements().push_back(inputList.getElements()[i]);
      }
    } else {
      for (ssize_t i = start; i > end; i += step) {
        resultList.getElements().push_back(inputList.getElements()[i]);
      }
    }

    std::move(output).emplace(std::move(resultList));
  };

  startTensor.getEltType()
      .dispatch<void>(startTensor.getBuffer())
      .template when<DType::si32, DType::si64>([&](auto *startDataPtr) {
        auto start = *startDataPtr;

        endTensor.getEltType()
            .dispatch<void>(endTensor.getBuffer())
            .template when<DType::si32, DType::si64>([&](auto *endDataPtr) {
              auto end = *endDataPtr;

              stepTensor.getEltType()
                  .dispatch<void>(stepTensor.getBuffer())
                  .template when<DType::si32, DType::si64>(
                      [&](auto *stepDataPtr) {
                        auto step = *stepDataPtr;
                        sliceFunctor(start, end, step);
                      });
            });
      });
}

//===----------------------------------------------------------------------===//
// mgp.guard
//===----------------------------------------------------------------------===//

static void mgp_guard(Chain inChain, RemainingArguments inputs,
                      RemainingResults results) {
  assert(inputs.size() == results.size());

  auto inputAsyncVals = inputs.get();
  auto resultAsnycVals = results.get();

  for (const auto &[input, result] :
       llvm::zip(inputAsyncVals, resultAsnycVals)) {
    result = std::move(input);
  }
}

//===----------------------------------------------------------------------===//
// mgp.model
//===----------------------------------------------------------------------===//

static void mgp_model() {
  ASSERT_STREAM(false, << "mgp.model should never be invoked");
}

//===----------------------------------------------------------------------===//
// mgp.extend.lifetime
//===----------------------------------------------------------------------===//

// No-op, will be scheduled once all are ready (preventing the inputs from being
// freed).
static void mgp_nop(RemainingArguments operands, Result<Chain> outChain) {
  std::move(outChain).emplace();
}

//===----------------------------------------------------------------------===//
// registerMGPPrimitives
//===----------------------------------------------------------------------===//

void M::registerMGPPrimitives(PrimitiveRegistry &registry) {
  ADD_PRIMITIVE(registry, "mgp.context.create", mgp_context_create);
  ADD_PRIMITIVE(registry, "mgp.state_slot.create", mgp_state_slot_create);
  ADD_PRIMITIVE(registry, "mgp.tensor_spec.dims.dynamic",
                mgp_tensor_spec_dims_dynamic);
  ADD_PRIMITIVE(registry, "mgp.generic.execute", mgp_generic_execute);
  ADD_PRIMITIVE(registry, "mgp.list.append", mgp_list_append);
  ADD_PRIMITIVE(registry, "mgp.guard", mgp_guard);
  ADD_PRIMITIVE(registry, "mgp.list.concat", mgp_list_concat);
  ADD_PRIMITIVE(registry, "mgp.list.create", mgp_list_create);
  ADD_PRIMITIVE(registry, "mgp.list.get", mgp_list_get);
  ADD_PRIMITIVE(registry, "mgp.list.insert", mgp_list_insert);
  ADD_PRIMITIVE(registry, "mgp.list.slice", mgp_list_slice);
  ADD_PRIMITIVE(registry, "mgp.model", mgp_model);

  ADD_PRIMITIVE(registry, "mgp.extend.lifetime", mgp_nop);

  registerMGPCPUPrimitives(registry);
}
