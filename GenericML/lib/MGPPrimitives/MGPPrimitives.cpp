//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/MGPPrimitives/MGPPrimitives.h"
#include "ArraySupport/StateContext.h"
#include "ArraySupport/StateSlot.h"
#include "ArraySupport/Tensor.h"
#include "GenericML/Support/Assert.h"
#include "GenericML/Support/CompiledKernelsContext.h"
#include "GenericML/Support/DebugPrint.h"
#include "GenericML/Support/HostDeviceSpecs.h"
#include "GenericML/Support/List.h"
#include "GraphRT/MEFExecutor/PrimitiveRegistry.h"
#include "LLCL/Runtime/Algorithms.h"
#include "LLCL/Runtime/AsyncValue.h"
#include "LLCL/Runtime/AsyncValueRef.h"
#include "LLCL/Runtime/Runtime.h"
#include "LLCL/Support/SpinWaiter.h"
#include "Support/DeviceSpecs.h"
#include "Support/MArchTarget/MArchTarget.h"
#include "Support/ML/SizeUtils.h"
#include "Support/ML/TensorShape.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include <algorithm>

#define DEBUG_TYPE "mgp"

using namespace M;
using namespace M::GML;

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// Profiler entry for launched CPU kernels.
using MGPProfilerEntry =
    ProfilerEntry<Trace::EnableTrace(Trace::kOther, 1), Trace::kOther>;

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
// mgp.create_runtime
//===----------------------------------------------------------------------===//

static void mgp_check_runtime(Chain inChain, StateContext &ctx,
                              Result<Chain> outChain, MEFAttr<uint64_t> aSlot,
                              MEFAttr<StringRef> bDevicesJSON,
                              Runtime &runtime) {
  auto providedDeviceSpecsOr =
      runtime.context->createIfMissing<DeviceSpecCollection>(
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

  ctx.emplaceStateSlot<RuntimeDeviceSpecMap>(*aSlot, /*numBuffers=*/0,
                                             std::move(*deviceSpecMapOr));
  std::move(outChain).emplace();
}

//===----------------------------------------------------------------------===//
// mgp.tensor_spec.create
//===----------------------------------------------------------------------===//

/// Creates a TensorSpec by combining the runtime and statically known
/// shape dimensions and the statically known DType. Only runtime dimensions
/// corresponding to unknown dimensions should be passed in runtimeDims.
static void mgp_tensor_spec_create(RemainingArguments runtimeDims,
                                   Result<TensorSpec> outSpec,
                                   MEFAttr<ArrayRef<uint64_t>> aRawDims,
                                   MEFAttr<uint8_t> bRawDType,
                                   Runtime &runtime) {
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
// mgp.tensor_spec.create.dynamic
//===----------------------------------------------------------------------===//

/// Creates a TensorSpec by combining the runtime and statically known
/// shape dimensions and a runtime DType. Only runtime dimensions corresponding
/// to unknown dimensions should be passed in runtimeDims.
static void mgp_tensor_spec_create_dynamic(DType runtimeDType,
                                           RemainingArguments runtimeDims,
                                           Result<TensorSpec> outSpec,
                                           MEFAttr<ArrayRef<uint64_t>> rawDims,
                                           Runtime &runtime) {
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
// mgp.tensor_spec.with_dtype
//===----------------------------------------------------------------------===//

static void mgp_tensor_spec_with_dtype(const TensorSpec &tensorSpec,
                                       Result<TensorSpec> outSpec,
                                       MEFAttr<uint8_t> rawDType,
                                       Runtime &runtime) {
  DType dtype(*rawDType);
  if (dtype.isInvalid())
    return std::move(outSpec).setToError("invalid dtype");
  std::move(outSpec).emplace(TensorSpec(tensorSpec.getDimsCopy(), dtype));
}

//===----------------------------------------------------------------------===//
// mgp.tensor_spec.dims.static
//===----------------------------------------------------------------------===//

/// Fulfills the result with dimensions of tensor spec which were known at
/// compile time.
static void mgp_tensor_spec_equal_static(const TensorSpec &tensorSpec,
                                         Result<bool> result,
                                         MEFAttr<ArrayRef<uint64_t>> rawDims,
                                         Runtime &runtime) {
  if (tensorSpec.getRank() != rawDims.size()) {
    std::move(result).emplace(false);
    return;
  }

  for (size_t i = 0; i < rawDims->size(); ++i) {
    const uint64_t expectedDimension = (*rawDims)[i];
    if (rawSizeToRuntimeSize(expectedDimension) != expectedDimension) {
      std::move(result).emplace(false);
      return;
    }
  }

  std::move(result).emplace(true);
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
// mgp.tensor_spec.size
//===----------------------------------------------------------------------===//

static void mgp_tensor_spec_size(const TensorSpec &tensorSpec,
                                 Result<size_t> result) {
  std::move(result).emplace(tensorSpec.getSizeInBytes());
}

//===----------------------------------------------------------------------===//
// mgp.tensor_spec.num_elems
//===----------------------------------------------------------------------===//

static void mgp_tensor_spec_num_elems(const TensorSpec &tensorSpec,
                                      Result<size_t> result) {
  std::move(result).emplace(tensorSpec.getNumElements());
}

//===----------------------------------------------------------------------===//
// mgp.tensor_spec.dtype
//===----------------------------------------------------------------------===//

static void mgp_tensor_spec_dtype(const TensorSpec &tensorSpec,
                                  Result<DType> result) {
  std::move(result).emplace(tensorSpec.getEltType());
}

//===----------------------------------------------------------------------===//
// mgp.debug.bool.print
//===----------------------------------------------------------------------===//

static void mgp_debug_bool_print(Chain inChain, const bool &cond,
                                 Result<Chain> outChain, Runtime &runtime) {
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
  addTask(runtime, [delay, outChain = outChain.copy()]() mutable {
    TimeTraceScope scope(
        MGPProfilerEntry::create("mgp.debug.cpu.spin_wait task"));
    size_t numLoops = 0;
    LLCL::BusyWaitSpinWaiter spinner(delay);
    while (!spinner.wait())
      ++numLoops;
    (void)numLoops;
    std::move(outChain).emplace();
  });
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
// mgp.generic.execute
//===----------------------------------------------------------------------===//

/// Profiler entry for launched CPU kernels.
using MGPProfilerEntry =
    ProfilerEntry<Trace::EnableTrace(Trace::kOther, 1), Trace::kOther>;

static void
mgp_generic_execute(StateContext &ctx, RemainingArguments operands,
                    Result<Chain> outChain, RemainingResults outResults,
                    MEFAttr<uint64_t> aSlot, MEFAttr<uint64_t> bModuleIndex,
                    MEFAttr<uint64_t> cFunctionIndex,
                    MEFAttr<StringRef> dSummary, MEFAttr<bool> eIsAsync,
                    Runtime &runtime, EncodedLocation loc) {
  auto profilerEntry = MGPProfilerEntry::create(*dSummary);
  M::MGP::CPURuntime &cpuRuntime =
      ctx.getStateSlotPayload<M::MGP::CPURuntime>(static_cast<size_t>(*aSlot));
  CompiledKernelAsyncABIFn *kernelPtr = cpuRuntime.getKernelAsyncABI(
      static_cast<size_t>(*bModuleIndex), static_cast<size_t>(*cFunctionIndex));

  LLCL::CompactRuntimePtr runtimePtr(runtime);
  // Allocate indirect results so that the primitive scheduler has something
  // to use. We always have to allocate indirects even if the kernel is not
  // async because we pass a copy of the results.
  outResults.allocateIndirectResults();
  auto runKernel = [kernelPtr,
                    // outResults _must_ be copied
                    outResults = outResults.copyAsVector(),
                    // std::move(operands) does not work, must use takeAsVector
                    takenOperands = operands.takeAsVector(),
                    outChain = outChain.copyWithLocation(),
                    summary = dSummary->str(),
                    profilerEntry = std::move(profilerEntry),
                    runtimePtr = std::move(runtimePtr),
                    loc = std::move(loc)]() mutable {
    std::vector<void *> kernelInputs;
    kernelInputs.reserve(outResults.size() + takenOperands.size());

    for (AnyAsyncValueRef &input : takenOperands)
      kernelInputs.push_back(&input);
    for (AnyAsyncValueRef &i : outResults)
      kernelInputs.push_back(&i);
    KGEN::MojoCallContext callContext(outChain.copy(),
                                      outChain.getLocation().copy());
    kernelPtr(reinterpret_cast<void **>(kernelInputs.data()),
              reinterpret_cast<void *>(runtimePtr.get()),
              reinterpret_cast<void *>(&callContext));

    if (outChain.isError()) {
      // The kernel failed while trying to launch.
      for (auto &result : outResults) {
        std::move(result).setToError(EncodedDiagnostic(
            outChain.copy().getDiagnostic().getMessage().copy(), loc.copy()));
      }
      return;
    }

    for (auto &result : outResults)
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
  GENERICML_ASSERT(false) << "mgp.model should never be invoked";
}

//===----------------------------------------------------------------------===//
// registerMGPPrimitives
//===----------------------------------------------------------------------===//

void M::registerMGPPrimitives(PrimitiveRegistry &registry) {
  ADD_PRIMITIVE(registry, "mgp.context.create", mgp_context_create);
  ADD_PRIMITIVE(registry, "mgp.state_slot.create", mgp_state_slot_create);
  ADD_PRIMITIVE(registry, "mgp.check_runtime", mgp_check_runtime);
  ADD_PRIMITIVE(registry, "mgp.tensor_spec.create", mgp_tensor_spec_create);
  ADD_PRIMITIVE(registry, "mgp.tensor_spec.create.dynamic",
                mgp_tensor_spec_create_dynamic);
  ADD_PRIMITIVE(registry, "mgp.tensor_spec.with_dtype",
                mgp_tensor_spec_with_dtype);
  ADD_PRIMITIVE(registry, "mgp.tensor_spec.equal.static",
                mgp_tensor_spec_equal_static);
  ADD_PRIMITIVE(registry, "mgp.tensor_spec.dims.dynamic",
                mgp_tensor_spec_dims_dynamic);
  ADD_PRIMITIVE(registry, "mgp.tensor_spec.size", mgp_tensor_spec_size);
  ADD_PRIMITIVE(registry, "mgp.tensor_spec.num_elems",
                mgp_tensor_spec_num_elems);
  ADD_PRIMITIVE(registry, "mgp.tensor_spec.dtype", mgp_tensor_spec_dtype);
  ADD_PRIMITIVE(registry, "mgp.debug.bool.print", mgp_debug_bool_print);
  ADD_PRIMITIVE(registry, "mgp.debug.cpu.spin_wait", mgp_debug_cpu_spin_wait);
  ADD_PRIMITIVE(registry, "mgp.assert", mgp_assert);
  ADD_PRIMITIVE(registry, "mgp.generic.execute", mgp_generic_execute);
  ADD_PRIMITIVE(registry, "mgp.list.append", mgp_list_append);
  ADD_PRIMITIVE(registry, "mgp.guard", mgp_guard);
  ADD_PRIMITIVE(registry, "mgp.list.concat", mgp_list_concat);
  ADD_PRIMITIVE(registry, "mgp.list.create", mgp_list_create);
  ADD_PRIMITIVE(registry, "mgp.list.get", mgp_list_get);
  ADD_PRIMITIVE(registry, "mgp.list.insert", mgp_list_insert);
  ADD_PRIMITIVE(registry, "mgp.list.slice", mgp_list_slice);
  ADD_PRIMITIVE(registry, "mgp.model", mgp_model);

  registerMGPCPUPrimitives(registry);
}
