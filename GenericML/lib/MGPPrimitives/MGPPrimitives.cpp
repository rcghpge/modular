//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/MGPPrimitives/MGPPrimitives.h"
#include "GenericML/Support/Assert.h"
#include "GenericML/Support/CompiledKernelsContext.h"
#include "GenericML/Support/Context.h"
#include "GenericML/Support/DebugPrint.h"
#include "GenericML/Support/HostDeviceSpecs.h"
#include "GenericML/Support/List.h"
#include "GenericML/Support/StateSlot.h"
#include "GenericML/Support/Tensor.h"
#include "GraphRT/MEFExecutor/PrimitiveRegistry.h"
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

/// Fulfills the result with a Context initialized with `numPrimContexts`
/// primitive contexts.
static void mgp_context_create(Result<Context> result,
                               MEFAttr<uint64_t> numSlots) {
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

  ctx.emplaceStateSlot<RuntimeDeviceSpecMap>(*aSlot, /*numBuffers=*/0,
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
  DType dtype(*rawDType);
  if (dtype.isInvalid())
    return std::move(outSpec).setToError("invalid dtype");
  std::move(outSpec).emplace(TensorSpec(tensorSpec.getDimsCopy(), dtype));
}

//===----------------------------------------------------------------------===//
// mgp.cpu.tensor_spec.dims.static
//===----------------------------------------------------------------------===//

/// Fulfills the result with dimensions of tensor spec which were known at
/// compile time.
static void mgp_cpu_tensor_spec_equal_static(
    const TensorSpec &tensorSpec, Result<bool> result,
    MEFAttr<ArrayRef<uint64_t>> rawDims, Runtime &runtime) {
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
// mgp.cpu.tensor_spec.dims.dynamic
//===----------------------------------------------------------------------===//

/// Fulfills the result with dimensions of tensor spec which were unknown
/// at compile time.
static void mgp_cpu_tensor_spec_dims_dynamic(
    const TensorSpec &tensorSpec, RemainingResults results,
    MEFAttr<ArrayRef<uint64_t>> rawDims, Runtime &runtime) {
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
  std::move(result).emplace(tensorSpec.getSizeInBytes());
}

//===----------------------------------------------------------------------===//
// mgp.cpu.tensor_spec.num_elems
//===----------------------------------------------------------------------===//

static void mgp_cpu_tensor_spec_num_elems(const TensorSpec &tensorSpec,
                                          Result<size_t> result) {
  std::move(result).emplace(tensorSpec.getNumElements());
}

//===----------------------------------------------------------------------===//
// mgp.cpu.tensor_spec.dtype
//===----------------------------------------------------------------------===//

static void mgp_cpu_tensor_spec_dtype(const TensorSpec &tensorSpec,
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
// mgp.to_grt_bool.deprecated
//===----------------------------------------------------------------------===//

static void mgp_to_grt_bool(Chain inChain,
                            AsyncValueRef<GML::BufferRef> bufferRef,
                            Result<bool> result, Runtime &runtime) {
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
      GENERICML_ASSERT(start + 1 == end && combined[start] == '\0');
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
// mgp.cpu.kernel.execute.deprecated
//===----------------------------------------------------------------------===//

static void mgp_generic_execute(Context &ctx, RemainingArguments operands,
                                RemainingResults results,
                                MEFAttr<uint64_t> aSlot,
                                MEFAttr<uint64_t> bModuleIndex,
                                MEFAttr<uint64_t> cFunctionIndex,
                                Runtime &runtime, EncodedLocation loc) {
  M::MGP::CPURuntime &cpuRuntime =
      ctx.getStateSlotPayload<M::MGP::CPURuntime>(static_cast<size_t>(*aSlot));
  CompiledKernelAsyncABIFn *kernelPtr = cpuRuntime.getKernelAsyncABI(
      static_cast<size_t>(*bModuleIndex), static_cast<size_t>(*cFunctionIndex));

  SmallVector<void *> kernelInputs;
  kernelInputs.reserve(results.size() + operands.size());

  for (AnyAsyncValueRef &input : operands)
    kernelInputs.push_back(&input);
  for (AnyAsyncValueRef &result : results)
    kernelInputs.push_back(&result);
  kernelPtr(reinterpret_cast<void **>(kernelInputs.data()),
            reinterpret_cast<void *>(&runtime));
}

//===----------------------------------------------------------------------===//
// mgp.list.append
//===----------------------------------------------------------------------===//

static void mgp_list_append(List &inputList, AnyAsyncValueRef element,
                            Result<List> output) {
  // Copy input list then append element into it.
  List newList = inputList.copy();
  newList.getElements().push_back(element.copy());
  std::move(output).emplace(std::move(newList));
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
        // objects like `GML::Tensor`.
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
  ADD_PRIMITIVE(registry, "mgp.cpu.tensor_spec.create",
                mgp_cpu_tensor_spec_create);
  ADD_PRIMITIVE(registry, "mgp.cpu.tensor_spec.create.dynamic",
                mgp_cpu_tensor_spec_create_dynamic);
  ADD_PRIMITIVE(registry, "mgp.cpu.tensor_spec.with_dtype",
                mgp_cpu_tensor_spec_with_dtype);
  ADD_PRIMITIVE(registry, "mgp.cpu.tensor_spec.equal.static",
                mgp_cpu_tensor_spec_equal_static);
  ADD_PRIMITIVE(registry, "mgp.cpu.tensor_spec.dims.dynamic",
                mgp_cpu_tensor_spec_dims_dynamic);
  ADD_PRIMITIVE(registry, "mgp.cpu.tensor_spec.size", mgp_cpu_tensor_spec_size);
  ADD_PRIMITIVE(registry, "mgp.cpu.tensor_spec.num_elems",
                mgp_cpu_tensor_spec_num_elems);
  ADD_PRIMITIVE(registry, "mgp.cpu.tensor_spec.dtype",
                mgp_cpu_tensor_spec_dtype);
  ADD_PRIMITIVE(registry, "mgp.debug.bool.print", mgp_debug_bool_print);
  ADD_PRIMITIVE(registry, "mgp.debug.cpu.spin_wait", mgp_debug_cpu_spin_wait);
  ADD_PRIMITIVE(registry, "mgp.to_grt_bool.deprecated", mgp_to_grt_bool);
  ADD_PRIMITIVE(registry, "mgp.assert", mgp_assert);
  ADD_PRIMITIVE(registry, "mgp.cpu.kernel.create_context",
                mgp_cpu_kernel_create_context);
  ADD_PRIMITIVE(registry, "mgp.cpu.kernel.load_archive",
                mgp_cpu_kernel_load_archive);
  ADD_PRIMITIVE(registry, "mgp.generic.execute", mgp_generic_execute);
  ADD_PRIMITIVE(registry, "mgp.list.append", mgp_list_append);
  ADD_PRIMITIVE(registry, "mgp.list.create", mgp_list_create);
  ADD_PRIMITIVE(registry, "mgp.list.get", mgp_list_get);
  ADD_PRIMITIVE(registry, "mgp.list.insert", mgp_list_insert);
  ADD_PRIMITIVE(registry, "mgp.model", mgp_model);

  registerMGPCPUPrimitives(registry);
}
