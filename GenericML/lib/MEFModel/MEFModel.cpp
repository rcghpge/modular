//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/MEFModel/MEFModel.h"
#include "GenericML/DeviceDriver/Driver.h"
#include "GenericML/MGPPrimitives/MGPPrimitives.h"
#include "GenericML/Support/MEFTypeUtils.h"
#include "GenericML/Support/MGPModel.h"
#include "GenericML/Support/Model.h"
#include "GenericML/Support/QuantSpec.h"
#include "MLRT/AsyncRT/Runtime/Algorithms.h"
#include "MLRT/Core/MojoValue.h"
#include "MLRT/GraphRT/MEFExecutor/MEFInvokable.h"
#include "MLRT/GraphRT/MEFExecutor/PrimitiveRegistry.h"
#include "MLRT/GraphRT/Primitives.h"
#include "Support/AssertStream.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/ML/CommonAttributes.h"
#include "Support/ML/CompiledFrameworkLabel.h"
#include "llvm/ADT/SmallVector.h"

using namespace M;
using namespace M::MO;

namespace M {
class MEFFile;
class ModelCollection;
} // namespace M

namespace M::MO {

struct ModularExecutableModel;

/// This class describes a model overall, including the MEF
/// invokables that implement its phases and metadata about what tensors it
/// takes and returns.
struct ModularModel final : public Model {
  static constexpr CompiledFrameworkLabel kFrameworkLabel{
      CompiledFrameworkLabel::kModularModel};

  /// Constructs a model using the positional input/output convention.
  ModularModel(std::string_view name, SmallVector<DeviceRef> devices,
               CompiledFrameworkLabel frameworkLabel,
               const MEFInvokable *initFn, const MEFInvokable *executeFn,
               ArrayRef<QuantTensorSpec> inputSpecs,
               ArrayRef<QuantTensorSpec> outputSpecs,
               SmallVector<DeviceRef> inputDevices,
               SmallVector<DeviceRef> outputDevices, EncodedLocation location)
      : Model(inputSpecs, outputSpecs), name(name), devices(std::move(devices)),
        initFn(initFn), executeFn(executeFn),
        inputDevices(std::move(inputDevices)),
        outputDevices(std::move(outputDevices)), location(std::move(location)) {
    ASSERT_STREAM(frameworkLabel == kFrameworkLabel,
                  << "Expected " << kFrameworkLabel.getAsString()
                  << " label for model, but got "
                  << frameworkLabel.getAsString());
  }

  /// Constructs a model using the named input/output convention.
  ModularModel(std::string_view name, SmallVector<DeviceRef> devices,
               CompiledFrameworkLabel frameworkLabel,
               const MEFInvokable *initFn, const MEFInvokable *executeFn,
               ArrayRef<QuantTensorSpec> inputSpecs,
               ArrayRef<QuantTensorSpec> outputSpecs,
               ArrayRef<StringRef> inputNames, ArrayRef<StringRef> outputNames,
               SmallVector<DeviceRef> inputDevices,
               SmallVector<DeviceRef> outputDevices, EncodedLocation location)
      : Model(inputSpecs, outputSpecs, inputNames, outputNames), name(name),
        devices(std::move(devices)), initFn(initFn), executeFn(executeFn),
        inputDevices(std::move(inputDevices)),
        outputDevices(std::move(outputDevices)), location(std::move(location)) {
    ASSERT_STREAM(frameworkLabel == kFrameworkLabel,
                  << "Expected " << kFrameworkLabel.getAsString()
                  << " label for model, but got "
                  << frameworkLabel.getAsString());
  }

  /// Returns a ModularModel from an mgp.model descriptor.
  static ErrorOr<std::unique_ptr<ModularModel>>
  getFromDescriptor(const PrimitiveDescriptor &descriptor) {
    return getModelFromDescriptor<ModularModel>(descriptor);
  }

  /// Used for matching primitives in MEFInvokable::{walk,recurse}Primitives.
  static StringLiteral getEntityName() { return "mgp.model"; }

  /// Calls the model's setup function using `runtime` and returns the results
  /// immediately. This function does not await the results to be ready.
  AsyncRT::ErrorDiagnosticOr<std::unique_ptr<ExecutableModel>>
  setup(Runtime &runtime, const WeightsRegistry *weightsRegistry,
        const M::GenericML::Driver::DeviceMap &deviceMap =
            M::GenericML::Driver::DeviceMap()) const final;

  /// Returns the name of the model.
  StringRef getName() const override { return name; }

  const MEFInvokable *getInitFn() const { return initFn; }
  const MEFInvokable *getExecuteFn() const { return executeFn; }
  const EncodedLocation &getLocation() const override { return location; }

  /// Return the devices used by the model.
  ArrayRef<DeviceRef> getDevices() const override { return devices; }

  /// Return an array of devices that map 1:1 to the devices used by each input.
  ArrayRef<DeviceRef> getInputDevices() const override { return inputDevices; }

  /// Return an array of devices that map 1:1 to the devices used by each
  /// output.
  ArrayRef<DeviceRef> getOutputDevices() const override {
    return outputDevices;
  }

private:
  std::string name;
  const SmallVector<DeviceRef> devices;
  const MEFInvokable *initFn;
  const MEFInvokable *executeFn;
  const SmallVector<DeviceRef> inputDevices;
  const SmallVector<DeviceRef> outputDevices;
  EncodedLocation location;
};

} // namespace M::MO

namespace M {

/// Instantiate model type traits.
template <>
struct ModelTraits<MO::ModularModel> {
  using ExecutableModelType = MO::ModularExecutableModel;
};
template <>
struct ModelTraits<MO::ModularExecutableModel> {
  using ModelType = MO::ModularModel;
};

} // namespace M

namespace M::MO {

struct ModularExecutableModel final
    : public MGPExecutableModelImpl<ModularExecutableModel> {
  using MGPExecutableModelImpl::MGPExecutableModelImpl;
};

AsyncRT::ErrorDiagnosticOr<std::unique_ptr<ExecutableModel>>
ModularModel::setup(Runtime &runtime, const WeightsRegistry *weightsRegistry,
                    const M::GenericML::Driver::DeviceMap &deviceMap) const {
  checkUniqueRuntime(runtime);

  SmallVector<AnyAsyncValueRef> initArgs;
  initArgs.reserve(kMgpModelNumFixedImplicitInitArgs + devices.size());

  // Populate init block args with a borrowed ref to the weights registry.
  initArgs.push_back(AnyAsyncValueRef::createReady<MojoValue>(
      runtime, MojoValue::createRef(weightsRegistry ? weightsRegistry->opaque()
                                                    : nullptr)));

  // Set device contexts from the device ref to device mapping.
  auto deviceContextsOr =
      mapDeviceRefsToDevices(getDevices(), deviceMap, runtime);
  if (deviceContextsOr.isError())
    return AsyncRT::EncodedDiagnostic(
        Twine("setup for model ") + getName() +
            " failed to map DeviceRefs to Devices: " +
            Twine(deviceContextsOr.getError()),
        location.copy());
  initArgs.append(std::make_move_iterator(deviceContextsOr->begin()),
                  std::make_move_iterator(deviceContextsOr->end()));

  // Run the init block.
  auto results = initFn->invoke(initArgs, runtime);
  if (results.size() != kMgpModelNumFixedImplicitExecArgs + devices.size())
    return AsyncRT::EncodedDiagnostic(
        Twine("setup for model ") + getName() + " returned " +
            Twine(results.size()) + " results (was expecting " +
            std::to_string(kMgpModelNumFixedImplicitExecArgs) + " + " +
            Twine(devices.size()) + " (device contexts))",
        location.copy());

  for (AnyAsyncValueRef &result : results) {
    if (result.isError()) {
      auto diagnostic = result.takeDiagnostic();
      return AsyncRT::EncodedDiagnostic(
          Twine("setup for model ") + getName() +
              " returned error: " + std::string(diagnostic.getMessage().get()),
          diagnostic.getLocation().copy());
    }
  }

  SmallVector<AnyAsyncValueRef> deviceContexts;
  deviceContexts.reserve(results.size() - kMgpModelNumFixedImplicitExecArgs);
  for (size_t i = kMgpModelNumFixedImplicitExecArgs; i < results.size(); ++i)
    deviceContexts.push_back(std::move(results[i]));

  return {std::make_unique<ModularExecutableModel>(
      /*chain=*/std::move(results[0]),
      /*mgpContext=*/std::move(results[1]),
      /*weightsRegistry=*/std::move(results[2]), std::move(deviceContexts),
      this, runtime)};
}

} // namespace M::MO

namespace M {
M::ErrorOr<std::unique_ptr<M::ModelCollection>>
MO::loadMGPMEFFile(RCRef<M::MEFFile> mefFile) {
  // Set up all our primitives.
  PrimitiveRegistry primitives;
  registerAllGraphRTPrimitives(primitives);
  registerMGPPrimitives(primitives);
  primitives.finalize();

  if (auto err = mefFile->installPrimitives(primitives))
    return err.takeError();

  SmallVector<M::ModelCollection::ModelPtrTy> models;
  auto *moduleInv = mefFile->getTopLevelInvokable();
  if (auto err = moduleInv->walkPrimitives<MO::ModularModel>(
          [&](auto &&m) { models.push_back(std::forward<decltype(m)>(m)); }))
    return err.takeError();

  return std::make_unique<M::ModelCollection>(std::move(models));
}

} // namespace M
