//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
// This file is a copy-paste of ModularModel.h and ModularModelFile.h with
// the name changed to MEFModel. TODO: #18388 Remove duplication.
//
#ifndef GRAPHCOMPILER_TRANSFORMS_MODIALECT_MEFMODEL_H
#define GRAPHCOMPILER_TRANSFORMS_MODIALECT_MEFMODEL_H

#include "GenericML/Support/Model.h"
#include "GraphRT/MEFExecutor/MEFInvokable.h"
#include "LLCL/ForwardDecls.h"
#include "Support/AssertStream.h"
#include "Support/ForwardDecls.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/ML/CompiledFrameworkLabel.h"
#include "Support/RCRef.h"
#include <memory>

namespace M {
class MEFFile;
class ModelCollection;
} // namespace M

namespace M::MO {

/// Load all the models from the contents of the specified MEFFile, which
/// holds them compiled to MEF format. Assumes the
/// MemoryBuffer will outlive the ModelCollection.
ErrorOr<std::unique_ptr<ModelCollection>>
loadMGPMEFFile(RCRef<MEFFile> mefFile);

/// This class describes a model overall, including the MEF
/// invokables that implement its phases and metadata about what tensors it
/// takes and returns.
class ModularModel final : public Model {
public:
  ModularModel(std::string name, CompiledFrameworkLabel frameworkLabel,
               const MEFInvokable *initFn, const MEFInvokable *executeFn,
               ArrayRef<QuantTensorSpec> inputSpecs,
               ArrayRef<QuantTensorSpec> outputSpecs, EncodedLocation location)
      : Model(inputSpecs, outputSpecs), name(std::move(name)), initFn(initFn),
        executeFn(executeFn), location(std::move(location)) {
    ASSERT_STREAM(frameworkLabel == CompiledFrameworkLabel::kModularModel,
                  << "Expected " << CompiledFrameworkLabel::kModularModel
                  << " label for model, but got "
                  << frameworkLabel.getAsString());
  }

  ModularModel(std::string name, CompiledFrameworkLabel frameworkLabel,
               const MEFInvokable *initFn, const MEFInvokable *executeFn,
               ArrayRef<QuantTensorSpec> inputSpecs,
               ArrayRef<QuantTensorSpec> outputSpecs,
               ArrayRef<StringRef> inputNames, ArrayRef<StringRef> outputNames,
               EncodedLocation location)
      : Model(inputSpecs, outputSpecs, inputNames, outputNames),
        name(std::move(name)), initFn(initFn), executeFn(executeFn),
        location(std::move(location)) {
    ASSERT_STREAM(frameworkLabel == CompiledFrameworkLabel::kModularModel,
                  << "Expected " << CompiledFrameworkLabel::kModularModel
                  << " label for model, but got "
                  << frameworkLabel.getAsString());
  }

  /// Returns a ModularModel from an mgp.model descriptor.
  static ErrorOr<std::unique_ptr<ModularModel>>
  getFromDescriptor(const PrimitiveDescriptor &descriptor);

  /// Calls the model's setup function using `runtime` and returns the results
  /// immediately. This function does not await the results to be ready.
  LLCL::ErrorDiagnosticOr<std::unique_ptr<ExecutableModel>>
  setup(Runtime &runtime) const override;

  static StringLiteral getEntityName() { return "mgp.model"; }

  /// Returns the name of the model.
  StringRef getName() const override { return name; }

  const MEFInvokable *getInitFn() const { return initFn; }
  const MEFInvokable *getExecuteFn() const { return executeFn; }

  const EncodedLocation &getLocation() const override { return location; }

private:
  std::string name;
  const MEFInvokable *initFn;
  const MEFInvokable *executeFn;

  EncodedLocation location;
};

class ModularExecutableModel final : public ExecutableModel {
public:
  const ModularModel *getModel() const {
    return static_cast<const ModularModel *>(ExecutableModel::getModel());
  }

  /// Runs the model, feeding in `args` as inputs to the model, taking ownership
  /// of the specified `AsyncValueRef`s.  This returns the result of invoking
  /// the model, with the result chain as the first result.
  ///
  /// This method does not `await` on its results.
  SmallVector<AnyAsyncValueRef>
  execute(MutableArrayRef<AnyAsyncValueRef> args) const override;

  ModularExecutableModel(AnyAsyncValueRef chain, AnyAsyncValueRef mgpContext,
                         const ModularModel *model, Runtime &runtime)
      : ExecutableModel(model, runtime), chain(std::move(chain)),
        mgpContext(std::move(mgpContext)) {}

  AnyAsyncValueRef
  convertToGenericTensor(AnyAsyncValueRef unknownTensor) const override {
    return unknownTensor;
  }

  ErrorOrSuccess waitForModelSetup() final;

private:
  AnyAsyncValueRef chain, mgpContext;
};

/// Load all the models from the contents of the specified MEFFile, which
/// holds them compiled to MEF format. Assumes the
/// MemoryBuffer will outlive the ModelCollection.
ErrorOr<std::unique_ptr<ModelCollection>>
loadMGPMEFFile(RCRef<MEFFile> mefFile);

} // namespace M::MO

#endif // GRAPHCOMPILER_TRANSFORMS_MODIALECT_MEFMODEL_H
