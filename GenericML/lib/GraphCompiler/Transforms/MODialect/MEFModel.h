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

#include "AsyncRT/ForwardDecls.h"
#include "GenericML/Support/MGPModel.h"
#include "GenericML/Support/Model.h"
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

class ModularExecutableModel;

/// Load all the models from the contents of the specified MEFFile, which
/// holds them compiled to MEF format. Assumes the
/// MemoryBuffer will outlive the ModelCollection.
ErrorOr<std::unique_ptr<ModelCollection>>
loadMGPMEFFile(RCRef<MEFFile> mefFile);

/// This class describes a model overall, including the MEF
/// invokables that implement its phases and metadata about what tensors it
/// takes and returns.
struct ModularModel final : public MGPModelImpl<ModularModel> {
  using MGPModelImpl::MGPModelImpl;
  using ExecutableModelType = ModularExecutableModel;

  static constexpr CompiledFrameworkLabel kFrameworkLabel{
      CompiledFrameworkLabel::kModularModel};
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
                         SmallVector<AnyAsyncValueRef> &&devices,
                         const Model *model, Runtime &runtime)
      : ExecutableModel(model, runtime), chain(std::move(chain)),
        mgpContext(std::move(mgpContext)), devices(std::move(devices)) {}

  AnyAsyncValueRef
  convertToGenericTensor(AnyAsyncValueRef unknownTensor) const override {
    return unknownTensor;
  }

  ErrorOrSuccess waitForModelSetup() final;

private:
  AnyAsyncValueRef chain, mgpContext;
  SmallVector<AnyAsyncValueRef> devices;
};

/// Load all the models from the contents of the specified MEFFile, which
/// holds them compiled to MEF format. Assumes the
/// MemoryBuffer will outlive the ModelCollection.
ErrorOr<std::unique_ptr<ModelCollection>>
loadMGPMEFFile(RCRef<MEFFile> mefFile);

} // namespace M::MO

#endif // GRAPHCOMPILER_TRANSFORMS_MODIALECT_MEFMODEL_H
