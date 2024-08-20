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

#include "GenericML/Support/MGPModel.h"
#include "Support/ForwardDecls.h"
#include "Support/ML/CompiledFrameworkLabel.h"
#include "Support/RCRef.h"
#include <memory>

namespace M::MO {
struct ModularExecutableModel;
struct ModularModel;
} // namespace M::MO

namespace M {
class MEFFile;
class ModelCollection;

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

  static constexpr CompiledFrameworkLabel kFrameworkLabel{
      CompiledFrameworkLabel::kModularModel};
};

struct ModularExecutableModel final
    : public MGPExecutableModelImpl<ModularExecutableModel> {
  using MGPExecutableModelImpl::MGPExecutableModelImpl;
};

/// Load all the models from the contents of the specified MEFFile, which
/// holds them compiled to MEF format. Assumes the
/// MemoryBuffer will outlive the ModelCollection.
ErrorOr<std::unique_ptr<ModelCollection>>
loadMGPMEFFile(RCRef<MEFFile> mefFile);

} // namespace M::MO

#endif // GRAPHCOMPILER_TRANSFORMS_MODIALECT_MEFMODEL_H
