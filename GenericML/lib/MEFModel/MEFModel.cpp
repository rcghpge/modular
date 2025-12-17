//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/MEFModel/MEFModel.h"
#include "GenericML/MGPPrimitives/MGPPrimitives.h"
#include "GenericML/Support/MEFTypeUtils.h"
#include "GenericML/Support/MGPModel.h"
#include "MLRT/GraphRT/MEFExecutor/MEFInvokable.h"
#include "MLRT/GraphRT/MEFExecutor/PrimitiveRegistry.h"
#include "MLRT/GraphRT/Primitives.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/ML/CompiledFrameworkLabel.h"
#include "llvm/ADT/SmallVector.h"

using namespace M;
using namespace M::MO;

namespace M::MO {
struct ModularExecutableModel;
struct ModularModel;
} // namespace M::MO

namespace M {
class MEFFile;
class ModelCollection;

} // namespace M

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

namespace /*M::*/ MO {

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

} // namespace MO

} // namespace M

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
  if (auto err = moduleInv->walkPrimitives<ModularModel>(
          [&](auto &&m) { models.push_back(std::forward<decltype(m)>(m)); }))
    return err.takeError();

  return std::make_unique<M::ModelCollection>(std::move(models));
}

} // namespace M
