//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Kernels/WeightsRegistry.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/bit.h"

using namespace M;

static constexpr int kResultSuccess = 0;

ErrorOr<WeightsRegistryRef>
WeightsRegistry::create(ArrayRef<const char *> names,
                        ArrayRef<const std::byte *> data) {
  if (names.size() != data.size())
    return Error("expected equal length names (" + Twine(names.size()) +
                 ") and data (" + Twine(data.size()) + ") arrays");

  void *weights;
  if (create_weights_registry(&weights, (void *)names.data(),
                              (void *)data.data(),
                              data.size()) == kResultSuccess) {
    return WeightsRegistryRef::create(
        MojoWeightsRegistryPtr(weights, destroy_weights_registry));
  }
  return Error("unable to create weights registry");
}

ErrorOr<const std::byte *> WeightsRegistry::operator[](const char *name) const {
  void *data;
  if (weights_registry_lookup(&data, mojoWeights.get(),
                              const_cast<char *>(name)) == kResultSuccess)
    return llvm::bit_cast<const std::byte *>(data);

  return Error("unable to look up " + Twine(name));
}
