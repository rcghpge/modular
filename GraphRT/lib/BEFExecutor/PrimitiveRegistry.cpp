//===- PrimitiveRegistry.cpp ----------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphRT/BEFExecutor/PrimitiveRegistry.h"
#include "llvm/ADT/StringRef.h"

using namespace M;

//===----------------------------------------------------------------------===//
// PrimitiveRegistry::Impl
//===----------------------------------------------------------------------===//

namespace M {
struct PrimitiveRegistry::Impl {
  bool isFinalized = false;
  std::vector<std::pair<StaticString, PrimitiveImplFn>> prims;
};
} // namespace M

//===----------------------------------------------------------------------===//
// PrimitiveRegistry
//===----------------------------------------------------------------------===//

PrimitiveRegistry::PrimitiveRegistry() : impl(std::make_unique<Impl>()) {}
PrimitiveRegistry::~PrimitiveRegistry() {}

void PrimitiveRegistry::add(StaticString name, PrimitiveImplFn fn) {
  impl->prims.emplace_back(name, fn);
}

void PrimitiveRegistry::finalize() {
  // Sort by name here so we can lookup using a binary search by name later.
  llvm::sort(impl->prims, [](const auto &lhs, const auto &rhs) {
    return StringRef(lhs.first.value) < StringRef(rhs.first.value);
  });
  auto duplicate =
      std::adjacent_find(impl->prims.begin(), impl->prims.end(),
                         [](const auto &lhs, const auto &rhs) {
                           // To check if they're equal, we can compare the
                           // pointers because they are both StaticStrings.
                           return lhs.first.value == rhs.first.value;
                         });
  if (duplicate != impl->prims.end())
    llvm::report_fatal_error("Found a duplicate in the primitive registry.");

  impl->isFinalized = true;
}

auto PrimitiveRegistry::get(StringRef name) const -> PrimitiveImplFn {
  if (!impl->isFinalized)
    llvm::report_fatal_error(
        "PrimitiveRegistry must be finalized before attempting to "
        "`get` an impl from it.");

  auto it = std::lower_bound(impl->prims.begin(), impl->prims.end(), name,
                             [](const auto &pair, StringRef name) {
                               return StringRef(pair.first.value) < name;
                             });

  // std::lower_bound returns the first element that is >= the searched element.
  // Therefore, we have to check if the value is equal (which is what we want).
  return it != impl->prims.end() && StringRef(it->first.value) == name
             ? it->second
             : PrimitiveImplFn();
}
