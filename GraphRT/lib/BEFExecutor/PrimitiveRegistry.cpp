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
  // This pair holds a char * instead of a StaticString because the
  // StaticString's const type makes the pair non-move-assignable. This is
  // somewhat less than ideal, but at least it's hidden away.
  std::vector<std::pair<char *, PrimitiveImplFn>> prims;
};
} // namespace M

//===----------------------------------------------------------------------===//
// PrimitiveRegistry
//===----------------------------------------------------------------------===//

PrimitiveRegistry::PrimitiveRegistry() : impl(std::make_unique<Impl>()) {}
PrimitiveRegistry::~PrimitiveRegistry() {}

void PrimitiveRegistry::add(StaticString name, PrimitiveImplFn fn) {
  // Use std::lower_bound to insert so the vector stays sorted as we insert
  // items.
  auto insertPt = std::lower_bound(
      impl->prims.begin(), impl->prims.end(), name,
      [](const auto &element, const StaticString &value) {
        return StringRef(element.first) < StringRef(value.value);
      });
  // std::lower_bound returns the first element >= to the searched element, so
  // we can see if we added something new by checking if we found an element
  // whose name is equal. Because it's a static string, we can just compare the
  // pointer values.
  bool added = insertPt == impl->prims.end() || insertPt->first != name.value;
  (void)added;
  assert(added && "re-registered existing name for a primitive");
  // Finally, we are registering a primitive, so insert it.
  impl->prims.emplace(insertPt, const_cast<char *>(name.value), fn);
}

auto PrimitiveRegistry::get(StringRef name) const -> PrimitiveImplFn {
  auto it = std::lower_bound(impl->prims.begin(), impl->prims.end(), name,
                             [](const auto &pair, StringRef name) {
                               return StringRef(pair.first) < name;
                             });

  // std::lower_bound returns the first element that is >= the searched element.
  // Therefore, we have to check if the value is equal (which is what we want).
  return it != impl->prims.end() && StringRef(it->first) == name
             ? it->second
             : PrimitiveImplFn();
}
