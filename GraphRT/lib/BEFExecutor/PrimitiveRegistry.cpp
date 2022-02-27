//===- PrimitiveRegistry.cpp ----------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphRT/BEFExecutor/PrimitiveRegistry.h"
#include "llvm/ADT/StringMap.h"
using namespace M;
using llvm::StringMap;

//===----------------------------------------------------------------------===//
// PrimitiveRegistry::Impl
//===----------------------------------------------------------------------===//

namespace M {
struct PrimitiveRegistry::Impl {
  // TODO: We should move away from using a StringMap here to a sorted vector.
  //   https://github.com/modularml/modular/issues/12
  // This would be better for code size and reduce memory use.
  StringMap<PrimitiveImplFn> prims;
};
} // namespace M

//===----------------------------------------------------------------------===//
// PrimitiveRegistry
//===----------------------------------------------------------------------===//

PrimitiveRegistry::PrimitiveRegistry() : impl(std::make_unique<Impl>()) {}
PrimitiveRegistry::~PrimitiveRegistry() {}

void PrimitiveRegistry::add(StaticString name, PrimitiveImplFn fn) {
  bool added = impl->prims.try_emplace(name.value, fn).second;
  (void)added;
  assert(added && "re-registered existing name for a primitive");
}

auto PrimitiveRegistry::get(StringRef name) const -> PrimitiveImplFn {
  auto it = impl->prims.find(name);
  return it == impl->prims.end() ? PrimitiveImplFn() : it->second;
}
