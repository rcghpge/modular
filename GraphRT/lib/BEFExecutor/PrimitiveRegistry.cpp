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
  StringMap<PrimitiveImplFn> prims;
};
} // namespace M

//===----------------------------------------------------------------------===//
// PrimitiveRegistry
//===----------------------------------------------------------------------===//

PrimitiveRegistry::PrimitiveRegistry() : impl(std::make_unique<Impl>()) {}
PrimitiveRegistry::~PrimitiveRegistry() {}

void PrimitiveRegistry::add(StringRef name, PrimitiveImplFn fn) {
  bool added = impl->prims.try_emplace(name, fn).second;
  (void)added;
  assert(added && "re-registered existing name for a primitive");
}

auto PrimitiveRegistry::get(StringRef name) const -> PrimitiveImplFn {
  auto it = impl->prims.find(name);
  return it == impl->prims.end() ? PrimitiveImplFn() : it->second;
}
