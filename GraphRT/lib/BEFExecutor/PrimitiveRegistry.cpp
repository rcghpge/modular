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
  std::vector<std::pair<StringLiteral, PrimitiveImplFn>> prims;
};
} // namespace M

//===----------------------------------------------------------------------===//
// PrimitiveRegistry
//===----------------------------------------------------------------------===//

PrimitiveRegistry::PrimitiveRegistry() : impl(std::make_unique<Impl>()) {}
PrimitiveRegistry::~PrimitiveRegistry() {}

void PrimitiveRegistry::add(StringLiteral name, PrimitiveImplFn fn) {
  impl->prims.emplace_back(name, fn);
}

void PrimitiveRegistry::finalize() {
  // Sort by name here so we can lookup using a binary search by name later.
  llvm::sort(impl->prims, [](const auto &lhs, const auto &rhs) {
    return lhs.first < rhs.first;
  });
  auto duplicate = std::adjacent_find(impl->prims.begin(), impl->prims.end());
  if (duplicate != impl->prims.end())
    llvm::report_fatal_error("Found a duplicate in the primitive registry.");

  impl->isFinalized = true;
}

auto PrimitiveRegistry::get(StringRef name) const -> PrimitiveImplFn {
  if (!impl->isFinalized)
    llvm::report_fatal_error(
        "PrimitiveRegistry must be finalized before attempting to "
        "`get` an impl from it.");

  auto it = std::lower_bound(
      impl->prims.begin(), impl->prims.end(), name,
      [](const auto &pair, StringRef name) { return pair.first < name; });

  // std::lower_bound returns the first element that is >= the searched element.
  // Therefore, we have to check if the value is equal (which is what we want).
  return it != impl->prims.end() && it->first == name ? it->second
                                                      : PrimitiveImplFn();
}

//===----------------------------------------------------------------------===//
// ADD_PRIMITIVE implementation logic
//===----------------------------------------------------------------------===//

/// Given a pointer to the start of a BEF array or string attribute, return the
/// length of array or string by decoding the prefix encoded VBR.
size_t M::decodeBEFArrayLength(const void *dataPtr) {
  // The attribute's kind is between the VBR and the actual value, so we have to
  // make sure to subtract that offset (hence the -2).
  const uint8_t *lenPtr = static_cast<const uint8_t *>(dataPtr) - 2;
  size_t size = 0;
  bool hasMoreBytes;
  do {
    hasMoreBytes = (lenPtr[0] & 0x80) != 0;
    size <<= 7;
    size |= lenPtr[0] & 0x7F;
    --lenPtr;
  } while (hasMoreBytes);
  return size;
}
