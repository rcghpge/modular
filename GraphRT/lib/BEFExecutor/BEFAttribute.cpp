//===- BEFAttribute.cpp ---------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphRT/BEFExecutor/PrimitiveRegistry.h"
using namespace M;
using namespace Detail;

//===----------------------------------------------------------------------===//
// BEFAttr<Dictionary>::Iterator
//===----------------------------------------------------------------------===//

BEFAttr<Dictionary>::Iterator &BEFAttr<Dictionary>::Iterator::operator+=(
    BEFAttr<Dictionary>::Iterator::Base::difference_type n) {
  for (auto i = 0; i < n; ++i) {
    // We have to increment our current pointer past the VBRs and the inline
    // name.
    (void)parseVBR(ptr);
    (void)parseVBR(ptr);
    ++iterIdx;
  }
  return *this;
}

bool BEFAttr<Dictionary>::Iterator::operator==(
    const BEFAttr<Dictionary>::Iterator &rhs) const {
  assert(basePtr == rhs.basePtr &&
         "Comparing iterators from different dictionaries!");
  return iterIdx == rhs.iterIdx && dictLen == rhs.dictLen;
}

BEFAttr<Dictionary>::Element BEFAttr<Dictionary>::Iterator::operator*() const {
  uint8_t *workptr = ptr;
  size_t nameOffset = parseVBR(workptr);
  size_t dataOffset = parseVBR(workptr);
  return Element{basePtr, nameOffset, dataOffset};
}

//===----------------------------------------------------------------------===//
// BEFAttr<Dictionary>
//===----------------------------------------------------------------------===//

Optional<BEFDynamicAttr> BEFAttr<Dictionary>::find(StringRef name) const {
  auto found = llvm::find_if(
      *this, [&](const Element &elt) { return elt.getName() == name; });

  if (found != end())
    return (*found).getValue();

  return None;
}

size_t BEFAttr<Dictionary>::parseVBR(uint8_t *&workptr) {
  size_t result;
  uint8_t nextByte = *workptr++;
  result = nextByte & 127;
  while ((nextByte & 128) != 0) {
    nextByte = *workptr++;

    result = (result << 7) | size_t(nextByte & 127);
  }

  return result;
}
