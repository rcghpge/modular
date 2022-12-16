//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphRT/BEFExecutor/PrimitiveRegistry.h"
using namespace M;
using namespace Detail;

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// Given a pointer to the start of a BEF array or string attribute, return the
/// length of array or string by decoding the prefix encoded VBR.
size_t M::decodeBEFArrayLength(const uint8_t *dataPtr) {
  // The attribute's kind is between the VBR and the actual value, so we have to
  // make sure to subtract that offset (hence the -2).
  const uint8_t *lenPtr = dataPtr - 2;
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

/// Parse a VBR and increment `workptr` in the process, so we can skip over
/// the VBR values. This is a little weird, being a reference to a pointer,
/// but it's to simplify the parsing use cases above by allowing us to
/// increment the pointer that's been passed-in.
size_t M::parseVBR(uint8_t *&workptr) {
  size_t result;
  uint8_t nextByte = *workptr++;
  // Mask just the low 7 bits of the first byte to start the integer
  result = nextByte & 127;
  // While the high bit is non-zero, get the next byte, increment the work
  // pointer, and pack the 7 data bits into the result integer.
  while ((nextByte & 128) != 0) {
    nextByte = *workptr++;
    result = (result << 7) | size_t(nextByte & 127);
  }

  return result;
}

//===----------------------------------------------------------------------===//
// BEFListAttr implementation
//===----------------------------------------------------------------------===//

BEFListAttr::Iterator &BEFListAttr::Iterator::operator+=(
    BEFListAttr::Iterator::Base::difference_type n) {
  for (auto i = 0; i < n; ++i) {
    // We have to increment our current pointer past the VBR for each element.
    (void)parseVBR(ptr);
    ++iterIdx;
  }
  return *this;
}

BEFDynamicAttr BEFListAttr::Iterator::operator*() const {
  auto *workPtr = ptr;
  return BEFDynamicAttr(basePtr - parseVBR(workPtr), attrSection);
}

//===----------------------------------------------------------------------===//
// BEFDictionaryAttr::Iterator
//===----------------------------------------------------------------------===//

BEFDictionaryAttr::Iterator &BEFDictionaryAttr::Iterator::operator+=(
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

BEFDictionaryAttr::Element BEFDictionaryAttr::Iterator::operator*() const {
  uint8_t *workptr = ptr;
  size_t nameOffset = parseVBR(workptr);
  size_t dataOffset = parseVBR(workptr);
  return Element{basePtr, nameOffset, dataOffset, attrSection};
}

//===----------------------------------------------------------------------===//
// BEFDictionaryAttr implementation
//===----------------------------------------------------------------------===//

std::optional<BEFDynamicAttr> BEFDictionaryAttr::find(StringRef name) const {
  auto found = llvm::find_if(
      *this, [&](const Element &elt) { return elt.getName() == name; });

  if (found != end())
    return (*found).getValue();

  return std::nullopt;
}
