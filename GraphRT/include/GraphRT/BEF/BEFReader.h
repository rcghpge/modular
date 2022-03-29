//===- GraphRT/BEF/BEFReader.h --------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// BEFReader is decoding logic shared between BEF2MLIR and the BEFExecutor.
//
//===----------------------------------------------------------------------===//

#ifndef GRAPHRT_BEF_BEFREADER_H
#define GRAPHRT_BEF_BEFREADER_H

#include "GraphRT/BEF/BEFEncoding.h"
#include "Support/Error.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/FunctionExtras.h"
#include "llvm/Support/MathExtras.h"

namespace M {

/// This helper class implements low level binary parsing logic for a byte
/// range.  The general protocol for failable parsing routines is that they
/// return false and fill in their operand on success.  On failure, they emit an
/// error to the `errorFunction` and return true.
class BEFReader {
public:
  explicit BEFReader(ArrayRef<uint8_t> contents,
                     llvm::unique_function<void(Error) const> errorFunction)
      : contents(contents), errorFunction(std::move(errorFunction)) {}

  // Return true if we've read the entire file.
  bool empty() const { return contents.empty(); }

  ArrayRef<uint8_t> getContents() const { return contents; }
  const uint8_t *data() const { return contents.data(); }

  /// Reset the contents that this reader will parse from to the specified
  /// range.
  void setContents(ArrayRef<uint8_t> data) { contents = data; }

  bool emitError(Error message) const {
    errorFunction(std::move(message));
    return true;
  }

  /// Parse a single byte from the stream.
  bool parseByte(uint8_t &value) {
    if (contents.empty())
      return emitError("ran out of bytes");
    value = contents.front();
    contents = contents.drop_front();
    return false;
  }

  /// Parse the byte ending at the specified offset.
  bool parseByteAt(size_t offset, uint8_t &value) const {
    if (contents.empty())
      return emitError("ran out of bytes");
    if (offset >= contents.size())
      return emitError("offset out of range");
    value = contents[offset];
    return false;
  }

  /// Parse two bytes in little endian format from the stream.
  bool parseInt2(uint16_t &value) {
    uint8_t a, b;
    if (parseByte(a) || parseByte(b))
      return true;
    value = uint16_t(a) | (uint16_t(b) << 8);
    return false;
  }

  /// Parse four bytes in little endian format from the stream.
  bool parseInt4(uint32_t &value) {
    uint16_t a, b;
    if (parseInt2(a) || parseInt2(b))
      return true;
    value = uint32_t(a) | (uint32_t(b) << 16);
    return false;
  }

  /// Parse N int4 values as an array.
  bool parseInt4Array(size_t numElements, SmallVectorImpl<uint32_t> &result) {
    while (numElements--) {
      result.push_back(0);
      if (parseInt4(result.back()))
        return true;
    }
    return false;
  }

  /// Parse a VBR encoded integer from the byte stream.
  bool parseVBR(size_t &result) {
    uint8_t nextByte;
    if (parseByte(nextByte))
      return true;

    result = nextByte & 127;
    while ((nextByte & 128) != 0) {
      if (parseByte(nextByte))
        return true;

      result = (result << 7) | size_t(nextByte & 127);
    }
    return false;
  }

  /// Parse an inverted VBR that ends at the specified offset.
  bool parseInvertedVBRAt(size_t offset, size_t &result) const {
    result = 0;
    uint8_t lastByte;
    do {
      if (offset == 0)
        return emitError("invalid inverted VBR");
      lastByte = contents[offset - 1];
      result = (result << 7) | (lastByte & 0x7F);
      --offset;
    } while (lastByte & 0x80);
    return false;
  }

  /// Skip over bytes until reaching the specified alignment.
  bool alignTo(unsigned alignment) {
    if (!llvm::isPowerOf2_32(alignment))
      return emitError("tried to align to non-power-of-two");

    uint8_t padding;
    while (uintptr_t(contents.data()) & (uintptr_t(alignment) - 1))
      if (parseByte(padding))
        return true;

    return false;
  }

  /// Take some number of bytes from the input file and put them into result.
  bool parseBytes(size_t length, ArrayRef<uint8_t> &result) {
    if (length > contents.size())
      return emitError("missing block of data");
    result = contents.take_front(length);
    contents = contents.drop_front(length);
    return false;
  }

  /// Ignore the first `length` bytes in this reader.
  bool dropBytes(size_t length) {
    ArrayRef<uint8_t> unused;
    return parseBytes(length, unused);
  }

  /// Parse a NUL terminated string into the specified StringRef, not including
  /// the NUL terminator.
  bool parseNULTerminatedString(StringRef &result) {
    auto *start = (const char *)contents.data();
    auto nul = (const char *)memchr(start, 0, contents.size());
    if (!nul)
      return emitError("malformed NUL terminated string");
    result = StringRef(start, nul - start);
    contents = contents.drop_front(result.size() + 1);
    return false;
  }

  /// Parse a NUL terminated string into the specified StringRef, not including
  /// the NUL terminator.
  bool parseNULTerminatedStringAt(size_t position, StringRef &result) const {
    if (position > contents.size())
      return emitError("invalid offset to string");
    auto buffer = contents.drop_front(position);
    auto *start = (const char *)buffer.data();
    auto nul = (const char *)memchr(start, 0, buffer.size());
    if (!nul)
      return emitError("malformed NUL terminated string");
    result = StringRef(start, nul - start);
    return false;
  }

  /// Parse a section header and the alignment metadata, returning the kind of
  /// section in `sectionID` and the contents of the section in `sectionData`.
  bool readSection(uint8_t &sectionID, ArrayRef<uint8_t> &sectionData) {
    size_t length;
    if (parseByte(sectionID) || parseVBR(length))
      return true;
    if (sectionID == (unsigned)BEF::SectionID::kInvalid)
      return emitError("invalid section ID");

    // The low bit of the size is a boolean indicating whether there is an
    // alignment byte + padding present.
    if ((length & 1) != 0) {
      // If there is an alignment byte, read the alignment byte, and skip over
      // padding bytes until we reach the desired alignment.
      uint8_t alignment;
      if (parseByte(alignment) || alignTo(alignment))
        return true;
    }

    // Okay, the returned data is whatever is left now.  We shift by one to
    // eliminate the alignment bit.
    return parseBytes(length >> 1, sectionData);
  }

private:
  ArrayRef<uint8_t> contents;
  llvm::unique_function<void(Error) const> errorFunction;
};

} // namespace M

#endif // GRAPHRT_BEF_BEFREADER_H