//===- GraphRT/BEF/BEFEncoding.h --------------------------------*- C++ -*-===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file defines magic numbers and other structures related to the BEF file
// format.  See the BinaryExecutableFormat.md file for more information.
//
//===----------------------------------------------------------------------===//

#ifndef GRAPHRT_BEF_BEFENCODING_H
#define GRAPHRT_BEF_BEFENCODING_H

#include "llvm/ADT/StringRef.h"

namespace M {

namespace BEF {

enum {
  // Defined by the BEF_FILE production.
  MAGIC1 = 0xBE,
  MAGIC2 = 0xF0,
  VERSION0 = 0,

  // This is an arbitrary value used to fill out alignment padding.
  ALIGNMENT_BYTE = 0xCB,

  // This indicates whether a section has an explicit alignment.
  SECTION_ALIGNMENT_FLAG = 0x01,

  // Primitives in BEF are 4-byte aligned.
  PRIMITIVE_ENTRY_ALIGNMENT = 4,
}; // anonymous enum

/// This bitfield encodes into a single byte metadata about the format of
/// emitted attributes.
enum class AttributeKind {
  kInvalid = 0,
  kI1,
  kI8,
  kI16,
  kI32,
  kI64,
  kF32,
  kF64,
  kString,
  kArrayType = 0x80
};

enum class SectionID {
  INVALID_SECTION = 0,

  LOCATION_FILENAMES = 1, // Strings for each filename used by a location.
  LOCATION_POSITIONS = 2, // Detailed information for locations

  STRINGS = 3, // All of the string data.
  ATTRIBUTES = 4,
  PRIMITIVE_NAMES = 5,
  BEF_FUNCTIONS = 6,
  BEF_FUNCTION_INDEX = 7,

  ATTRIBUTE_KIND_MD = 8,
  STRINGS_MD = 9,           // Strings for metadata
  ATTRIBUTE_NAME_MD = 10,   // Table of primitive attribute names.
  MLIRTYPE_STRINGS_MD = 11, // String table of MLIR types.
  MLIRTYPE_MD = 12,
};

/// Return true if the magic number makes this look like a BEF file.
inline bool hasBEFMagicNumber(StringRef contents) {
  return contents.size() > 2 && uint8_t(contents[0]) == MAGIC1 &&
         uint8_t(contents[1]) == MAGIC2;
}

} // namespace BEF
} // namespace M

#endif // GRAPHRT_MLIR_TO_BEF_H