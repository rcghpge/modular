//===- GraphRT/BEF/BEFEncoding.h ------------------------------------------===//
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
enum class AttributeKind : uint8_t {
  kInvalid = 0,
  /// The primitive types store their types as a simple function of their
  /// bitwidth for simplicity in getting the AttributeKind from a type. Floats
  /// have the low bit set to indicate that they're floats.
  kI1 = 1,
  kI8 = 8,
  kI16 = 16,
  kI32 = 32,
  kI64 = 64,
  kF32 = 33,
  kF64 = 65,
  /// There is no special meaning to this value, it's 66 because kF64 is 65.
  kString = 66,
  /// This is a bitmask, meant to be applied to one of the basic types (I*, F*)
  kArrayType = 0x80
};

/// Given an AttributeKind for an element, return the AttributeKind for an array
/// of that element.
static inline AttributeKind getArrayOf(AttributeKind elementKind) {
  assert(elementKind != AttributeKind::kString &&
         elementKind != AttributeKind::kArrayType &&
         "Cannot make arrays of arrays or arrays of strings");
  return AttributeKind(uint8_t(elementKind) |
                       uint8_t(AttributeKind::kArrayType));
}

enum class SectionID {
  INVALID_SECTION = 0,

  LOCATION_FILENAMES = 1, // Strings for each filename used by a location.
  LOCATION_POSITIONS = 2, // Detailed information for locations

  STRINGS = 3, // All of the string data.
  ATTRIBUTES = 4,
  PRIMITIVE_NAMES = 5,
  INVOKABLES_BODIES = 6,
  INVOKABLES_INDEX = 7,

  ATTRIBUTE_INDEX_MD = 8,
  STRINGS_MD = 9,           // Strings for metadata
  ATTRIBUTE_NAME_MD = 10,   // Table of primitive attribute names.
  MLIRTYPE_STRINGS_MD = 11, // String table of MLIR types.
  MLIRTYPE_MD = 12,
};

/// Return true if the magic number makes this look like a BEF file.
inline bool hasBEFMagicNumber(llvm::StringRef contents) {
  return contents.size() > 2 && uint8_t(contents[0]) == MAGIC1 &&
         uint8_t(contents[1]) == MAGIC2;
}

/// Provide a mapping from C++ types to the corresponding attribute kinds.
template <typename CXXType>
struct AttrKindForCXXType {
  // Default mapping allows a static_assert instead of template substitution
  // failure to catch invalid cases.
  static constexpr AttributeKind kind = AttributeKind::kInvalid;
};

#define DECLARE_ATTR_MAPPING(KIND, CXXTYPE)                                    \
  template <>                                                                  \
  struct AttrKindForCXXType<CXXTYPE> {                                         \
    static constexpr AttributeKind kind = AttributeKind::KIND;                 \
  };
DECLARE_ATTR_MAPPING(kI1, bool);
DECLARE_ATTR_MAPPING(kI8, int8_t);
DECLARE_ATTR_MAPPING(kI8, uint8_t);
DECLARE_ATTR_MAPPING(kI16, int16_t);
DECLARE_ATTR_MAPPING(kI16, uint16_t);
DECLARE_ATTR_MAPPING(kI32, int32_t);
DECLARE_ATTR_MAPPING(kI32, uint32_t);
DECLARE_ATTR_MAPPING(kI64, int64_t);
DECLARE_ATTR_MAPPING(kI64, uint64_t);
DECLARE_ATTR_MAPPING(kF32, float);
DECLARE_ATTR_MAPPING(kF64, double);
#undef DECLARE_ATTR_MAPPING

} // namespace BEF
} // namespace M

#endif // GRAPHRT_MLIR_TO_BEF_H