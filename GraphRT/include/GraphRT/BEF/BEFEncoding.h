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

#include "Support/LLVMForwardDecls.h"
#include "llvm/ADT/StringRef.h"

namespace M {

namespace BEF {

//===----------------------------------------------------------------------===//
// General BEF constants
//===----------------------------------------------------------------------===//

enum {
  // Defined by the BEF_FILE production.
  kMagic1 = 0xBE,
  kMagic2 = 0xF0,
  kVersion0 = 0,

  // This is an arbitrary value used to fill out alignment padding.
  kAlignmentByte = 0xCB,

  // This indicates whether a section has an explicit alignment.
  kSectionAlignmentFlag = 0x01,

  // Primitives in BEF are 4-byte aligned.
  kPrimitiveEntryAlignment = 4,
}; // anonymous enum

/// Return true if the magic number makes this look like a BEF file.
inline bool hasBEFMagicNumber(llvm::StringRef contents) {
  return contents.size() > 2 && uint8_t(contents[0]) == kMagic1 &&
         uint8_t(contents[1]) == kMagic2;
}

//===----------------------------------------------------------------------===//
// BEF section constants
//===----------------------------------------------------------------------===//

enum class SectionID {
  kInvalid = 0,

  kLocationFilenames = 1, // Strings for each filename used by a location.
  kLocationPositions = 2, // Detailed information for locations

  kStrings = 3, // All of the string data.
  kAttributes = 4,
  kPrimitiveNames = 5,
  kInvokablesBodies = 6,
  kInvokablesIndex = 7,

  kAttributeIndexMD = 8,
  kStringsMD = 9,          // Strings for metadata
  kAttributeNamesMD = 10,  // Table of primitive attribute names.
  kMLIRTypeStringsMD = 11, // String table of MLIR types.
  kMLIRTypeMD = 12,
};

//===----------------------------------------------------------------------===//
// BEF attribute constants and helpers
//===----------------------------------------------------------------------===//

/// A two-bit classification.
enum class AttrCategory : uint8_t { Float, Signed, Unsigned };

/// This bitfield encodes into a single byte metadata about the format of
/// emitted attributes.
enum class AttributeKind : uint8_t {
  kInvalid = 0,

  /// This is a bitmask, when combined with one of the basic types, indicates
  /// an array of that type.
  mArrayType = 1,

  /// The primitive types are encoded with the low bit reserved for "is array",
  /// the next two bits being their category, and the next 2 bits being the
  /// number of bytes (in log form).
  kS8 = (0 << 3) | uint8_t(AttrCategory::Signed) << 1,
  kU8 = (0 << 3) | uint8_t(AttrCategory::Unsigned) << 1,
  kS16 = (1 << 3) | uint8_t(AttrCategory::Signed) << 1,
  kU16 = (1 << 3) | uint8_t(AttrCategory::Unsigned) << 1,
  kS32 = (2 << 3) | uint8_t(AttrCategory::Signed) << 1,
  kU32 = (2 << 3) | uint8_t(AttrCategory::Unsigned) << 1,
  kF32 = (2 << 3) | uint8_t(AttrCategory::Float) << 1,
  kS64 = (3 << 3) | uint8_t(AttrCategory::Signed) << 1,
  kU64 = (3 << 3) | uint8_t(AttrCategory::Unsigned) << 1,
  kF64 = (3 << 3) | uint8_t(AttrCategory::Float) << 1,

  // There is no special encoding to remaining enum values, they are just
  // numbered to be out of the way of the others.
  kBool = 1 << 5,
  kString = 2 << 5,
  kDictionary = 3 << 5,
  kList = 4 << 5,
  kType = 5 << 5,
};

static inline AttrCategory getCategory(AttributeKind kind) {
  return (AttrCategory)((uint8_t(kind) >> 1) & 3);
}

static inline unsigned getWidthInBits(AttributeKind kind) {
  return (1 << ((uint8_t)kind >> 3)) * 8;
}

/// Given an AttributeKind for an element, return the AttributeKind for an array
/// of that element.
static inline constexpr AttributeKind getArrayOf(AttributeKind elementKind) {
  assert(elementKind <= AttributeKind::kBool &&
         "cannot make arrays of arrays or arrays of strings");
  return AttributeKind(uint8_t(elementKind) |
                       uint8_t(AttributeKind::mArrayType));
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
DECLARE_ATTR_MAPPING(kBool, bool);
DECLARE_ATTR_MAPPING(kS8, int8_t);
DECLARE_ATTR_MAPPING(kU8, uint8_t);
DECLARE_ATTR_MAPPING(kS16, int16_t);
DECLARE_ATTR_MAPPING(kU16, uint16_t);
DECLARE_ATTR_MAPPING(kS32, int32_t);
DECLARE_ATTR_MAPPING(kU32, uint32_t);
DECLARE_ATTR_MAPPING(kS64, int64_t);
DECLARE_ATTR_MAPPING(kU64, uint64_t);
DECLARE_ATTR_MAPPING(kF32, float);
DECLARE_ATTR_MAPPING(kF64, double);
DECLARE_ATTR_MAPPING(kString, StringRef);
#undef DECLARE_ATTR_MAPPING

//===----------------------------------------------------------------------===//
// BEF type constants and helpers
//===----------------------------------------------------------------------===//

enum class TypeKind : uint8_t {
  /// Note that many of these should be interpreted as type classes, i.e. the
  /// enum represents a mask, not a concrete enumerator.  These enum values are
  /// prefixed with `m` to designate this.
  kInvalid = 0,

  //------ encoding for ordinary primitives ----------------------------------//

  /// Bit 0 indicates if type is complex.
  mComplex = 1,

  /// Bits 1 and 2 encode primitive category.
  mOther = 0 << 1,
  mSigned = 1 << 1,
  mUnsigned = 2 << 1,
  mFloat = 3 << 1,

  /// Integer and floating point types.  The upper four bits for
  /// float/signed/unsigned are the # of bytes - 1.
  kS8 = mSigned,
  kU8 = mUnsigned,
  kS16 = (1 << 3) | mSigned,
  kU16 = (1 << 3) | mUnsigned,
  kS32 = (3 << 3) | mSigned,
  kU32 = (3 << 3) | mUnsigned,
  kS64 = (7 << 3) | mSigned,
  kU64 = (7 << 3) | mUnsigned,
  kS128 = (15 << 3) | mSigned,
  kU128 = (15 << 3) | mUnsigned,

  // bit 3 encodes whether the float is IEEE or not.
  mIEEEFloat = (1 << 3) | mFloat, // only use it as mask

  kF16 = mIEEEFloat,
  kF32 = (1 << 4) | mIEEEFloat,
  kF64 = (2 << 4) | mIEEEFloat,
  kF128 = (3 << 4) | mIEEEFloat,

  kBF16 = mFloat,
  kF24 = (1 << 4) | mFloat,
  kF80 = (2 << 4) | mFloat,
  kTF32 = (3 << 4) | mFloat,

  //------ encodings for other types ----------------------------------//
  kBool = (1 << 3) | mOther,
  kTensor = (2 << 3) | mOther,    // Has following bytes.
  kQuantized = (3 << 3) | mOther, // Has following bytes.
};

inline TypeKind operator|(TypeKind a, TypeKind b) {
  return static_cast<TypeKind>(static_cast<uint8_t>(a) |
                               static_cast<uint8_t>(b));
}

/// Provide a mapping from C++ types to the corresponding type kinds.
template <typename CXXType>
struct TypeKindForCXXType {
  // Default mapping allows a static_assert instead of template substitution
  // failure to catch invalid cases.
  static constexpr TypeKind kind = TypeKind::kInvalid;
};

#define DECLARE_TYPE_MAPPING(KIND, CXXTYPE)                                    \
  template <>                                                                  \
  struct TypeKindForCXXType<CXXTYPE> {                                         \
    static constexpr TypeKind kind = TypeKind::KIND;                           \
  };
DECLARE_TYPE_MAPPING(kBool, bool);
DECLARE_TYPE_MAPPING(kS8, int8_t);
DECLARE_TYPE_MAPPING(kU8, uint8_t);
DECLARE_TYPE_MAPPING(kS16, int16_t);
DECLARE_TYPE_MAPPING(kU16, uint16_t);
DECLARE_TYPE_MAPPING(kS32, int32_t);
DECLARE_TYPE_MAPPING(kU32, uint32_t);
DECLARE_TYPE_MAPPING(kS64, int64_t);
DECLARE_TYPE_MAPPING(kU64, uint64_t);
DECLARE_TYPE_MAPPING(kF32, float);
DECLARE_TYPE_MAPPING(kF64, double);
#undef DECLARE_TYPE_MAPPING

} // namespace BEF
} // namespace M

#endif // GRAPHRT_MLIR_TO_BEF_H
