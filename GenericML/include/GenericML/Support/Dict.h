//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_SUPPORT_DICT_H
#define GENERICML_SUPPORT_DICT_H

#include "GenericML/Support/Value.h"
#include "llvm/ADT/MapVector.h"
#include <iostream>

namespace llvm {
/// `DenseMapInfo` for `Variant` such that it can be used as a
/// `MapVector` key.
template <>
struct DenseMapInfo<M::GML::Variant> {
  static inline M::GML::Variant getEmptyKey() {
    return M::GML::Variant(M::GML::DenseMapInfoKey::Empty);
  }
  static inline M::GML::Variant getTombstoneKey() {
    return M::GML::Variant(M::GML::DenseMapInfoKey::Tombstone);
  }
  static unsigned getHashValue(const M::GML::Variant &val) {
    return std::hash<M::GML::Variant>()(val);
  }
  static bool isEqual(const M::GML::Variant &lhs, const M::GML::Variant &rhs) {
    return lhs == rhs;
  }
};
} // namespace llvm

namespace M::GML {

/// `Dict`-like (pythonic) container class with `map` implementation. An
/// insertion order preserving `map` is used such that printing dictionary
/// contents for tests and debugging is consistent.
class Dict : public Container {
public:
  Dict() = default;
  Dict(const Dict &other);

  Dict &operator=(const Dict &other);
  Dict operator=(Dict &&other);

  /// Return a reference to the implementation.
  llvm::MapVector<Variant, Variant> &getElements();
  const llvm::MapVector<Variant, Variant> &getElements() const;

  /// Returns the string representation.
  std::string to_string() const;

private:
  llvm::MapVector<Variant, Variant> elements;
};

/// Returns true if two `Dict`s are equal. Equality is defined by the map
/// contents.
bool operator==(const Dict &lhs, const Dict &rhs);

/// Returns true if two `Dict`s are inequal. Inequality is defined by the
/// map contents.
bool operator!=(const Dict &lhs, const Dict &rhs);

/// Always returns false, since `<` is not supported on `Dict`.
bool operator<(const Dict &lhs, const Dict &rhs);

/// Writes the string representation of `Dict` to a stream.
raw_ostream &operator<<(raw_ostream &os, const Dict &dict);
std::ostream &operator<<(std::ostream &os, const Dict &dict);

} // namespace M::GML

namespace std {

/// Returns the string representation.
std::string to_string(const M::GML::Dict &dict);

/// Computes the hash for `Dict`, which is based on the map contents.
template <>
struct hash<M::GML::Dict> {
  size_t operator()(const M::GML::Dict &dict) const;
};

} // namespace std

#endif // GENERICML_SUPPORT_DICT_H
