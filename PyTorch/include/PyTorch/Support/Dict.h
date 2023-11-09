//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef PYTORCH_SUPPORT_DICT_H
#define PYTORCH_SUPPORT_DICT_H

#include "GenericML/Support/Tensor.h"
#include "Kernels/Misc.h"
#include "LLCL/Runtime/AnyAsyncValueRef.h"
#include "PyTorch/Support/Value.h"
#include "llvm/ADT/MapVector.h"
#include <iostream>
#include <map>

namespace M::Torch {

/// Wrapper around `AnyAsyncValueRef` providing methods required for using
/// `AnyAsyncValueRef` as a map key (`hash`, `to_string`, `==`).
class ValueKey {
public:
  ValueKey();

  /// Creates a value key wrapping an
  ValueKey(const LLCL::AnyAsyncValueRef &value);
  ValueKey(const ValueKey &other);

  /// Returns the string representation of the wrapped `AnyAsyncValueRef`.
  std::string to_string() const;

  /// Returns a reference to the wrapped `AnyAsyncValueRef`.
  const LLCL::AnyAsyncValueRef &getValue() const;

  /// Copy assignment operator required for assignment to `Variant`. Note that
  /// the implementation copies the value ref and increments the ref count.
  ValueKey &operator=(const ValueKey &other);

private:
  LLCL::AnyAsyncValueRef value;
};

/// Defines equality on the underlying `AnyAsyncValueRef`s. See `Value.h` for
/// details.
bool operator==(const ValueKey &lhs, const ValueKey &rhs);

/// Defines the less-than operator on the underlying `AnyAsyncValueRef`s. See
/// `Value.h` for details.
bool operator<(const ValueKey &lhs, const ValueKey &rhs);

} // namespace M::Torch

namespace std {

/// Computes the hash function of the underlying `AnyAsyncValueRef`.
template <>
struct hash<M::Torch::ValueKey> {
  size_t operator()(const M::Torch::ValueKey &value) const;
};

} // namespace std

namespace M::Torch {

// Enum class to flag special keys used for `DenseMapInfo`.
enum class DenseMapInfoKey {
  Empty = 0,
  Tombstone = 1,
};

/// Variant type used to support primitive, string and composed keys in
/// heterogeneous dictionaries.
using Variant = std::variant<DenseMapInfoKey, std::string, double, long, bool,
                             None, ValueKey>;

} // namespace M::Torch

namespace llvm {
/// `DenseMapInfo` for `Variant` such that it can be used as a `MapVector` key.
template <>
struct DenseMapInfo<M::Torch::Variant> {
  static inline M::Torch::Variant getEmptyKey() {
    return M::Torch::Variant(M::Torch::DenseMapInfoKey::Empty);
  }
  static inline M::Torch::Variant getTombstoneKey() {
    return M::Torch::Variant(M::Torch::DenseMapInfoKey::Tombstone);
  }
  static unsigned getHashValue(const M::Torch::Variant &val) {
    return std::hash<M::Torch::Variant>()(val);
  }
  static bool isEqual(const M::Torch::Variant &lhs,
                      const M::Torch::Variant &rhs) {
    return lhs == rhs;
  }
};
} // namespace llvm

namespace M::Torch {

/// Visitor for converting `Variant` to a string representation.
struct VariantToStringVisitor {
  std::string operator()(const DenseMapInfoKey &value) const;
  std::string operator()(const std::string &value) const;
  std::string operator()(double value) const;
  std::string operator()(long value) const;
  std::string operator()(bool value) const;
  std::string operator()(None value) const;
  std::string operator()(const ValueKey &value) const;
};

/// Returns the string representation of `Variant`. Delegates to the individual
/// key types.
std::string to_string(const Variant &key);

/// Writes the string representation of `Variant` to a stream.
raw_ostream &operator<<(raw_ostream &os, const Variant &key);
std::ostream &operator<<(std::ostream &os, const Variant &key);

/// `Dict`-like (pythonic) container class with `map` implementation. An
/// insertion order preserving `map` is used such that printing dictionary
/// contents for tests and debugging is consistent.
class Dict : public Container {
public:
  Dict() = default;
  Dict(const Dict &other);

  /// Return a reference to the implementation.
  llvm::MapVector<Variant, LLCL::AnyAsyncValueRef> &getElements();
  const llvm::MapVector<Variant, LLCL::AnyAsyncValueRef> &getElements() const;

  /// Returns the string representation.
  std::string to_string() const;

private:
  llvm::MapVector<Variant, LLCL::AnyAsyncValueRef> elements;
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

} // namespace M::Torch

namespace std {

/// Returns the string representation.
std::string to_string(const M::Torch::ValueKey &value);

/// Returns the string representation.
std::string to_string(const M::Torch::Dict &dict);

/// Computes the hash for `Dict`, which is based on the map contents.
template <>
struct hash<M::Torch::Dict> {
  size_t operator()(const M::Torch::Dict &dict) const;
};

} // namespace std

#endif // PYTORCH_SUPPORT_DICT_H
