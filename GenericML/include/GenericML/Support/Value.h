//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_SUPPORT_VALUE_H
#define GENERICML_SUPPORT_VALUE_H

#include "ArraySupport/Tensor.h"
#include "AsyncRT/Runtime/AnyAsyncValueRef.h"

namespace M::GML {

/// Base class for all container classes (`Dict`, `List`, `Tuple`). We implement
/// a range of container and composed types to support gluing the C and Python
/// APIs to the runtime types used inside a model. Models that interact with
/// non-tensor inputs and outputs require a different set of runtime types.
/// This is required because, depending on the compiled model's internal values,
/// they can't be used outside of the execution. On the other hand,
/// `pybind11::dict` and friends cannot be passed deeper into our stack (beyond
/// the Python bindings) due to requiring RTTI. The classes here are a
/// lightweight implementation of similar classes wrapped by Python's `Dict`,
/// `List`, `Tuple` and `None` types. The implementation does not depend on
/// MLIR, Python, Pybind or framework-specific implementations (all of which can
/// lead to symbol or RTTI conflicts). The implementation does not require
/// support for dynamic casts.
class Container {};

/// Class representing the pythonic `None` value. Used to tunnel `None` between
/// the C/Python API and `None` values in the compiled models.
class None {
public:
  None() = default;

  None(const None &other) = default;

  None &operator=(const None &other) = default;
  None &operator=(None &&other);

  /// Returns the string representation "<none>".
  std::string to_string() const;

  /// Returns true for all instances of `None`.
  bool operator==(const None &other) const;

  /// Returns false for all instances of `None`.
  bool operator<(const None &other) const;
};

raw_ostream &operator<<(raw_ostream &os, const None &none);
std::ostream &operator<<(std::ostream &os, const None &none);

// Enum class to flag special keys used for `DenseMapInfo`.
enum class DenseMapInfoKey {
  Empty = 0,
  Tombstone = 1,
  Value = 2,
};

raw_ostream &operator<<(raw_ostream &os, const DenseMapInfoKey &info);
std::ostream &operator<<(std::ostream &os, const DenseMapInfoKey &info);

/// Wrapper around `AnyAsyncValueRef` providing methods required for using
/// `AnyAsyncValueRef` as a map key (`hash`, `to_string`, `==`) as well as
/// Simplifying string and YAML representations.
class Variant {
public:
  Variant();

  Variant(const AsyncRT::AnyAsyncValueRef &other);
  Variant(const Variant &other);
  Variant(const DenseMapInfoKey &info);

  /// Returns the string representation of the wrapped `AnyAsyncValueRef`.
  std::string to_string() const;

  DenseMapInfoKey getInfo() const;

  /// Returns a reference to the wrapped `AnyAsyncValueRef`.
  const AsyncRT::AnyAsyncValueRef &getValue() const;

  /// Copy assignment operator required for assignment to `Variant`. Note
  /// that the implementation copies the value ref and increments the ref count.
  Variant &operator=(const Variant &other);
  Variant &operator=(Variant &&other);

private:
  DenseMapInfoKey info;
  AsyncRT::AnyAsyncValueRef value;
};

/// Defines equality on the underlying `AnyAsyncValueRef`s. See `Value.h` for
/// details.
bool operator==(const Variant &lhs, const Variant &rhs);

/// Defines inequality on the underlying `AnyAsyncValueRef`s. See `Value.h` for
/// details.
bool operator!=(const Variant &lhs, const Variant &rhs);

/// Defines the less-than operator on the underlying `AnyAsyncValueRef`s. See
/// `Value.h` for details.
bool operator<(const Variant &lhs, const Variant &rhs);

/// Writes the string representation of `Variant` to a stream.
raw_ostream &operator<<(raw_ostream &os, const Variant &variant);
std::ostream &operator<<(std::ostream &os, const Variant &variant);

/// This prints the (nested) values to stdout. Leaf node tensors are printed
/// with `Kernels::printTensor`.
ErrorOrSuccess printGenericValue(const AsyncRT::AnyAsyncValueRef &value,
                                 ResultOutputStyle style, unsigned precision);

/// Returns a string representation of an `AnyAsyncValueRef` referencing
/// a supported composed type or tensor. Note: The function is deliberately
/// not called `to_string` due to not being generic.
std::string valueToString(const AsyncRT::AnyAsyncValueRef &value);

/// Returns true if two `AnyAsyncValueRef`s are equal. Equality is defined
/// similarly to PyTorch/Python. Container types are uniquely identified by
/// their content, tensors by their underlying buffer pointer. Note: The
/// function is deliberately not overriding the `==` operator due to not being
/// generic.
bool valueEqual(const AsyncRT::AnyAsyncValueRef &lhs,
                const AsyncRT::AnyAsyncValueRef &rhs);

/// Returns true if one `AnyAsyncValueRef` is less than the other. Less-than is
/// defined similarly to PyTorch/Python. Container types are compared
/// by their content, tensors by their underlying buffer pointer. Note: The
/// function is deliberately not overriding the `<` operator due to not being
/// generic.
bool valueLessThan(const AsyncRT::AnyAsyncValueRef &lhs,
                   const AsyncRT::AnyAsyncValueRef &rhs);

/// Returns a hash for  `AnyAsyncValueRef`s. The hash is computed based on
/// container type contents, and for tensors, the underlying buffer pointer is
/// used. Note: The function is deliberately not overriding the hash-struct
/// due to not being generic. Unsupported wrapped types are mapped to
/// `std::numeric_limits<size_t>::max()` as hash.
size_t valueHash(const AsyncRT::AnyAsyncValueRef &value);

/// Combines an existing hash with a new hash using an algorithm based on the
/// golden ratio.
size_t hashCombine(size_t seed, size_t value);

} // namespace M::GML

namespace std {

std::string to_string(const M::GML::DenseMapInfoKey &value);

/// Computes the hash for `None`, which is constant 0.
template <>
struct hash<M::GML::None> {
  size_t operator()(const M::GML::None &none) const { return 0; }
};

/// Returns the string representation "<none>".
std::string to_string(const M::GML::None &none);

/// Computes the hash function of the underlying `AnyAsyncValueRef`.
template <>
struct hash<M::GML::Variant> {
  size_t operator()(const M::GML::Variant &value) const;
};

/// Returns the string representation of `Variant`. Delegates to the
/// individual key types.
std::string to_string(const M::GML::Variant &variant);

} // namespace std

#endif // GENERICML_SUPPORT_VALUE_H
