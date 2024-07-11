//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_SUPPORT_LIST_H
#define GENERICML_SUPPORT_LIST_H

#include "AsyncRT/Runtime/AnyAsyncValueRef.h"
#include "GenericML/Support/Value.h"
#include <cstdint>

namespace M::GML {

/// Base class for `List` and `Tuple` containers with `vector` implementation.
class ListLike : public Container {
public:
  ListLike() = default;
  ListLike(const ListLike &other);
  ListLike(ListLike &&other);
  virtual ~ListLike() = default;
  ListLike(const SmallVector<Variant> &elements);

  ListLike &operator=(ListLike &&other);

  // Returns a reference to the implementation.
  SmallVector<Variant> &getElements();
  const SmallVector<Variant> &getElements() const;

  /// Returns the string representation.
  virtual std::string to_string() const = 0;

protected:
  SmallVector<Variant> elements;
};

/// `Tuple`-like (pythonic) container class with `vector` implementation.
class Tuple : public ListLike {
public:
  Tuple() = default;
  Tuple(const Tuple &other) = default;
  Tuple(Tuple &&other) = default;

  Tuple operator=(const Tuple &other);
  Tuple &operator=(Tuple &&other);

  /// Returns the string representation.
  std::string to_string() const override;
};

/// Returns true if two `Tuple`s are equal. Equality is defined by the vector
/// contents.
bool operator==(const Tuple &lhs, const Tuple &rhs);

/// Returns true if two `Tuple`s are inequal. Inequality is defined by the
/// vector contents.
bool operator!=(const Tuple &lhs, const Tuple &rhs);

/// Returns true if one `Tuple` is less than the other. Comparison is defined by
/// the vector contents.
bool operator<(const Tuple &lhs, const Tuple &rhs);

/// `List`-like (pythonic) container class with `vector` implementation.
class List : public ListLike {
public:
  List() = default;
  List(const List &other) = default;
  List(List &&other) = default;
  List(ArrayRef<AnyAsyncValueRef> values);

  List copy();

  List operator=(const List &other);
  List &operator=(List &&other);

  /// Returns the string representation.
  std::string to_string() const override;
};

/// Returns true if two `List`s are equal. Equality is defined by the vector
/// contents.
bool operator==(const List &lhs, const List &rhs);

/// Returns true if two `List`s are inequal. Inequality is defined by the
/// vector contents.
bool operator!=(const List &lhs, const List &rhs);

/// Returns true if one `List` is less than the other. Comparison is defined by
/// the vector contents.
bool operator<(const List &lhs, const List &rhs);

raw_ostream &operator<<(raw_ostream &os, const ListLike &list);
std::ostream &operator<<(std::ostream &os, const ListLike &list);

} // namespace M::GML

namespace std {

/// Returns the string representation.
std::string to_string(const M::GML::List &list);

/// Returns the string representation.
std::string to_string(const M::GML::Tuple &tuple);

/// Computes the hash for `List` and `Dict`, which is based on the vector
/// contents.
template <>
struct hash<M::GML::ListLike> {
  size_t operator()(const M::GML::ListLike &listLike) const;
};

} // namespace std

#endif // GENERICML_SUPPORT_LIST_H
