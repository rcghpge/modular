//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "PyTorch/Support/List.h"
#include "PyTorch/Support/Dict.h"
#include "PyTorch/Support/Value.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include <utility>

namespace M::Torch {

ListLike::ListLike(SmallVector<LLCL::AnyAsyncValueRef> elements)
    : elements(std::move(elements)) {}

ListLike::ListLike(const ListLike &other) {
  for (const LLCL::AnyAsyncValueRef &elem : other.getElements()) {
    getElements().push_back(LLCL::AnyAsyncValueRef::copy(elem.getPointer()));
  }
}

SmallVector<LLCL::AnyAsyncValueRef> &ListLike::getElements() {
  return elements;
}
const SmallVector<LLCL::AnyAsyncValueRef> &ListLike::getElements() const {
  return elements;
}

raw_ostream &operator<<(raw_ostream &os, const ListLike &list) {
  os << list.to_string();
  return os;
}

std::ostream &operator<<(std::ostream &os, const ListLike &list) {
  os << list.to_string();
  return os;
}

std::string Tuple::to_string() const {
  std::stringstream ss;
  ss << "Tuple(";
  for (size_t i = 0, e = elements.size(); i < e; ++i) {
    ss << valueToString(elements[i]) << (i == e - 1 ? "" : ", ");
  }
  ss << ")";
  return ss.str();
}

bool operator==(const Tuple &lhs, const Tuple &rhs) {
  if (lhs.getElements().size() != rhs.getElements().size())
    return false;
  for (size_t i = 0; i < lhs.getElements().size(); ++i) {
    if (!valueEqual(lhs.getElements()[i], rhs.getElements()[i]))
      return false;
  }
  return true;
}

bool operator!=(const Tuple &lhs, const Tuple &rhs) { return !(lhs == rhs); }

bool operator<(const Tuple &lhs, const Tuple &rhs) {
  for (size_t i = 0; i < lhs.getElements().size(); ++i) {
    if (valueLessThan(lhs.getElements()[i], rhs.getElements()[i]))
      return true;
  }
  if (lhs.getElements().size() < rhs.getElements().size())
    return true;
  return false;
}

std::string List::to_string() const {
  std::stringstream ss;
  ss << "List[";
  for (size_t i = 0, e = elements.size(); i < e; ++i) {
    ss << valueToString(elements[i]) << (i == e - 1 ? "" : ", ");
  }
  ss << "]";
  return ss.str();
}

bool operator==(const List &lhs, const List &rhs) {
  if (lhs.getElements().size() != rhs.getElements().size())
    return false;
  for (size_t i = 0; i < lhs.getElements().size(); ++i) {
    if (!valueEqual(lhs.getElements()[i], rhs.getElements()[i]))
      return false;
  }
  return true;
}

bool operator!=(const List &lhs, const List &rhs) { return !(lhs == rhs); }

bool operator<(const List &lhs, const List &rhs) {
  for (size_t i = 0; i < lhs.getElements().size(); ++i) {
    if (valueLessThan(lhs.getElements()[i], rhs.getElements()[i]))
      return true;
  }
  if (lhs.getElements().size() < rhs.getElements().size())
    return true;
  return false;
}

} // namespace M::Torch

namespace std {

std::string to_string(const M::Torch::List &list) { return list.to_string(); }

std::string to_string(const M::Torch::Tuple &tuple) {
  return tuple.to_string();
}

size_t
hash<M::Torch::ListLike>::operator()(const M::Torch::ListLike &list) const {
  std::size_t seed = list.getElements().size();
  for (auto &elem : list.getElements()) {
    seed ^= M::Torch::hashCombine(seed, M::Torch::valueHash(elem));
  }
  return seed;
}

} // namespace std
