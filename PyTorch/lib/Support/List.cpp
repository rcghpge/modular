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

using namespace M;
using namespace M::Torch;

ListLike::ListLike(SmallVector<Variant> elements)
    : elements(std::move(elements)) {}

ListLike::ListLike(const ListLike &other) {
  for (const Variant &elem : other.getElements())
    getElements().push_back(
        LLCL::AnyAsyncValueRef::copy(elem.getValue().getPointer()));
}

SmallVector<Variant> &ListLike::getElements() { return elements; }
const SmallVector<Variant> &ListLike::getElements() const { return elements; }

raw_ostream &M::Torch::operator<<(raw_ostream &os, const ListLike &list) {
  os << list.to_string();
  return os;
}

std::ostream &M::Torch::operator<<(std::ostream &os, const ListLike &list) {
  os << list.to_string();
  return os;
}

Tuple Tuple::operator=(const Tuple &other) {
  elements.clear();
  std::transform(other.elements.begin(), other.elements.end(),
                 this->elements.begin(),
                 [](const Variant &value) { return value.getValue().copy(); });
  return *this;
}

Tuple &Tuple::operator=(Tuple &&other) {
  elements.clear();
  elements = std::move(other.elements);
  return *this;
}

std::string Tuple::to_string() const {
  std::stringstream ss;
  ss << "Tuple(";
  for (size_t i = 0, e = elements.size(); i < e; ++i) {
    ss << elements[i] << (i == e - 1 ? "" : ", ");
  }
  ss << ")";
  return ss.str();
}

bool M::Torch::operator==(const Tuple &lhs, const Tuple &rhs) {
  if (lhs.getElements().size() != rhs.getElements().size())
    return false;
  for (size_t i = 0; i < lhs.getElements().size(); ++i) {
    if (lhs.getElements()[i] != rhs.getElements()[i])
      return false;
  }
  return true;
}

bool M::Torch::operator!=(const Tuple &lhs, const Tuple &rhs) {
  return !(lhs == rhs);
}

bool M::Torch::operator<(const Tuple &lhs, const Tuple &rhs) {
  for (size_t i = 0; i < lhs.getElements().size(); ++i) {
    if (lhs.getElements()[i] < rhs.getElements()[i])
      return true;
  }
  if (lhs.getElements().size() < rhs.getElements().size())
    return true;
  return false;
}

List List::operator=(const List &other) {
  elements.clear();
  std::transform(other.elements.begin(), other.elements.end(),
                 this->elements.begin(),
                 [](const Variant &value) { return value.getValue().copy(); });
  return *this;
}

List &List::operator=(List &&other) {
  elements.clear();
  elements = std::move(other.elements);
  return *this;
}

std::string List::to_string() const {
  std::stringstream ss;
  ss << "List[";
  for (size_t i = 0, e = elements.size(); i < e; ++i) {
    ss << elements[i] << (i == e - 1 ? "" : ", ");
  }
  ss << "]";
  return ss.str();
}

bool M::Torch::operator==(const List &lhs, const List &rhs) {
  if (lhs.getElements().size() != rhs.getElements().size())
    return false;
  for (size_t i = 0; i < lhs.getElements().size(); ++i) {
    if (lhs.getElements()[i] != rhs.getElements()[i])
      return false;
  }
  return true;
}

bool M::Torch::operator!=(const List &lhs, const List &rhs) {
  return !(lhs == rhs);
}

bool M::Torch::operator<(const List &lhs, const List &rhs) {
  for (size_t i = 0; i < lhs.getElements().size(); ++i) {
    if (lhs.getElements()[i] < rhs.getElements()[i])
      return true;
  }
  if (lhs.getElements().size() < rhs.getElements().size())
    return true;
  return false;
}

namespace std {

std::string to_string(const M::Torch::List &list) { return list.to_string(); }

std::string to_string(const M::Torch::Tuple &tuple) {
  return tuple.to_string();
}

size_t
hash<M::Torch::ListLike>::operator()(const M::Torch::ListLike &list) const {
  std::size_t seed = list.getElements().size();
  for (auto &elem : list.getElements()) {
    seed ^= M::Torch::hashCombine(seed, std::hash<M::Torch::Variant>()(elem));
  }
  return seed;
}

} // namespace std
