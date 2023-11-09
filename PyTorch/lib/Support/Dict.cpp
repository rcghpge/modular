//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "PyTorch/Support/Dict.h"
#include "PyTorch/Support/List.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include <numeric>

namespace std {
size_t
hash<M::Torch::ValueKey>::operator()(const M::Torch::ValueKey &value) const {
  return M::Torch::valueHash(value.getValue());
}
} // namespace std

namespace M::Torch {

ValueKey::ValueKey() : value(LLCL::AnyAsyncValueRef()){};

ValueKey::ValueKey(const LLCL::AnyAsyncValueRef &value)
    : value(LLCL::AnyAsyncValueRef::copy(value.getPointer())) {}

ValueKey::ValueKey(const ValueKey &other) {
  value = LLCL::AnyAsyncValueRef::copy(other.getValue().getPointer());
}

std::string ValueKey::to_string() const { return valueToString(value); }

const LLCL::AnyAsyncValueRef &ValueKey::getValue() const { return value; }

ValueKey &ValueKey::operator=(const ValueKey &other) {
  value = LLCL::AnyAsyncValueRef::copy(other.getValue().getPointer());
  return *this;
}

bool operator==(const ValueKey &lhs, const ValueKey &rhs) {
  return valueEqual(lhs.getValue(), rhs.getValue());
}

bool operator<(const ValueKey &lhs, const ValueKey &rhs) {
  return valueLessThan(lhs.getValue(), rhs.getValue());
}

std::string
VariantToStringVisitor::operator()(const DenseMapInfoKey &value) const {
  switch (value) {
  case DenseMapInfoKey::Empty:
    return "<empty>";
  case DenseMapInfoKey::Tombstone:
    return "<tombstone>";
  }
}
std::string VariantToStringVisitor::operator()(const std::string &value) const {
  return value;
}
std::string VariantToStringVisitor::operator()(double value) const {
  return std::to_string(value);
}
std::string VariantToStringVisitor::operator()(long value) const {
  return std::to_string(value);
}
std::string VariantToStringVisitor::operator()(bool value) const {
  return std::to_string(value);
}
std::string VariantToStringVisitor::operator()(None value) const {
  return std::to_string(value);
}

std::string VariantToStringVisitor::operator()(const ValueKey &value) const {
  return std::to_string(value);
}

std::string to_string(const Variant &key) {
  return std::visit(VariantToStringVisitor(), key);
}

raw_ostream &operator<<(raw_ostream &os, const Variant &key) {
  os << to_string(key);
  return os;
}

std::ostream &operator<<(std::ostream &os, const Variant &key) {
  os << to_string(key);
  return os;
}

llvm::MapVector<Variant, LLCL::AnyAsyncValueRef> &Dict::getElements() {
  return elements;
}

Dict::Dict(const Dict &other) {
  for (const auto &elem : other.getElements()) {
    getElements()[elem.first] =
        LLCL::AnyAsyncValueRef::copy(elem.second.getPointer());
  }
}

const llvm::MapVector<Variant, LLCL::AnyAsyncValueRef> &
Dict::getElements() const {
  return elements;
}

std::string Dict::to_string() const {
  std::stringstream ss;
  ss << "Dict{";
  size_t e = elements.size();
  for (auto [elementIndex, elementValue] : llvm::enumerate(elements)) {
    if (std::holds_alternative<std::string>(elementValue.first))
      ss << "\"" << elementValue.first << "\"";
    else
      ss << elementValue.first;
    ss << ": " << valueToString(elementValue.second)
       << (elementIndex == e - 1 ? "" : ", ");
  }
  ss << "}";
  return ss.str();
}

raw_ostream &operator<<(raw_ostream &os, const Dict &dict) {
  os << dict.to_string();
  return os;
}

std::ostream &operator<<(std::ostream &os, const Dict &dict) {
  os << dict.to_string();
  return os;
}

bool operator==(const Dict &lhs, const Dict &rhs) {
  if (lhs.getElements().size() != rhs.getElements().size())
    return false;
  for (auto &kv : lhs.getElements()) {
    auto it = rhs.getElements().find(kv.first);
    if (it == rhs.getElements().end() || !valueEqual(it->second, kv.second))
      return false;
  }
  return true;
}

bool operator!=(const Dict &lhs, const Dict &rhs) { return !(lhs == rhs); }

bool operator<(const Dict &lhs, const Dict &rhs) { return false; }

} // namespace M::Torch

namespace std {

std::string to_string(const M::Torch::ValueKey &value) {
  return value.to_string();
}

std::string to_string(const M::Torch::Dict &dict) { return dict.to_string(); }

size_t hash<M::Torch::Dict>::operator()(const M::Torch::Dict &dict) const {
  std::hash<M::Torch::Variant> h;
  std::size_t seed = dict.getElements().size();
  for (auto &elem : dict.getElements()) {
    seed = M::Torch::hashCombine(seed, h(elem.first));
    seed = M::Torch::hashCombine(seed, M::Torch::valueHash(elem.second));
  }
  return seed;
}

} // namespace std
