//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "PyTorch/Support/Dict.h"
#include "PyTorch/Support/List.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include <numeric>
#include <sstream>

using namespace M;
using namespace M::Torch;

Dict::Dict(const Dict &other) {
  for (const auto &elem : other.getElements()) {
    getElements()[elem.first] = Variant(
        LLCL::AnyAsyncValueRef::copy(elem.second.getValue().getPointer()));
  }
}

Dict &Dict::operator=(const Dict &other) {
  elements.clear();
  for (auto &elem : other.getElements())
    elements.insert(elem);
  return *this;
}

Dict Dict::operator=(Dict &&other) {
  this->elements.clear();
  this->elements = std::move(other.elements);
  return *this;
}

llvm::MapVector<Variant, Variant> &Dict::getElements() { return elements; }

const llvm::MapVector<Variant, Variant> &Dict::getElements() const {
  return elements;
}

std::string Dict::to_string() const {
  std::stringstream ss;
  ss << "Dict{";
  size_t e = elements.size();
  for (auto [elementIndex, elementValue] : llvm::enumerate(elements)) {
    ss << elementValue.first;
    ss << ": " << elementValue.second << (elementIndex == e - 1 ? "" : ", ");
  }
  ss << "}";
  return ss.str();
}

raw_ostream &M::Torch::operator<<(raw_ostream &os, const Dict &dict) {
  os << dict.to_string();
  return os;
}

std::ostream &M::Torch::operator<<(std::ostream &os, const Dict &dict) {
  os << dict.to_string();
  return os;
}

bool M::Torch::operator==(const Dict &lhs, const Dict &rhs) {
  if (lhs.getElements().size() != rhs.getElements().size())
    return false;
  for (auto &kv : lhs.getElements()) {
    auto it = rhs.getElements().find(kv.first);
    if (it == rhs.getElements().end() || (it->second != kv.second))
      return false;
  }
  return true;
}

bool M::Torch::operator!=(const Dict &lhs, const Dict &rhs) {
  return !(lhs == rhs);
}

bool M::Torch::operator<(const Dict &lhs, const Dict &rhs) { return false; }

namespace std {

std::string to_string(const M::Torch::Dict &dict) { return dict.to_string(); }

size_t hash<M::Torch::Dict>::operator()(const M::Torch::Dict &dict) const {
  std::hash<M::Torch::Variant> h;
  std::size_t seed = dict.getElements().size();
  for (auto &elem : dict.getElements()) {
    seed = M::Torch::hashCombine(seed, h(elem.first));
    seed = M::Torch::hashCombine(seed,
                                 std::hash<M::Torch::Variant>()(elem.second));
  }
  return seed;
}

} // namespace std
