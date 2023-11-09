//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "PyTorch/Support/Value.h"
#include "PyTorch/Support/Dict.h"
#include "PyTorch/Support/List.h"
#include "Support/LLVMCompilerForwardDecls.h"

namespace M::Torch {

bool None::operator==(const None &other) const { return true; }

bool None::operator<(const None &other) const { return false; }

std::string None::to_string() const { return "<none>"; }

ErrorOrSuccess printGenericValue(const LLCL::AnyAsyncValueRef &value,
                                 Kernels::ResultOutputStyle style,
                                 unsigned precision) {
  auto &os = llvm::outs();
  if (List *list = dyn_cast<List>(value)) {
    size_t size = list->getElements().size();
    os << "List[";
    for (auto [index, value] : llvm::enumerate(list->getElements())) {
      ErrorOrSuccess errorOr = printGenericValue(value, style, precision);
      if (errorOr.isError())
        return errorOr;
      if (index < size - 1)
        os << ", ";
    }
    os << "]";
  } else if (Tuple *tuple = dyn_cast<Tuple>(value)) {
    size_t size = tuple->getElements().size();
    os << "Tuple[";
    for (auto [index, value] : llvm::enumerate(tuple->getElements())) {
      ErrorOrSuccess errorOr = printGenericValue(value, style, precision);
      if (errorOr.isError())
        return errorOr;
      if (index < size - 1)
        os << ", ";
    }
    os << "]";
  } else if (Dict *dict = dyn_cast<Dict>(value)) {
    size_t size = dict->getElements().size();
    os << "Dict{";
    for (auto [index, value] : llvm::enumerate(dict->getElements())) {
      os << "\"" << value.first << "\": ";
      ErrorOrSuccess errorOr =
          printGenericValue(value.second, style, precision);
      if (errorOr.isError())
        return errorOr;
      if (index < size - 1)
        os << ", ";
    }
    os << "}";
  } else if (None *none = dyn_cast<None>(value)) {
    os << none;
  } else if (Variant *variant = dyn_cast<Variant>(value)) {
    if (std::holds_alternative<std::string>(*variant))
      os << "\"" << *variant << "\"";
    else
      os << *variant;
  } else if (GML::Tensor *tensor = dyn_cast<GML::Tensor>(value)) {
    ErrorOrSuccess errorOr =
        printTensor(tensor->getBuffer(), tensor->getSpec(), style, precision);
    if (errorOr.isError())
      return errorOr;
  }
  return success();
}

std::string valueToString(const LLCL::AnyAsyncValueRef &value) {
  std::stringstream ss;
  if (List *list = dyn_cast<List>(value)) {
    ss << *list;
  } else if (Tuple *tuple = dyn_cast<Tuple>(value)) {
    ss << *tuple;
  } else if (Dict *dict = dyn_cast<Dict>(value)) {
    ss << *dict;
  } else if (None *none = dyn_cast<None>(value)) {
    ss << *none;
  } else if (Variant *variant = dyn_cast<Variant>(value)) {
    if (std::holds_alternative<std::string>(*variant))
      ss << "\"" << *variant << "\"";
    else
      ss << *variant;
  } else if (GML::Tensor *tensor = dyn_cast<GML::Tensor>(value)) {
    ss << tensor;
  } else {
    return "<unknown>";
  }
  return ss.str();
};

bool valueEqual(const LLCL::AnyAsyncValueRef &lhs,
                const LLCL::AnyAsyncValueRef &rhs) {
  if (List *llist = dyn_cast<List>(lhs)) {
    if (List *rlist = dyn_cast<List>(rhs))
      return *llist == *rlist;
  } else if (Tuple *ltuple = dyn_cast<Tuple>(lhs)) {
    if (Tuple *rtuple = dyn_cast<Tuple>(rhs))
      return *ltuple == *rtuple;
  } else if (Dict *ldict = dyn_cast<Dict>(lhs)) {
    if (Dict *rdict = dyn_cast<Dict>(rhs))
      return *ldict == *rdict;
  } else if (None *lnone = dyn_cast<None>(lhs)) {
    if (None *rnone = dyn_cast<None>(rhs))
      return *lnone == *rnone;
  } else if (Variant *lvariant = dyn_cast<Variant>(lhs)) {
    if (Variant *rvariant = dyn_cast<Variant>(rhs))
      return *lvariant == *rvariant;
  } else if (GML::Tensor *ltensor = dyn_cast<GML::Tensor>(lhs)) {
    if (GML::Tensor *rtensor = dyn_cast<GML::Tensor>(rhs))
      // Consider tensors equal if the underlying data pointer is the same.
      return ltensor->getBuffer() == rtensor->getBuffer();
  }
  return false;
}

bool valueLessThan(const LLCL::AnyAsyncValueRef &lhs,
                   const LLCL::AnyAsyncValueRef &rhs) {
  if (List *llist = dyn_cast<List>(lhs)) {
    if (List *rlist = dyn_cast<List>(rhs))
      return *llist < *rlist;
  } else if (Tuple *ltuple = dyn_cast<Tuple>(lhs)) {
    if (Tuple *rtuple = dyn_cast<Tuple>(rhs))
      return *ltuple < *rtuple;
  } else if (Dict *ldict = dyn_cast<Dict>(lhs)) {
    if (Dict *rdict = dyn_cast<Dict>(rhs))
      return *ldict < *rdict;
  } else if (None *lnone = dyn_cast<None>(lhs)) {
    if (None *rnone = dyn_cast<None>(rhs))
      return *lnone < *rnone;
  } else if (Variant *lvariant = dyn_cast<Variant>(lhs)) {
    if (Variant *rvariant = dyn_cast<Variant>(rhs))
      return *lvariant < *rvariant;
  } else if (GML::Tensor *ltensor = dyn_cast<GML::Tensor>(lhs)) {
    if (GML::Tensor *rtensor = dyn_cast<GML::Tensor>(rhs))
      // Consider tensors equal if the underlying data pointer is the same.
      return ltensor->getBuffer() < rtensor->getBuffer();
  }
  return false;
}

size_t valueHash(const LLCL::AnyAsyncValueRef &value) {
  if (List *list = dyn_cast<List>(value))
    return std::hash<ListLike>()(*list);
  else if (Tuple *tuple = dyn_cast<Tuple>(value))
    return std::hash<ListLike>()(*tuple);
  else if (Dict *dict = dyn_cast<Dict>(value))
    return std::hash<Dict>()(*dict);
  else if (None *none = dyn_cast<None>(value))
    return std::hash<None>()(*none);
  else if (Variant *variant = dyn_cast<Variant>(value))
    return std::hash<Variant>()(*variant);
  else if (GML::Tensor *tensor = dyn_cast<GML::Tensor>(value))
    return reinterpret_cast<size_t>(tensor->getBuffer());
  else
    return std::numeric_limits<size_t>::max();
}

size_t hashCombine(size_t seed, size_t value) {
  // Similar to `boost::hash_combine`, but with 64 bit support.
  // 2^64 / golden ratio in base 16.
  return seed ^ (value + 0x9e3779b97f4a7c15LLU + (seed << 6) + (seed >> 2));
}

} // namespace M::Torch

namespace std {
std::string to_string(const M::Torch::None &none) { return none.to_string(); }

} // namespace std
