//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef PYTORCH_SUPPORT_VALUE_H
#define PYTORCH_SUPPORT_VALUE_H

#include "GenericML/Support/Tensor.h"
#include "Kernels/Misc.h"
#include "LLCL/Runtime/AnyAsyncValueRef.h"
#include <cstdint>
#include <memory>

namespace M::Torch {

/// Base class for all container classes (`Dict`, `List`, `Tuple`). We implement
/// a range of container and composed types to support gluing the C and Python
/// APIs to the runtime types used inside a model. Models that interact with
/// non-tensor inputs and outputs require a different set of runtime types.
/// This is required for PyTorch because `IValue`s can't be used outside of
/// `TorchInterface` due to symbol conflicts, and we want to avoid linking
/// `TorchInterface` to the generic Python bindings (which would allow using
/// `StockTorchInterface::IValue`s). On the other hand, `pybind11::dict` and
/// friends cannot be passed deeper into our stack (beyond the Python bindings)
/// due to requiring RTTI. The classes here are a lightweight implementation of
/// similiar classes wrapped by PyTorch's `IValue`s and Python's `Dict`, `List`,
/// `Tuple` and `None` types. The implementation does not depend on MLIR,
/// Python, Pybind or PyTorch (all of which can lead to symbol or RTTI
/// conflicts). The implementation does not require support for dynamic casts.
class Container {};

/// Class representing the pythonic `None` value. Used to tunnel `None` between
/// the C/Python API and PyTorch's `IValue`s.
class None {
public:
  None() = default;

  /// Returns the string representation "<none>".
  std::string to_string() const;

  /// Returns true for all instances of `None`.
  bool operator==(const None &other) const;

  /// Returns false for all instances of `None`.
  bool operator<(const None &other) const;
};

/// This prints the (nested) values to stdout. Leaf node tensors are printed
/// with `Kernels::printTensor`.
ErrorOrSuccess printGenericValue(const LLCL::AnyAsyncValueRef &value,
                                 Kernels::ResultOutputStyle style,
                                 unsigned precision);

/// Returns a string representation of an `AnyAsyncValueRef` referencing
/// a supported composed type or tensor. Note: The function is deliberately
/// not called `to_string` due to not being generic.
std::string valueToString(const LLCL::AnyAsyncValueRef &value);

/// Returns true if two `AnyAsyncValueRef`s are equal. Equality is defined
/// similarly to PyTorch/Python. Container types are uniquely identified by
/// their content, tensors by their underlying buffer pointer. Note: The
/// function is deliberately not overriding the `==` operator due to not being
/// generic.
bool valueEqual(const LLCL::AnyAsyncValueRef &lhs,
                const LLCL::AnyAsyncValueRef &rhs);

/// Returns true if one `AnyAsyncValueRef` is less than the other. Less-than is
/// defined similarly to PyTorch/Python. Container types are compared
/// by their content, tensors by their underlying buffer pointer. Note: The
/// function is deliberately not overriding the `<` operator due to not being
/// generic.
bool valueLessThan(const LLCL::AnyAsyncValueRef &lhs,
                   const LLCL::AnyAsyncValueRef &rhs);

/// Returns a hash for  `AnyAsyncValueRef`s. The hash is computed based on
/// container type contents, and for tensors, the underlying buffer pointer is
/// used. Note: The function is deliberately not overriding the hash-struct
/// due to not being generic. Unsupported wrapped types are mapped to
/// `std::numeric_limits<size_t>::max()` as hash.
size_t valueHash(const LLCL::AnyAsyncValueRef &value);

/// Combines an existing hash with a new hash using an algorithm based on the
/// golden ratio.
size_t hashCombine(size_t seed, size_t value);

} // namespace M::Torch

namespace std {

/// Computes the hash for `None`, which is constant 0.
template <>
struct hash<M::Torch::None> {
  size_t operator()(const M::Torch::None &none) const { return 0; }
};

/// Returns the string representation "<none>".
std::string to_string(const M::Torch::None &none);

} // namespace std

#endif // PYTORCH_SUPPORT_VALUE_H
