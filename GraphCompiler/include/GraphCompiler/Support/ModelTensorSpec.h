//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares the ModelTensorSpec class describing tensor semantics.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_SUPPORT_MODELTENSORSPEC_H
#define GENERICML_SUPPORT_MODELTENSORSPEC_H

#include "Support/ML/TensorSpec.h"
#include "llvm/ADT/StringRef.h"

namespace M {

/// Describes the semantics of a tensor, e.g. its name, shape, and dtype.
class ModelTensorSpec {
public:
  ModelTensorSpec(StringRef tensorName) : tensorName(tensorName) {}

  template <typename ShapeType>
  ModelTensorSpec(const ShapeType &shape, DType eltType, StringRef tensorName)
      : spec(shape, eltType), tensorName(tensorName) {}

  template <typename T>
  ModelTensorSpec(const std::initializer_list<T> &shape, DType eltType,
                  StringRef tensorName)
      : spec(shape, eltType), tensorName(tensorName) {}

  ModelTensorSpec(StringRef tensorName, TensorSpec tensorSpec)
      : spec(std::move(tensorSpec)), tensorName(tensorName) {}

  StringRef getName() const { return tensorName; }

  const TensorSpec &getSpec() const { return spec; }

private:
  TensorSpec spec;
  std::string tensorName;
};

} // namespace M

#endif // GENERICML_SUPPORT_MODELTENSORSPEC_H
