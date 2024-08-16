//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
#ifndef SDK_ENGINEAPI_PYTHON_TORCHINPUTSPEC_H
#define SDK_ENGINEAPI_PYTHON_TORCHINPUTSPEC_H

#include "GenericML/API/max/c/types.h"
#include "PyTorch/API/max/cpp/pytorch/aliases.h"
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace M::core {

class TorchInputSpec {
public:
  using InputShapeTy = std::optional<
      std::vector<std::optional<std::variant<int64_t, std::string>>>>;
  TorchInputSpec(const InputShapeTy &shape, M_Dtype dtype);
  TorchInputSpec(TorchInputSpec &&other) noexcept = default;

  TorchInputSpec(const TorchInputSpec &other)
      : TorchInputSpec(other.specShape, other.specDtype) {}

  InputShapeTy getShape() const { return specShape; }

  M_Dtype getDtype() const { return specDtype; }

  TorchInputSpecPtr specPtr;
  InputShapeTy specShape;
  M_Dtype specDtype;
};

} // namespace M::core

#endif // SDK_ENGINEAPI_PYTHON_TORCHINPUTSPEC_H
