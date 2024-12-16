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
  TorchInputSpec(const InputShapeTy &shape, M_Dtype dtype, std::string device);
  TorchInputSpec(TorchInputSpec &&other) noexcept = default;

  TorchInputSpec(const TorchInputSpec &other)
      : TorchInputSpec(other.specShape, other.specDtype, other.specDevice) {}

  InputShapeTy getShape() const { return specShape; }

  M_Dtype getDtype() const { return specDtype; }

  std::string getDevice() const { return specDevice; }

  TorchInputSpecPtr specPtr;
  InputShapeTy specShape;
  M_Dtype specDtype;
  std::string specDevice;
};

} // namespace M::core

#endif // SDK_ENGINEAPI_PYTHON_TORCHINPUTSPEC_H
