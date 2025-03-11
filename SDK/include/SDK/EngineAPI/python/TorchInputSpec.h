//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
#ifndef SDK_ENGINEAPI_PYTHON_TORCHINPUTSPEC_H
#define SDK_ENGINEAPI_PYTHON_TORCHINPUTSPEC_H

#include "PyTorch/API/max/cpp/pytorch/aliases.h"
#include "Support/ML/DType.h"
#include <Python.h>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace M::core {

class TorchInputSpec {
public:
  using InputShapeTy = std::optional<
      std::vector<std::optional<std::variant<int64_t, std::string>>>>;
  TorchInputSpec(const InputShapeTy &shape, M::DType dtype);
  TorchInputSpec(const InputShapeTy &shape, M::DType dtype, std::string device);
  TorchInputSpec(TorchInputSpec &&other) noexcept = default;

  TorchInputSpec(const TorchInputSpec &other)
      : TorchInputSpec(other.specShape, other.specDtype, other.specDevice) {}

  InputShapeTy getShape() const { return specShape; }

  M::DType getDtype() const { return specDtype; }

  std::string getDevice() const { return specDevice; }

  TorchInputSpecPtr specPtr;
  InputShapeTy specShape;
  M::DType specDtype;
  std::string specDevice;
};

} // namespace M::core

#endif // SDK_ENGINEAPI_PYTHON_TORCHINPUTSPEC_H
