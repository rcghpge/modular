//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SDK_ENGINECLI_BENCHMARK_TENSOR_H
#define SDK_ENGINECLI_BENCHMARK_TENSOR_H

#include "Support/ML/TensorSpec.h"
#include <memory>

namespace M::BenchmarkCLI {

/// Wrapper around a TensorSpec and some data.
///
/// Used because a Tensor requires an AsyncRT runtime to be able to
/// create tensors, which we'd like to avoid, since we don't otherwise deal with
/// AsyncRT runtimes directly.  Otherwise mimics a similar API to
/// Tensor.
class Tensor {
public:
  class DataOwner {
  public:
    virtual ~DataOwner() = default;
  };

  Tensor();
  Tensor(TensorSpec spec, const void *data,
         std::unique_ptr<DataOwner> dataOwner);
  Tensor(const Tensor &) = delete;
  Tensor(Tensor &&);
  ~Tensor();

  Tensor &operator=(const Tensor &) = delete;
  Tensor &operator=(Tensor &&);

  /// Returns the tensor spec/shape/element type.
  const TensorSpec &getSpec() const { return spec; }
  const TensorShape &getShape() const { return spec; }
  DType getEltType() const { return spec.getEltType(); }

  /// Returns the size of the underlying buffer in bytes.
  size_t getSizeInBytes() const { return spec.getSizeInBytes(); }

  /// Returns a pointer to the underlying buffer.
  const void *getBuffer() const { return data; }

  template <typename T>
  static std::unique_ptr<DataOwner> own(T value) {
    return std::make_unique<DataOwnerImpl<std::remove_reference_t<T>>>(
        std::move(value));
  }

private:
  template <typename T>
  class DataOwnerImpl : public DataOwner {
  public:
    explicit DataOwnerImpl(T owned) : owned(std::move(owned)) {}

  private:
    T owned;
  };

  TensorSpec spec;
  const void *data;
  std::unique_ptr<DataOwner> dataOwner;
};

} // namespace M::BenchmarkCLI

#endif // SDK_ENGINECLI_BENCHMARK_TENSOR_H
