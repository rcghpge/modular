//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// SafeTensor reader which is GPU aware.
//
//===----------------------------------------------------------------------===//

#ifndef RUNTIME_SAFETENSOR_H
#define RUNTIME_SAFETENSOR_H

#include "Support/Buffer.h"
#include "Support/ErrorOr.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/ML/DType.h"
#include "llvm/Support/JSON.h"
#include <filesystem>
#include <utility>

namespace M {
namespace AsyncRT {
class DeviceContext;
using DeviceContextRef = RCRef<const DeviceContext>;
}; // namespace AsyncRT

/// SafeTensor format reader with GPU-aware data loading capabilities.
/// Provides safe access to tensor data and metadata with error checking.
struct SafeTensor {

  /// TensorInfo is a helper class that provides a JSON-serializable
  /// representation of tensor information.
  struct TensorInfo {
    TensorInfo(DType dtype, ArrayRef<size_t> shape,
               std::pair<size_t, size_t> offsets)
        : dtype(dtype), shape(shape), offsets(offsets) {}

    static ErrorOr<TensorInfo> get(const llvm::json::Object &tensorInfo);

    size_t getSizeInBytes() const;

    llvm::json::Value toJSON() const;

    DType dtype;
    SmallVector<size_t> shape;
    std::pair<size_t, size_t> offsets;
  };

  /// Constructs a SafeTensor from parsed metadata and tensor information.
  SafeTensor(BufferRef buffer, llvm::json::Object &&metadata,
             llvm::DenseMap<std::string, TensorInfo> &&tensorInfo)
      : buffer(buffer), metadata(std::move(metadata)),
        tensorInfo(std::move(tensorInfo)) {}

  /// Loads and parses a SafeTensor file from the given data.
  static ErrorOr<SafeTensor> get(BufferRef data);

  /// Loads and parses a SafeTensor file from the given path.
  static ErrorOr<SafeTensor> get(const std::filesystem::path &path);

  /// Serialize the SafeTensor to a JSON object.
  llvm::json::Value toJSON() const;

  /// Gets the buffer for the SafeTensor.
  BufferRef getBuffer() const { return buffer; }

  /// Gets the metadata for the SafeTensor.
  llvm::json::Object getMetadata() const { return metadata; }

  /// Gets the tensor info for the SafeTensor.
  llvm::DenseMap<std::string, TensorInfo> getTensorInfo() const {
    return tensorInfo;
  }

  /// Gets metadata for a specific tensor.
  ErrorOr<SafeTensor::TensorInfo> getTensorInfo(StringRef name) const;

private:
  BufferRef buffer;
  llvm::json::Object metadata;
  llvm::DenseMap<std::string, TensorInfo> tensorInfo;
};

struct SafeTensorBundle {
  SafeTensorBundle(DenseMap<std::string, size_t> tensorNameToIndex,
                   SmallVector<SafeTensor> files)
      : tensorNameToIndex(std::move(tensorNameToIndex)),
        files(std::move(files)) {}

  /// Loads and parses a SafeTensorBundle from the given paths.
  static ErrorOr<SafeTensorBundle> get(ArrayRef<std::filesystem::path> paths);

  /// Gets metadata for a specific tensor.
  ErrorOr<SafeTensor::TensorInfo> getTensorInfo(StringRef name) const;

  /// Gets the Tensor data for a specific tensor.
  /// TODO: Honor the device argument.
  ErrorOr<BufferRef> getTensorData(StringRef name,
                                   AsyncRT::DeviceContextRef device) const;

private:
  DenseMap<std::string, size_t> tensorNameToIndex;
  SmallVector<SafeTensor> files;
};

}; // namespace M

// Provide the DenseMapInfo for std::string so we can use it in llvm::DenseMaps.
namespace llvm {
template <>
struct DenseMapInfo<std::string> : DenseMapInfo<StringRef> {
  static std::string getEmptyKey() { return ""; }
  static std::string getTombstoneKey() {
    return DenseMapInfo<StringRef>::getTombstoneKey().str();
  }
  static unsigned getHashValue(const std::string &str) {
    return DenseMapInfo<StringRef>::getHashValue(str);
  }
  static bool isEqual(const std::string &LHS, const std::string &RHS) {
    return LHS == RHS;
  }
};

} // namespace llvm

#endif // RUNTIME_SAFETENSOR_H
