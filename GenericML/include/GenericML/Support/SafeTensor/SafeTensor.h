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

#include "GenericML/DeviceDriver/Driver.h"
#include "MLRT/AsyncRT/DeviceContext/DeviceContext.h"
#include "Support/ErrorOr.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/ML/DType.h"
#include "Support/RCRef.h"
#include "Support/SymbolExport.h"
#include "llvm/Support/JSON.h"
#include <filesystem>
#include <utility>

namespace M {

/// SafeTensor format reader with GPU-aware data loading capabilities.
/// Provides safe access to tensor data and metadata with error checking.
struct MODULAR_CXX_EXPORT SafeTensor {

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

  /// Loads and parses a SafeTensor file from the given data.
  static ErrorOr<SafeTensor>
  get(RCRef<Driver::Device> device, AsyncRT::DeviceBufferRef data,
      std::optional<llvm::sys::fs::mapped_file_region> mmap);

  /// Loads and parses a SafeTensor file from the given path.
  static ErrorOr<SafeTensor> get(const std::filesystem::path &path,
                                 RCRef<Driver::Device> device);

  /// Serialize the SafeTensor to a JSON object.
  llvm::json::Value toJSON() const;

  /// Gets the metadata for the SafeTensor.
  llvm::json::Object getMetadata() const { return metadata; }

  /// Gets the tensor info for the SafeTensor.
  const llvm::DenseMap<std::string, TensorInfo> &getTensorInfo() const {
    return tensorInfo;
  }

  /// Gets metadata for a specific tensor.
  ErrorOr<SafeTensor::TensorInfo> getTensorInfo(StringRef name) const;

  /// Gets a tensor from given info.
  std::shared_ptr<Driver::DeviceMemory>
  getTensorMemory(const TensorInfo &info) const;

private:
  /// Constructs a SafeTensor from parsed metadata and tensor information.
  SafeTensor(uint64_t headerLength, RCRef<Driver::Device> &&device,
             AsyncRT::DeviceBufferRef &&parentBuffer,
             llvm::json::Object &&metadata,
             llvm::DenseMap<std::string, TensorInfo> &&tensorInfo)
      : headerLength(headerLength), device(std::move(device)),
        parentBuffer(std::move(parentBuffer)), metadata(std::move(metadata)),
        tensorInfo(std::move(tensorInfo)) {}

  uint64_t headerLength;
  // The device the parent buffer lives on
  RCRef<Driver::Device> device;
  // Parent buffer for all subbuffers created.
  AsyncRT::DeviceBufferRef parentBuffer;
  // Holds a memmap RAII object if this SafeTensor object is backed by memmap'd
  // storage. This gets shared with any tensors created from it.
  std::optional<std::shared_ptr<llvm::sys::fs::mapped_file_region>> mmap;
  llvm::json::Object metadata;
  llvm::DenseMap<std::string, TensorInfo> tensorInfo;
};

struct SafeTensorBundle {
  SafeTensorBundle(DenseMap<std::string, size_t> tensorNameToIndex,
                   SmallVector<SafeTensor> files)
      : tensorNameToIndex(std::move(tensorNameToIndex)),
        files(std::move(files)) {}

  /// Loads and parses a SafeTensorBundle from the given paths.
  static ErrorOr<SafeTensorBundle> get(ArrayRef<std::filesystem::path> paths,
                                       RCRef<Driver::Device> ctx);

  /// Gets a tensor from given info.
  ErrorOr<std::shared_ptr<Driver::DeviceMemory>>
  getTensorMemory(StringRef name) const;

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
