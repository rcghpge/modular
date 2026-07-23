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

#include "GraphCompiler/Driver/Device/Driver.h"
#include "MLRT/Driver/DeviceContext/DeviceContext.h"
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
  static ErrorOr<SafeTensor> get(RCRef<M::GraphCompiler::Driver::Device> device,
                                 M::Driver::DeviceBufferRef data);

  /// Loads and parses a SafeTensor file from the given path.
  static ErrorOr<SafeTensor>
  get(const std::filesystem::path &path,
      RCRef<M::GraphCompiler::Driver::Device> device);

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
  std::shared_ptr<M::GraphCompiler::Driver::DeviceMemory>
  getTensorMemory(const TensorInfo &info) const;

private:
  /// Constructs a SafeTensor from parsed metadata and tensor information.
  SafeTensor(uint64_t headerLength,
             RCRef<M::GraphCompiler::Driver::Device> &&device,
             M::Driver::DeviceBufferRef &&parentBuffer,
             llvm::json::Object &&metadata,
             llvm::DenseMap<std::string, TensorInfo> &&tensorInfo)
      : headerLength(headerLength), device(std::move(device)),
        parentBuffer(std::move(parentBuffer)), metadata(std::move(metadata)),
        tensorInfo(std::move(tensorInfo)) {}

  uint64_t headerLength;
  // The device the parent buffer lives on
  RCRef<M::GraphCompiler::Driver::Device> device;
  // Parent buffer for all subbuffers created.
  M::Driver::DeviceBufferRef parentBuffer;
  llvm::json::Object metadata;
  llvm::DenseMap<std::string, TensorInfo> tensorInfo;
};

struct SafeTensorBundle {
  SafeTensorBundle(DenseMap<std::string, size_t> tensorNameToIndex,
                   SmallVector<SafeTensor> files)
      : tensorNameToIndex(std::move(tensorNameToIndex)),
        files(std::move(files)) {}

  /// Loads and parses a SafeTensorBundle from the given paths.
  static ErrorOr<SafeTensorBundle>
  get(ArrayRef<std::filesystem::path> paths,
      RCRef<M::GraphCompiler::Driver::Device> ctx);

  /// Gets a tensor from given info.
  ErrorOr<std::shared_ptr<M::GraphCompiler::Driver::DeviceMemory>>
  getTensorMemory(StringRef name) const;

private:
  DenseMap<std::string, size_t> tensorNameToIndex;
  SmallVector<SafeTensor> files;
};

}; // namespace M

#include "Support/ADT/DenseStringMap.h" // IWYU pragma: keep

#endif // RUNTIME_SAFETENSOR_H
