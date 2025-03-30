//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// SafeTensor reader which is GPU aware.
//
//===----------------------------------------------------------------------===//

#ifndef RUNTIME_IO_SAFETENSOR_H
#define RUNTIME_IO_SAFETENSOR_H

#include "Support/Buffer.h"
#include "Support/ErrorOr.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/ML/DType.h"
#include "llvm/Support/JSON.h"
#include <filesystem>
#include <utility>

namespace M {

/// SafeTensorInfo is a helper class that provides a JSON-serializable
/// representation of tensor information.
struct SafeTensorInfo {
  SafeTensorInfo(DType dtype, ArrayRef<size_t> shape,
                 std::pair<size_t, size_t> offsets)
      : dtype(dtype), shape(shape), offsets(offsets) {}

  static ErrorOr<SafeTensorInfo> get(const llvm::json::Object &tensorInfo);

  size_t getSizeInBytes() const;

  llvm::json::Value toJSON() const;

  DType dtype;
  SmallVector<size_t> shape;
  std::pair<size_t, size_t> offsets;
};

/// SafeTensor format reader with GPU-aware data loading capabilities.
/// Provides safe access to tensor data and metadata with error checking.
struct SafeTensor {
  /// Constructs a SafeTensor from parsed metadata and tensor information.
  SafeTensor(BufferRef buffer, llvm::json::Object &&metadata,
             llvm::DenseMap<std::string, llvm::json::Object> &&tensorInfo)
      : buffer(buffer), metadata(std::move(metadata)),
        tensorInfo(std::move(tensorInfo)) {}

  /// Retrieves the top-level metadata for the SafeTensor file.
  llvm::json::Object getMetadata() const { return metadata; }

  /// Gets metadata for a specific tensor.
  ErrorOr<SafeTensorInfo> getTensorInfo(StringRef name) const;

  /// Loads and parses a SafeTensor file from the given data.
  static ErrorOr<SafeTensor> get(BufferRef data);

  /// Loads and parses a SafeTensor file from the given path.
  static ErrorOr<SafeTensor> get(const std::filesystem::path &path);

  /// Serialize the SafeTensor to a JSON object.
  llvm::json::Value toJSON() const;

private:
  BufferRef buffer;
  llvm::json::Object metadata;
  llvm::DenseMap<std::string, llvm::json::Object> tensorInfo;
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

#endif // RUNTIME_IO_SAFETENSOR_H
