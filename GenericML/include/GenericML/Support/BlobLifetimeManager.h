//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_SUPPORT_BLOBLIFETIMEMANAGER_H
#define GENERICML_SUPPORT_BLOBLIFETIMEMANAGER_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/FunctionExtras.h"

namespace M::GML {

/// Helper to allow for resource blob ownership to be transferred and freed
/// earlier by buffers.
struct BlobLifetimeManager {
  BlobLifetimeManager(
      llvm::unique_function<void(void *data, size_t size, size_t align)>
          deleter,
      llvm::ArrayRef<char> data, size_t alignment)
      : deleter(std::move(deleter)), data(data), alignment(alignment) {}
  ~BlobLifetimeManager() {
    deleter(const_cast<char *>(data.data()), data.size(), alignment);
  }
  llvm::unique_function<void(void *data, size_t size, size_t align)> deleter;
  llvm::ArrayRef<char> data;
  size_t alignment;
};

} // namespace M::GML

#endif // GENERICML_SUPPORT_BLOBLIFETIMEMANAGER_H
