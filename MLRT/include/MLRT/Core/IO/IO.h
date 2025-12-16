//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef RUNTIME_IO_H
#define RUNTIME_IO_H

#include "Support/ErrorOr.h"
#include <fcntl.h>
#include <filesystem>

#include "MLRT/Core/IO/CUFile.h"
#include "Support/ADT/SmartVariant.h"

namespace M::IO {
struct File {
  /// Initializes the cuFile library and driver resources. Must be called before
  /// any other cuFile API function. This would also register the uninitializer
  /// for you.
  static ErrorOrSuccess init();

  /// Registers a device memory buffer with cuFile for direct DMA transfers.
  ErrorOrSuccess registerMemory(void *devPtr, size_t size);

  /// Deregisters a previously registered device memory buffer.
  ErrorOrSuccess deregisterMemory(void *devPtr);

  /// Opens a file and registers it with cuFile.
  ErrorOrSuccess open(const std::filesystem::path &path, int flags = O_RDONLY);

  /// Deregisters the file handle from cuFile and closes the file descriptor.
  ErrorOrSuccess close();

  /// Reads data directly from the file into a registered device buffer.
  ErrorOr<ssize_t> read(void *devPtr, size_t size, off_t fileOffset,
                        off_t devOffset);

  /// Writes data directly from a registered device buffer to the file.
  ErrorOr<ssize_t> write(void *devPtr, size_t size, off_t fileOffset,
                         off_t devOffset);

private:
  SmartVariant<CUFile> impl;
};
}; // namespace M::IO

#endif // RUNTIME_IO_H
