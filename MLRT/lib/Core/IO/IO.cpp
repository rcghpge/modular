//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "MLRT/Core/IO/IO.h"

using namespace M;
using namespace M::IO;

ErrorOrSuccess File::init() {
  // TODO: Implement this.
  // if ( isa<CUFile>(impl))
  //   return cast<CUFile>(impl).init();

  return Error("Unsupported IO backend");
}

ErrorOrSuccess File::registerMemory(void *devPtr, size_t size) {
  if (isa<CUFile>(impl))
    return cast<CUFile>(impl).registerMemory(devPtr, size);

  return Error("Unsupported IO backend");
}

ErrorOrSuccess File::deregisterMemory(void *devPtr) {
  if (isa<CUFile>(impl))
    return cast<CUFile>(impl).deregisterMemory(devPtr);

  return Error("Unsupported IO backend");
}

ErrorOrSuccess File::open(const std::filesystem::path &path, int flags) {
  if (isa<CUFile>(impl))
    return cast<CUFile>(impl).open(path, flags);

  return Error("Unsupported IO backend");
}

ErrorOrSuccess File::close() {
  if (isa<CUFile>(impl))
    return cast<CUFile>(impl).close();

  return Error("Unsupported IO backend");
}

ErrorOr<ssize_t> File::read(void *devPtr, size_t size, off_t fileOffset,
                            off_t devOffset) {
  if (isa<CUFile>(impl))
    return cast<CUFile>(impl).read(devPtr, size, fileOffset, devOffset);

  return Error("Unsupported IO backend");
}

ErrorOr<ssize_t> File::write(void *devPtr, size_t size, off_t fileOffset,
                             off_t devOffset) {
  if (isa<CUFile>(impl))
    return cast<CUFile>(impl).write(devPtr, size, fileOffset, devOffset);

  return Error("Unsupported IO backend");
}
