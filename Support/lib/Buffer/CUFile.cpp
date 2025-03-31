//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "./CUFile.h"
#include "Support/PlatformUtils.h"

#include "llvm/ADT/Twine.h"
#include <errno.h>  // For errno
#include <string.h> // For strerror

#ifdef MODULAR_LINUX
/// We do not want to depend on cuda.h, so we define a minimal set of types that
/// are defined by cuda.h here.
using CUresult = int;
using CUstream = void *;
#include <cufile.h>
#include <math.h>

/// We need to define abs for CUfileError_t since IS_CUFILE_ERR uses abs in it.
static int abs(CUfileError_t err) { return abs((int)err.err); }
#endif // MODULAR_LINUX

using namespace M;

#ifdef MODULAR_LINUX
#define RETURN_ERROR_IF_INVALID_OS()                                           \
  do {                                                                         \
  } while (0)
#else // MODULAR_LINUX
#define RETURN_ERROR_IF_INVALID_OS()                                           \
  return Error("CUFile is not supported on this operating system");
#endif // MODULAR_LINUX

/// This function must be called before any other cuFile API function.
/// It initializes the cuFile library and driver resources.
ErrorOrSuccess CUFile::openDriver() {
  RETURN_ERROR_IF_INVALID_OS();

  CUfileError_t status = cuFileDriverOpen();
  if (IS_CUFILE_ERR(status))
    return Error(CUFILE_ERRSTR(status));

  return success();
}

/// This function releases resources acquired by `openDriver`.
/// It should be called when the application no longer needs cuFile services.
ErrorOrSuccess CUFile::closeDriver() {
  RETURN_ERROR_IF_INVALID_OS();

  CUfileError_t status = cuFileDriverClose();
  if (IS_CUFILE_ERR(status))
    return Error(CUFILE_ERRSTR(status));

  return success();
}

/// Registering memory allows cuFile to perform direct DMA transfers
/// between the storage and the GPU memory.
ErrorOrSuccess CUFile::registerMemory(void *devPtr, size_t size) {
  RETURN_ERROR_IF_INVALID_OS();

  CUfileError_t status = cuFileBufRegister(devPtr, size, 0);
  if (IS_CUFILE_ERR(status))
    return Error(CUFILE_ERRSTR(status));

  return success();
}

/// Deregisters a previously registered device memory buffer.
ErrorOrSuccess CUFile::deregisterMemory(void *devPtr) {
  RETURN_ERROR_IF_INVALID_OS();

  CUfileError_t status = cuFileBufDeregister(devPtr);
  if (IS_CUFILE_ERR(status))
    return Error(CUFILE_ERRSTR(status));

  return success();
}

/// Opens the specified file using standard POSIX open, ensuring O_DIRECT
/// is included in the flags for direct I/O. It then registers the file
/// descriptor with cuFile.
ErrorOrSuccess CUFile::open(const std::filesystem::path &path, int flags) {
  RETURN_ERROR_IF_INVALID_OS();

  // Ensure O_DIRECT is set for GDS operations.
  fd = ::open(path.c_str(), flags | O_DIRECT);
  if (fd < 0)
    return Error(Twine("failed to open file '") + path.string() +
                 "': " + strerror(errno));

  CUfileDescr_t descr{};
  descr.handle.fd = fd;
  descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;

  CUfileError_t status = cuFileHandleRegister(&handle, &descr);
  if (IS_CUFILE_ERR(status)) {
    ::close(fd); // Use ::close to avoid ambiguity
    fd = -1;     // Reset fd after closing
    return Error(CUFILE_ERRSTR(status));
  }

  return success();
}

/// Deregisters the file handle from cuFile and closes the underlying
/// file descriptor.
ErrorOrSuccess CUFile::close() {
  RETURN_ERROR_IF_INVALID_OS();

  if (handle) {
    cuFileHandleDeregister(handle);
    handle = nullptr;
  }
  if (fd >= 0) {
    ::close(fd);
    fd = -1;
  }

  return success();
}

/// Reads data directly from the file into a registered device buffer.
ErrorOr<ssize_t> CUFile::read(void *devPtr, size_t size, off_t fileOffset,
                              off_t devOffset) {
  RETURN_ERROR_IF_INVALID_OS();

  if (!handle)
    return Error("cuFile handle is not initialized (file not open?)");
  if (fd < 0)
    return Error("File descriptor is invalid (file not open?)");

  ssize_t bytesRead = cuFileRead(handle, devPtr, size, fileOffset, devOffset);

  if (bytesRead == -1)
    return Error(Twine("failed to read from cuFile because of '") +
                 strerror(errno) + "'");

  if (bytesRead < 0)
    return Error(Twine("cuFile read failed: ") + CUFILE_ERRSTR(bytesRead));

  return bytesRead;
}

/// Writes data directly from a registered device buffer to the file.
ErrorOr<ssize_t> CUFile::write(void *devPtr, size_t size, off_t fileOffset,
                               off_t devOffset) {
  RETURN_ERROR_IF_INVALID_OS();

  if (!handle)
    return Error("cuFile handle is not initialized (file not open?)");
  if (fd < 0)
    return Error("File descriptor is invalid (file not open?)");

  ssize_t bytesWritten =
      cuFileWrite(handle, devPtr, size, fileOffset, devOffset);

  if (bytesWritten == -1)
    return Error(Twine("failed to read from cuFile because of '") +
                 strerror(errno) + "'");

  // Check for errors reported by cuFileWrite (negative return value)
  if (bytesWritten < 0)
    return Error(Twine("cuFile write failed: ") + CUFILE_ERRSTR(bytesWritten));

  return bytesWritten;
}
