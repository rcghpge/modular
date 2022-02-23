//===- BEFFile.cpp --------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements the main entrypoint for the BEFToMLIR library.
//
//===----------------------------------------------------------------------===//

#include "GraphRT/BEFExecutor/BEFFile.h"
#include "BEFFileImpl.h"
#include "LLCL/Support/RCRef.h"
#include "llvm/ADT/StringRef.h"
using namespace M;

//===----------------------------------------------------------------------===//
// BEFFile
//===----------------------------------------------------------------------===//

/// Home the vtable in a .cpp file.
BEFFile::~BEFFile() {}

//===----------------------------------------------------------------------===//
// BEFFileImpl
//===----------------------------------------------------------------------===//

BEFFileImpl::BEFFileImpl() {}
BEFFileImpl::~BEFFileImpl() {}

/// Open and read a BEF file, setting up our internal state and returning a
/// pointer to our initialized object on success.  On failure, an error
/// message is emitted to the errorHandler and a null RCRef is returned.
///
RCRef<BEFFile> BEFFile::open(StringRef file, Runtime &runtime) {
  ArrayRef<uint8_t> fileData((const uint8_t *)file.data(), file.size());

  auto impl = RCRef<BEFFileImpl>::create();
  impl->file = fileData;
  return std::move(impl);
}