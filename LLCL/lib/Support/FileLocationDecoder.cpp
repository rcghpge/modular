//===- FileLocationDecoder.cpp --------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements FileLocationDecoder.h classes.
//
//===----------------------------------------------------------------------===//

#include "LLCL/Support/FileLocationDecoder.h"
#include "llvm/ADT/StringRef.h"

using namespace LLCL;

/// The FileNameLocationDecoder constructor just stores the filename.
FileNameLocationDecoder::FileNameLocationDecoder(llvm::StringRef filename)
    : filename(filename){};

/// Decode the file name information in this object into a DecodedLocation.
LLCL::DecodedLocation
FileNameLocationDecoder::decode(const EncodedLocation &loc) const {
  return LLCL::DecodedLocation{filename};
};

/// Implement the LocationDecoder hook for addRef.
void FileNameLocationDecoder::addRef() const {
  RCRef<ReferenceCounted<FileNameLocationDecoder>>::lowLevelAddRef(
      const_cast<FileNameLocationDecoder *>(this));
}

/// Implement the LocationDecoder hook for dropRef.
void FileNameLocationDecoder::dropRef() const {
  RCRef<ReferenceCounted<FileNameLocationDecoder>>::lowLevelDropRef(
      const_cast<FileNameLocationDecoder *>(this));
}
