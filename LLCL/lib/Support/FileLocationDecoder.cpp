//===----------------------------------------------------------------------===//
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

using namespace M::LLCL;

/// The FileNameLocationDecoder constructor just stores the filename.
FileNameLocationDecoder::FileNameLocationDecoder(llvm::StringRef filename)
    : filename(filename) {}

/// Gets and Encoded Location object that contains the filename.
EncodedLocation
FileNameLocationDecoder::getEncodedLocation(llvm::StringRef filename) {
  return EncodedLocation(0, RCRef<FileNameLocationDecoder>::create(filename));
}

/// Decode the file name information in this object into a DecodedLocation.
M::LLCL::DecodedLocation
FileNameLocationDecoder::decode(const EncodedLocation &loc) const {
  return M::LLCL::DecodedLocation{filename};
}

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
