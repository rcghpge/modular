//===- LLCL/Support/FileLocationDecoder.h ---------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Implements a decoder that only knows the filename of the source file and is
// unable to resolve the line/column information.
//
//===----------------------------------------------------------------------===//

#ifndef LLCL_SUPPORT_FILELOCATIONDECODER_H
#define LLCL_SUPPORT_FILELOCATIONDECODER_H

#include "LLCL/Support/Location.h"
#include "LLCL/Support/ReferenceCounted.h"
#include "Support/LLVMForwardDecls.h"
#include <string>

namespace LLCL {

/// This class implements LocationDecoder and reports just the the file name.
/// FileNameLocationDecoder is used when line and column information cannot be
/// determined.
class FileNameLocationDecoder final
    : public ReferenceCounted<FileNameLocationDecoder>,
      public LocationDecoder {
public:
  FileNameLocationDecoder(llvm::StringRef filename);

  /// Implement the only remaining LocationDecoder hook. The only thing we
  /// can decode is the file name.
  LLCL::DecodedLocation decode(const EncodedLocation &loc) const override;
  void addRef() const override;
  void dropRef() const override;

private:
  std::string filename;
};

} // namespace LLCL

#endif // LLCL_SUPPORT_FILELOCATIONDECODER_H
