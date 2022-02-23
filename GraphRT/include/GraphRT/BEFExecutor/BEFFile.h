//===- GraphRT/BEFExecutor/BEFFile.h ----------------------------*- C++ -*-===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GRAPHRT_BEFEXECUTOR_BEFFILE_H
#define GRAPHRT_BEFEXECUTOR_BEFFILE_H

#include "LLCL/ForwardDecls.h"
#include "LLCL/Support/ReferenceCounted.h"
#include "Support/LLVM.h"

namespace M {
class PrimitiveRegistry;

/// Instances of this class represent opened an decoded BEF files, allowing
/// clients to interact with the function list, invoke functions to execute,
/// etc.
class BEFFile : public LLCL::ReferenceCounted<BEFFile> {
public:
  virtual ~BEFFile();

  /// Open and read a BEF file, setting up our internal state and returning a
  /// pointer to our initialized object on success.  On failure, an error
  /// message is emitted to the errorHandler and a null RCRef is returned.
  ///
  /// TODO: This should (optionally) manage ownership of the underlying data
  /// passed in.  For example, we could take a closure to run when the lifetime
  /// of the BEFFile is done, or have a reference counted abstraction for
  /// holding memory buffers.
  static RCRef<BEFFile> open(StringRef file, Runtime &runtime);

protected:
  BEFFile() {}
};

} // namespace M

#endif // GRAPHRT_BEFEXECUTOR_BEFFILE_H
