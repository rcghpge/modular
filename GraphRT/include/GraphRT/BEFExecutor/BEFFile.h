//===- GraphRT/BEFExecutor/BEFFile.h --------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GRAPHRT_BEFEXECUTOR_BEFFILE_H
#define GRAPHRT_BEFEXECUTOR_BEFFILE_H

#include "LLCL/ForwardDecls.h"
#include "LLCL/Support/ReferenceCounted.h"
#include "Support/ErrorOr.h"
#include "Support/LLVM.h"

namespace M {
class Invokable;
class PrimitiveRegistry;

/// Instances of this class represent opened and decoded BEF files, allowing
/// clients to interact with the invokable list, etc.
class BEFFile : public LLCL::ReferenceCounted<BEFFile> {
public:
  virtual ~BEFFile();

  /// Open and read a BEF file, setting up our internal state and returning a
  /// pointer to our initialized object on success.  On an initial parse
  /// failure, an error message is returned in the `ErrorOr`.  BEF file format
  /// errors discovered lazily when running invokables, so they are returned
  /// through the invokable's result AsyncValue's.
  ///
  /// TODO: This should (optionally) manage ownership of the underlying data
  /// passed in.  For example, we could take a closure to run when the lifetime
  /// of the BEFFile is done, or have a reference counted abstraction for
  /// holding memory buffers.
  static ErrorOr<RCRef<BEFFile>> open(StringRef file,
                                      const PrimitiveRegistry &primitives);

  /// Get a list of invokables out of the BEF file.
  std::vector<const Invokable *> getAllInvokables() const;

  /// Find an invokable with the specified name, returning null if it isn't
  /// found.
  const Invokable *getInvokable(StringRef name) const;

protected:
  BEFFile() {}
};

} // namespace M

#endif // GRAPHRT_BEFEXECUTOR_BEFFILE_H
