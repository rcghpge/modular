//===- GraphRT/BEFExecutor/BEFFile.h --------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GRAPHRT_BEFEXECUTOR_BEFFILE_H
#define GRAPHRT_BEFEXECUTOR_BEFFILE_H

#include "GraphRT/BEF/BEFContents.h"
#include "LLCL/ForwardDecls.h"
#include "LLCL/Support/ReferenceCounted.h"
#include "Support/ErrorOr.h"
#include "Support/LLVMForwardDecls.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

namespace M {
class BEFInvokable;
class PrimitiveRegistry;
class PrimitiveDescriptor;

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
  /// Note that this doesn't set up the primitive table, it requires an explicit
  /// call to `installPrimitives` to do so.
  ///
  /// TODO: This should (optionally) manage ownership of the underlying data
  /// passed in.  For example, we could take a closure to run when the lifetime
  /// of the BEFFile is done, or have a reference counted abstraction for
  /// holding memory buffers.
  static ErrorOr<RCRef<BEFFile>> open(StringRef file);

  /// Check that the primitives name table is valid and can be resolved to
  /// entries in the specified registry.
  ErrorOrSuccess installPrimitives(const PrimitiveRegistry &registry);

  /// This method inspects the contents of the BEF file to make an educated
  /// guess of what it contains, allowing for better QoI and usability of tools.
  BEFContents sniffFileContents() const;

  /// Open and read a BEF file, setting up our internal state and returning a
  /// pointer to our initialized object on success.  On an initial parse
  /// failure, an error message is returned in the `ErrorOr`.  BEF file format
  /// errors discovered lazily when running invokables, so they are returned
  /// through the invokable's result AsyncValue's.
  ///
  /// Note that this doesn't set up the primitive table, it requires an explicit
  /// call to `installPrimitives` to do so.
  ///
  /// TODO: This should (optionally) manage ownership of the underlying data
  /// passed in.  For example, we could take a closure to run when the lifetime
  /// of the BEFFile is done, or have a reference counted abstraction for
  /// holding memory buffers.
  static ErrorOr<RCRef<BEFFile>> open(StringRef file,
                                      const PrimitiveRegistry &registry);

  /// Return the top level invokable in the file. This object corresponds to the
  /// MLIR module in the file used to create the BEF. Returns nullptr if the BEF
  /// file is fully empty (unlikely, but technically possible).
  const BEFInvokable *getTopLevelInvokable() const;

protected:
  BEFFile() {}
};

/// Return true if the specified file contents look like it is a BEF file, by
/// checking the magic number.
bool isBEFFile(llvm::StringRef contents);

} // namespace M

#endif // GRAPHRT_BEFEXECUTOR_BEFFILE_H
