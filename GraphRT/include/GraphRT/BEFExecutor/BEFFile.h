//===- GraphRT/BEFExecutor/BEFFile.h --------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GRAPHRT_BEFEXECUTOR_BEFFILE_H
#define GRAPHRT_BEFEXECUTOR_BEFFILE_H

#include "GraphRT/BEFExecutor/TopLevelEntity.h"
#include "LLCL/ForwardDecls.h"
#include "LLCL/Support/ReferenceCounted.h"
#include "Support/ErrorOr.h"
#include "Support/LLVMForwardDecls.h"
#include "llvm/ADT/ArrayRef.h"

namespace M {
class BEFInvokable;
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

  /// Get a list of top level ops out of the BEF file.
  ArrayRef<TopLevelEntity> getAllTopLevelEntities() const;

  /// Get a list of all the top level ops in a BEF file of the specified type
  /// `T`
  template <typename T>
  SmallVector<T, 1> getAllTopLevelEntities() const {
    SmallVector<T, 1> out;
    for (const TopLevelEntity &op : getAllTopLevelEntities())
      if (const auto &t = dyn_cast<T>(op))
        out.push_back(std::move(*t));

    return out;
  }

  /// Find a top level op with the specified name, returning null if it isn't
  /// found.
  Optional<const TopLevelEntity> getTopLevelEntity(StringRef primitive,
                                                   StringRef symbol) const;

  /// Find a top level op with some nice sugar to avoid having to pass in a
  /// string directly.
  template <typename T>
  Optional<const T> getTopLevelEntity(StringRef symbol) const {
    return llvm::dyn_cast_if_present<T>(
        getTopLevelEntity(T::getEntityName(), symbol));
  }

protected:
  BEFFile() {}
};

} // namespace M

#endif // GRAPHRT_BEFEXECUTOR_BEFFILE_H
