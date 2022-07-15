//===- BEFFileImpl.h ------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This is the implementation logic for the BEFExecutor.
//
//===----------------------------------------------------------------------===//

#ifndef BEFFILEIMPL_H
#define BEFFILEIMPL_H

#include "GraphRT/BEFExecutor/BEFFile.h"
#include "GraphRT/BEFExecutor/BEFInvokable.h"
#include "GraphRT/BEFExecutor/PrimitiveRegistry.h"
#include "llvm/ADT/ArrayRef.h"

namespace M {
class BEFInvokableImpl;

/// This class is the implementation details behind the BEFFile::Open method,
/// which maintains all the state necessary for the BEFExecutor.
class BEFFileImpl final : public BEFFile {
public:
  BEFFileImpl() = default;
  ~BEFFileImpl() override;

  /// Return the invokable with the specified invokable ID.
  BEFInvokable *getInvokableByID(size_t number) const;

  /// This method enumerates the primitive name table, calling the specified
  /// callable on each entry.  The callable function may return an error to stop
  /// iteration and diagnose problems, if it does, it is propagated out.
  ErrorOrSuccess
  enumeratePrimitiveNames(const std::function<ErrorOrSuccess(StringRef)>& callbackFn);

  // Everything in this class is public because the entire class is private to
  // the BEFExecutor library implementation.
public:
  // This struct describes one invokable in the InvokableIndex section.
  struct InvokableIndex {
    uint32_t offsetInInvokableBodiesSection;
    uint32_t offsetOfNameInStringSection;
    uint32_t offsetOfLocation;
    uint16_t numArgs, numResults;
  };

  // These are sections that we expect to be in the BEF file.  They are lazily
  // decoded when executing invokables.
  ArrayRef<uint8_t> primitiveNames;
  ArrayRef<uint8_t> locationFilenames;
  ArrayRef<uint8_t> locationPositions;
  ArrayRef<uint8_t> strings;
  ArrayRef<uint8_t> attributes;
  ArrayRef<uint8_t> invokableBodies;

  /// This is the list of primitive implementations arranged by their ID.
  std::vector<PrimitiveRegistry::PrimitiveImplFn> primitives;

  /// This contains all of the BEFInvokable records inline in dense memory even
  /// though clients treat them like individual objects.  Their lifetime is
  /// completely bound to BEFFileImpl's.
  BEFInvokableImpl *invokables = nullptr;
  size_t numInvokables = 0;
};

/// BEF specific implementation of the Invokable interface.
class BEFInvokableImpl final : public BEFInvokable, public LocationDecoder {
public:
  BEFInvokableImpl(uint16_t numArguments, uint16_t numResults,
                   uint32_t bodyOffset, BEFFileImpl *owner)
      : BEFInvokable(numArguments, numResults), bodyOffset(bodyOffset),
        owner(owner) {}

  ~BEFInvokableImpl() override = default;

  /// Visit each primitive in this invokable.
  ErrorOrSuccess walkPrimitives(
      std::function<ErrorOrSuccess(StringRef, const PrimitiveDescriptor &)>
          visitor) const override;

  /// Invokable this using the specified runtime, passing the specified
  /// arguments. This returns one AsyncValue for each result.
  void invoke(MutableArrayRef<AnyAsyncValueRef> arguments,
              MutableArrayRef<AnyAsyncValueRef> results,
              Runtime &runtime) const override;

  /// Add a new reference to this object.  The lifetime of invokables are
  /// generally tied to the file they are loaded from (e.g. a BEFFile) so they
  /// don't need explicit ownership management in most cases, but some cases
  /// that can extend their lifetime (e.g. implementations of async
  /// control flow operations) should extend their lifetime.
  void addRef() const override;

  /// Drop a reference to this object, potentially deallocating it.
  void dropRef() const override;

  /// Implement the only remaining LocationDecoder hook. Since we have a pointer
  /// to the BEFFileImpl we can easily decode a location provided to us.
  LLCL::DecodedLocation decode(const EncodedLocation &loc) const override;

public:
  const uint32_t bodyOffset;
  BEFFileImpl *const owner;
};

} // namespace M

#endif // BEFFILEIMPL_H
