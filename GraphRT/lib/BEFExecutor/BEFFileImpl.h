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
#include "GraphRT/BEFExecutor/Invokable.h"
#include "GraphRT/BEFExecutor/PrimitiveRegistry.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringMap.h"

namespace M {
class BEFInvokable;

/// This class is the implementation details behind the BEFFile::Open method,
/// which maintains all the state necessary for the BEFExecutor.
class BEFFileImpl final : public BEFFile {
public:
  BEFFileImpl() = default;
  ~BEFFileImpl() override;

  /// Return the invokable with the specified invokable ID.
  const Invokable *getInvokableByID(size_t number) const;

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
  ArrayRef<uint8_t> locationFilenames;
  ArrayRef<uint8_t> locationPositions;
  ArrayRef<uint8_t> strings;
  ArrayRef<uint8_t> attributes;
  ArrayRef<uint8_t> invokableBodies;

  /// This is the list of primitive implementations arranged by their ID.
  std::vector<PrimitiveRegistry::PrimitiveImplFn> primitives;

  /// This map holds the index in `invokables` of invokables with the specified
  /// name.
  ///
  /// TODO: This data structure is major overkill: it contributes significant
  /// code size and each name inserted into this is a malloc. See:
  /// https://github.com/modularml/modular/issues/11 for more info.
  llvm::StringMap<size_t> invokableSymbolTable;

  /// This contains all of the BEFInvokable records inline in dense memory even
  /// though clients treat them like individual objects.  Their lifetime is
  /// completely bound to BEFFileImpl's.
  const BEFInvokable *invokables = nullptr;
  size_t numInvokables = 0;
};

/// BEF specific implementation of the Invokable interface.
class BEFInvokable final : public Invokable {
public:
  BEFInvokable(StringRef name, uint16_t numArguments, uint16_t numResults,
               uint32_t bodyOffset, uint32_t locationOffset, BEFFileImpl *owner)
      : Invokable(name, numArguments, numResults), bodyOffset(bodyOffset),
        locationOffset(locationOffset), owner(owner) {}

  virtual ~BEFInvokable() = default;

  /// Invokable this using the specified runtime, passing the specified
  /// arguments. This returns one AsyncValue for each result.
  virtual void invoke(MutableArrayRef<RCRef<AsyncValue>> arguments,
                      MutableArrayRef<RCRef<AsyncValue>> results,
                      Runtime &runtime) const override;

  /// Add a new reference to this object.  The lifetime of invokables are
  /// generally tied to the file they are loaded from (e.g. a BEFFile) so they
  /// don't need explicit ownership management in most cases, but some cases
  /// that can extend their lifetime (e.g. implementations of async
  /// control flow operations) should extend their lifetime.
  virtual void addRef() const override;

  /// Drop a reference to this object, potentially deallocating it.
  virtual void dropRef() const override;

public:
  const uint32_t bodyOffset, locationOffset;
  BEFFileImpl *const owner;
};

} // namespace M

#endif // BEFFILEIMPL_H
