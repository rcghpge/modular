//===- BEFFileImpl.h --------------------------------------------*- C++ -*-===//
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
#include "GraphRT/BEFExecutor/Function.h"
#include "GraphRT/BEFExecutor/PrimitiveRegistry.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringMap.h"

namespace M {
class BEFFunction;

/// This class is the implementation details behind the BEFFile::Open method,
/// which maintains all the state necessary for the BEFExecutor.
class BEFFileImpl : public BEFFile {
public:
  BEFFileImpl() {}
  ~BEFFileImpl() override;

  /// Return the function with the specified BEF function ID.
  const Function *getFunctionByID(size_t number) const;

  // Everything in this class is public because the entire class is private to
  // the BEFExecutor library implementation.
public:
  // This struct describes one function in the FunctionIndex section.
  struct FunctionIndex {
    uint32_t offsetInFunctionSection;
    uint32_t offsetOfNameInStringSection;
    uint32_t offsetOfLocation;
    uint16_t numArgs, numResults;
  };

  // These are sections that we expect to be in the BEF file.  They are lazily
  // decoded when executing functions.
  ArrayRef<uint8_t> locationFilenames;
  ArrayRef<uint8_t> locationPositions;
  ArrayRef<uint8_t> strings;
  ArrayRef<uint8_t> attributes;
  ArrayRef<uint8_t> befFunctions;

  /// This is the list of primitive implementations arranged by their ID.
  std::vector<PrimitiveRegistry::PrimitiveImplFn> primitives;

  /// This map holds the index in `functions` of functions with the specified
  /// name.
  ///
  /// TODO: This data structure is major overkill: it contributes significant
  /// code size and each name inserted into this is a malloc. See:
  /// https://github.com/modularml/modular/issues/11 for more info.
  llvm::StringMap<size_t> functionSymbolTable;

  /// This contains all of the BEFFunction records inline in dense memory even
  /// though clients treat them like individual objects.  Their lifetime is
  /// completely bound to BEFFileImpl's.
  const BEFFunction *functions = nullptr;
  size_t numFunctions = 0;
};

/// BEF specific implementation of the Function interface.
class BEFFunction final : public Function {
public:
  BEFFunction(StringRef name, uint16_t numArguments, uint16_t numResults,
              uint32_t functionOffset, uint32_t locationOffset,
              BEFFileImpl *owner)
      : Function(name, numArguments, numResults),
        functionOffset(functionOffset), locationOffset(locationOffset),
        owner(owner) {}

  virtual ~BEFFunction() = default;

  /// Execute this function using the specified runtime, passing the specified
  /// arguments. This returns one AsyncValue for each result.
  virtual void execute(MutableArrayRef<RCRef<AsyncValue>> arguments,
                       MutableArrayRef<RCRef<AsyncValue>> results,
                       Runtime &runtime) const override;

  /// Add a new reference to this object.  The lifetime of functions are
  /// generally tied to the file they are loaded from (e.g. a BEFFile) so they
  /// don't need explicit ownership management in most cases, but some cases
  /// that can extend their lifetime (e.g. implementations of async
  /// control flow operations) should extend their lifetime.
  virtual void addRef() const override;

  /// Drop a reference to this object, potentially deallocating it.
  virtual void dropRef() const override;

public:
  const uint32_t functionOffset, locationOffset;
  BEFFileImpl *const owner;
};

} // namespace M

#endif // BEFFILEIMPL_H
