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
#include "GraphRT/BEFExecutor/PrimitiveRegistry.h"
#include "llvm/ADT/ArrayRef.h"

namespace M {

/// This class is the implementation details behind the BEFFile::Open method,
/// which maintains all the state necessary for the BEFExecutor.
class BEFFileImpl : public BEFFile {
public:
  explicit BEFFileImpl(Runtime &runtime) : runtime(runtime) {}
  ~BEFFileImpl() override;

public:
  Runtime &runtime;

  // These are sections that we expect to be in the BEF file.  They are lazily
  // decoded when executing functions.
  ArrayRef<uint8_t> locationFilenames;
  ArrayRef<uint8_t> locationPositions;
  ArrayRef<uint8_t> strings;
  ArrayRef<uint8_t> attributes;
  ArrayRef<uint8_t> befFunctions;

  /// This is the list of primitive implementations arranged by their ID.
  std::vector<PrimitiveRegistry::PrimitiveImplFn> primitives;
};

} // namespace M

#endif // BEFFILEIMPL_H
