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
  ArrayRef<uint8_t> file;
};

} // namespace M

#endif // BEFFILEIMPL_H
