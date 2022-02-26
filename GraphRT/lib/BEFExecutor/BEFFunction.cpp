//===- BEFFunction.cpp ----------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements logic required to execute a function.  This is the core
// execution logic for the BEF executor runtime.
//
//===----------------------------------------------------------------------===//

#include "GraphRT/BEFExecutor/Function.h"
#include "llvm/Support/raw_ostream.h"

#include "BEFFileImpl.h"
using namespace M;

//===----------------------------------------------------------------------===//
// Function implementation
//===----------------------------------------------------------------------===//

// NOTE: This could be moved out to somewhere non-BEF specific if there are
// other kinds of functions someday.
void Function::vtableAnchor() {}

//===----------------------------------------------------------------------===//
// BEFFunction
//===----------------------------------------------------------------------===//

/// Add a new reference to this object.  The lifetime of BEF functions are
/// completely tied to BEFFileImpl, so just extend its reference count.
void BEFFunction::addRef() const { owner->addRef(); }

/// Drop a reference to this object, potentially deallocating it.
void BEFFunction::dropRef() const { owner->dropRef(); }

/// Execute this function, passing the specified arguments. This returns one
/// AsyncValue for each result.
void BEFFunction::execute(ArrayRef<AsyncValue *> arguments,
                          MutableArrayRef<RCRef<AsyncValue>> results) const {
  llvm::errs() << "execute " << getName() << functionOffset << locationOffset
               << "\n";
}