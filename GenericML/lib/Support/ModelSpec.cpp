//===- ModelSpec.cpp ------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements an in-memory representation for a model specification.
//
//===----------------------------------------------------------------------===//

#include "GenericML/Support/ModelSpec.h"
#include "GenericML/Support/Tensor.h"
#include "LLCL/Runtime/AsyncValue.h"
#include "llvm/ADT/StringRef.h"
using namespace M;

//===----------------------------------------------------------------------===//
// ModelSpec
//===----------------------------------------------------------------------===//

ModelSpec::~ModelSpec() = default;
