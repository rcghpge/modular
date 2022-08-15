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
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
using namespace M;

//===----------------------------------------------------------------------===//
// ModelSpec::NameSubstitutions
//===----------------------------------------------------------------------===//
StringRef ModelSpec::NameSubstitutions::SubstitutePreferredToNativeName(
    StringRef name) const {
  auto itr = preferredToNativeMap.find(name);
  return itr == preferredToNativeMap.end() ? name : itr->second;
}

//===----------------------------------------------------------------------===//
// ModelSpec
//===----------------------------------------------------------------------===//

ModelSpec::~ModelSpec() = default;

ErrorOr<const ModelSpec::ModelBinary *>
ModelSpec::GetModelBinary(StringRef format) const {
  auto itr = std::find_if(model.begin(), model.end(),
                          [format](const ModelBinary &modelBinary) {
                            return modelBinary.format == format;
                          });
  if (itr == model.end())
    return Error(
        Twine("The model spec does not contain an entry for format '") +
        format + "'");
  return &*itr;
}
