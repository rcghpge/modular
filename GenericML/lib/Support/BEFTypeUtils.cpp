//===- BEFTypeUtils.cpp ---------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements utilities to work with BEFType and its subclasses in the
// context of GenericML.
//
//===----------------------------------------------------------------------===//

#include "GenericML/Support/BEFTypeUtils.h"
#include "GenericML/Support/QuantSpec.h"
#include "GraphRT/BEF/BEFTypes.h"
#include "Support/ErrorOr.h"
using namespace M;

Optional<Error>
M::decodeQuantizedSignature(BEFFunctionType signature,
                            SmallVectorImpl<QuantTensorSpec> &inputSpecs,
                            SmallVectorImpl<QuantTensorSpec> &outputSpecs) {
  SmallVector<BEFType> inputTypes, outputTypes;
  signature.decode(inputTypes, outputTypes);

  auto fillArgSpecs = [](ArrayRef<BEFType> argTypes,
                         auto &argSpecs) -> Optional<Error> {
    for (auto type : argTypes) {
      auto specOr = QuantTensorSpec::get(type);
      if (specOr.isError())
        return specOr.takeError();
      argSpecs.push_back(specOr.get());
    }
    return None;
  };

  if (auto err = fillArgSpecs(inputTypes, inputSpecs))
    return std::move(err.getValue());
  return fillArgSpecs(outputTypes, outputSpecs);
}
