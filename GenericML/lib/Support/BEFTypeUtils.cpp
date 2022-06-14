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
using namespace M;

Optional<Error>
M::decodeQuantizedSignature(BEFFunctionType signature,
                            SmallVectorImpl<TensorSpec> &inputSpecs,
                            SmallVectorImpl<TensorSpec> &outputSpecs) {
  // TODO: we should extract quantization information and store it for the
  // model signature

  SmallVector<BEFType> inputTypes, outputTypes;
  signature.decode(inputTypes, outputTypes);

  auto fillArgSpecs = [](ArrayRef<BEFType> argTypes,
                         auto &argSpecs) -> Optional<Error> {
    for (auto type : argTypes) {
      auto tSpecOr = TensorSpec::get(type);
      if (tSpecOr.isError())
        return tSpecOr.takeError();
      argSpecs.push_back(tSpecOr.get());
    }
    return None;
  };

  if (auto err = fillArgSpecs(inputTypes, inputSpecs))
    return std::move(err.getValue());
  return fillArgSpecs(outputTypes, outputSpecs);
}
