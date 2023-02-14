//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements utilities to work with MEFType and its subclasses in the
// context of GenericML.
//
//===----------------------------------------------------------------------===//

#include "GenericML/Support/MEFTypeUtils.h"
#include "GenericML/Support/QuantSpec.h"
#include "GraphRT/MEF/MEFTypes.h"
#include "Support/ErrorOr.h"

using namespace M;

ErrorOrSuccess
M::decodeQuantizedSignature(MEFFunctionType signature,
                            SmallVectorImpl<QuantTensorSpec> &inputSpecs,
                            SmallVectorImpl<QuantTensorSpec> &outputSpecs) {
  SmallVector<MEFType> inputTypes, outputTypes;
  signature.decode(inputTypes, outputTypes);

  auto fillArgSpecs = [](ArrayRef<MEFType> argTypes,
                         auto &argSpecs) -> ErrorOrSuccess {
    for (auto type : argTypes) {
      UNWRAP_ERROR(spec, QuantTensorSpec::get(type));
      argSpecs.push_back(spec);
    }
    return success();
  };

  RETURN_ERROR(fillArgSpecs(inputTypes, inputSpecs));
  return fillArgSpecs(outputTypes, outputSpecs);
}
