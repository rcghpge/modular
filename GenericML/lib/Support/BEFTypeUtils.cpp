//===----------------------------------------------------------------------===//
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

ErrorOrSuccess
M::decodeQuantizedSignature(BEFFunctionType signature,
                            SmallVectorImpl<QuantTensorSpec> &inputSpecs,
                            SmallVectorImpl<QuantTensorSpec> &outputSpecs) {
  SmallVector<BEFType> inputTypes, outputTypes;
  signature.decode(inputTypes, outputTypes);

  auto fillArgSpecs = [](ArrayRef<BEFType> argTypes,
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
