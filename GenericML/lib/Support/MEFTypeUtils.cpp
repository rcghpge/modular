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
#include "Support/ML/CommonAttributes.h"

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
      auto specOr = QuantTensorSpec::get(type);
      if (specOr.isError())
        return specOr.takeError();
      argSpecs.push_back(std::move(*specOr));
    }
    return success();
  };

  if (auto err = fillArgSpecs(inputTypes, inputSpecs))
    return err.takeError();
  return fillArgSpecs(outputTypes, outputSpecs);
}

// Decodes a list of names from a `MEFListAttr` in an `MEFDictionaryAttr`.
// For example, decodes {argument_names = ["arg0", "arg1"]} into
// SmallVector<StringRef>{"arg0", "arg1"}.
static ErrorOrSuccess decodeNamesFromAttrList(SmallVectorImpl<StringRef> &names,
                                              MEFDictionaryAttr attrDictionary,
                                              std::string_view key) {
  auto namesAttrOr = attrDictionary.find(key);
  if (!namesAttrOr)
    return Error(Twine(key) + " not found in MGP ModelOp attr dict");
  MEFDynamicAttr &namesAttr = *namesAttrOr;

  llvm::transform(namesAttr.cast<MEFListAttr>(), std::back_inserter(names),
                  [](MEFDynamicAttr element) {
                    return element.cast<MEFAttr<StringRef>>().get();
                  });

  return success();
}

ErrorOrSuccess M::decodeTensorNames(MEFDictionaryAttr attrDictionary,
                                    SmallVectorImpl<StringRef> &inputNames,
                                    SmallVectorImpl<StringRef> &outputNames) {
  ErrorOrSuccess errOr = decodeNamesFromAttrList(inputNames, attrDictionary,
                                                 kMgpModelArgumentNames);
  if (errOr.isError())
    return errOr.takeError();

  return decodeNamesFromAttrList(outputNames, attrDictionary,
                                 kMgpModelResultNames);
}
