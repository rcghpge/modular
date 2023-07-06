//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares utilities to work with MEFType and its subclasses in the
// context of GenericML.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_SUPPORT_MEFTYPEUTILS_H
#define GENERICML_SUPPORT_MEFTYPEUTILS_H

#include "GraphRT/MEFExecutor/PrimitiveRegistry.h"
#include "Support/LLVMForwardDecls.h"

namespace M {

class MEFFunctionType;
class ErrorOrSuccess;
class QuantTensorSpec;

/// Decodes a MEFFunctionType signature into input and output QuantTensorSpecs.
ErrorOrSuccess
decodeQuantizedSignature(MEFFunctionType signature,
                         SmallVectorImpl<QuantTensorSpec> &inputSpecs,
                         SmallVectorImpl<QuantTensorSpec> &outputSpecs);

/// Decodes argument and result names from `MEFListAttr`s in an
/// `MEFDictionaryAttr`.
/// For example, decodes {argument_names = ["a", "b"], result_names = ["c"]}
/// into `inputNames` and `outputNames`, respectively.
ErrorOrSuccess decodeTensorNames(MEFDictionaryAttr attrDictionary,
                                 SmallVectorImpl<StringRef> &inputNames,
                                 SmallVectorImpl<StringRef> &outputNames);

} // namespace M

#endif // GENERICML_SUPPORT_MEFTYPEUTILS_H
