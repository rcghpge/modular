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

} // namespace M

#endif // GENERICML_SUPPORT_MEFTYPEUTILS_H
