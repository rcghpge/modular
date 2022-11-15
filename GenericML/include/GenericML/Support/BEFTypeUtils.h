//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares utilities to work with BEFType and its subclasses in the
// context of GenericML.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_SUPPORT_BEFTYPEUTILS_H
#define GENERICML_SUPPORT_BEFTYPEUTILS_H

#include "Support/LLVMForwardDecls.h"

namespace M {

class BEFFunctionType;
class ErrorOrSuccess;
class QuantTensorSpec;

/// Decodes a BEFFunctionType signature into input and output QuantTensorSpecs.
ErrorOrSuccess
decodeQuantizedSignature(BEFFunctionType signature,
                         SmallVectorImpl<QuantTensorSpec> &inputSpecs,
                         SmallVectorImpl<QuantTensorSpec> &outputSpecs);

} // namespace M

#endif // GENERICML_SUPPORT_BEFTYPEUTILS_H
