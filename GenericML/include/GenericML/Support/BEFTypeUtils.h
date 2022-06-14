//===- GenericML/Support/BEFTypeUtils.h -----------------------------------===//
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

#include "GenericML/Support/QuantSpec.h"
#include "GraphRT/BEF/BEFTypes.h"

namespace M {

/// Decodes a BEFFunctionType signature into input and output QuantTensorSpecs.
Optional<Error>
decodeQuantizedSignature(BEFFunctionType signature,
                         SmallVectorImpl<QuantTensorSpec> &inputSpecs,
                         SmallVectorImpl<QuantTensorSpec> &outputSpecs);

} // namespace M

#endif // GENERICML_SUPPORT_BEFTYPEUTILS_H
