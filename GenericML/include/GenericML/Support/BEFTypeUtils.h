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

#include "GenericML/Support/TensorSpec.h"
#include "GraphRT/BEF/BEFTypes.h"

namespace M {

/// Decodes a signature in BEFFunctionType into input and output TensorSpecs.
Optional<Error>
decodeQuantizedSignature(BEFFunctionType signature,
                         SmallVectorImpl<TensorSpec> &inputSpecs,
                         SmallVectorImpl<TensorSpec> &outputSpecs);

} // namespace M

#endif // GENERICML_SUPPORT_BEFTYPEUTILS_H
