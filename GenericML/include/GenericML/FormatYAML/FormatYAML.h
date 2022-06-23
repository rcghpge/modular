//===- GenericML/FormatYAML.h ---------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This is a file provides YAMLIO serialization interfaces for types found in
// GenericML such as TensorSpec and QuantSpec.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_FORMATYAML_H
#define GENERICML_FORMATYAML_H

#include "llvm/Support/YAMLTraits.h"

namespace M {
class QuantTensorSpec;
}

/// Provide the QuantTensorSpec mapping trait for YAML conversion.
LLVM_YAML_DECLARE_MAPPING_TRAITS(::M::QuantTensorSpec);

#endif // GENERICML_FORMATYAML_H
