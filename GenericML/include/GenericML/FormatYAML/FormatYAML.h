//===- GenericML/FormatYAML.h ---------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This is a file provides YAMLIO serialization and deserialization interfaces
// for types found in GenericML such as TensorSpec and QuantSpec.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_FORMATYAML_H
#define GENERICML_FORMATYAML_H

#include "Support/ErrorOr.h"
#include "Support/LLVMForwardDecls.h"
#include "llvm/Support/YAMLTraits.h"

namespace llvm {
class MemoryBuffer;
};

namespace M {
class ModelSpec;
class QuantTensorSpec;

ErrorOr<ModelSpec>
loadModelSpecFromYAML(std::unique_ptr<llvm::MemoryBuffer> yamlBuffer);
} // namespace M

/// Provide the ModelSpec mapping trait for YAML conversion.
LLVM_YAML_DECLARE_MAPPING_TRAITS(::M::ModelSpec);

/// Provide the QuantTensorSpec mapping trait for YAML conversion.
LLVM_YAML_DECLARE_MAPPING_TRAITS(::M::QuantTensorSpec);

#endif // GENERICML_FORMATYAML_H
