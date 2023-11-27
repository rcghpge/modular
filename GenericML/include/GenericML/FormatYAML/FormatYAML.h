//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file provides YAMLIO serialization and deserialization interfaces
// for types found in GenericML such as TensorSpec and QuantSpec.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_FORMATYAML_H
#define GENERICML_FORMATYAML_H

#include "Support/ErrorOr.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/LLVMYAMLForwardDecls.h"
#include <string>

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
namespace llvm::yaml {
template <>
struct MappingTraits<M::ModelSpec> {
  static void mapping(IO &io, M::ModelSpec &modelSpec);
  static std::string validate(IO &io, M::ModelSpec &modelSpec);
};
} // namespace llvm::yaml

/// Provide the QuantTensorSpec mapping trait for YAML conversion.
LLVM_FWD_YAML_DECLARE_MAPPING_TRAITS(M::QuantTensorSpec);

/// Use LLVM_YAML_IS_SEQUENCE_VECTOR to specify SequenceTraits<> on the
/// SmallVector<QuantTensorSpec> type.
LLVM_FWD_YAML_IS_SEQUENCE_VECTOR(M::QuantTensorSpec);

#endif // GENERICML_FORMATYAML_H
