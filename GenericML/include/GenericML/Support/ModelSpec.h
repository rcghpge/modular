//===- GenericML/Support/ModelSpec.h --------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements an in-memory representation for a model specification.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_SUPPORT_MODELSPEC_H
#define GENERICML_SUPPORT_MODELSPEC_H

#include "GenericML/Support/QuantSpec.h"
#include "Support/LLVMForwardDecls.h"
#include "llvm/ADT/StringRef.h"

namespace M {

//===----------------------------------------------------------------------===//
// ModelSpec
//===----------------------------------------------------------------------===//

/// This class is a framework-independent abstraction for a single ML model's
/// information. It is intended to contain the name, input and output tensor
/// specs, and the input and output layer names.
class ModelSpec {
public:
  using ModelIOSpec = std::pair<std::string, M::QuantTensorSpec>;
  struct ModelBinary {
    std::string format;
    std::string url;
    std::string archivedFile;
  };
  ModelSpec() = default;
  ModelSpec(const ModelSpec &) = default;
  ModelSpec(ModelSpec &&) = default;
  ModelSpec &operator=(ModelSpec &&) = default;

  ~ModelSpec();
  StringRef getName() const { return name; };

  std::string name;
  SmallVector<ModelBinary, 3> model;
  SmallVector<ModelIOSpec, 5> inputs, outputs;
};
} // namespace M

#endif // GENERICML_SUPPORT_MODELSPEC_H
