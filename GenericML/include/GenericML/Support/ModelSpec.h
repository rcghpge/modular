//===----------------------------------------------------------------------===//
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
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"

namespace M {

//===----------------------------------------------------------------------===//
// ModelSpec
//===----------------------------------------------------------------------===//

/// This class is a framework-independent abstraction for a single ML model's
/// information. It is intended to contain the name, input and output tensor
/// specs, and the input and output layer names.
///
/// Keep in sync with the description in Models/docs/index.md, the YAML
/// conversion in GenericML/lib/FormatYAML/FormatYAML.cpp, and the Python
/// reader in GenericML/tools/model/model/utils/manifest.py.
class ModelSpec {
public:
  /// Per-tensor specification
  struct ModelIOSpec {
    std::string name;
    M::QuantTensorSpec spec;

    /// Data-source indicating how this tensor should be initialized.
    /// This can take any of the following forms:
    ///
    ///   - keyword: zeros|ones|negones|random
    ///   - filename: absolute path or relative to manifest location
    ///
    std::optional<std::string> data;

    std::optional<std::string> size;
    std::optional<std::string> sha1sum;

    /// TODO: This is a temporary flag that is used as we bring up mo shape
    /// inference.
    std::optional<bool> legacyShapeInference;

    ModelIOSpec() = default;
    ModelIOSpec(std::string name, M::QuantTensorSpec spec,
                std::optional<std::string> data = {},
                std::optional<std::string> size = {},
                std::optional<std::string> sha1sum = {})
        : name(std::move(name)), spec(std::move(spec)), data(std::move(data)),
          size(std::move(size)), sha1sum(std::move(sha1sum)) {}
  };

  /// Maps input or output names as they appear in the model spec (the
  /// 'preferred' names) to the corresponding names as they appear in the
  /// underlying model format (the 'native' names). Names not in this map are
  /// assumed to match.
  ///
  /// Unfortunately there are trivial differences in names between 'equivalent'
  /// models in different formats (eg tensorflow vs ONNX runtime).
  ///
  /// Only significant for formats which use named inputs or outputs.
  struct NameSubstitutions {
    llvm::StringMap<std::string> preferredToNativeMap;

    /// Returns the 'native' preferredName corresponding to the given
    /// 'preferred' model preferredName.
    ///
    /// CAUTION: The given preferredName must be from the model spec. The
    /// result's lifetime is then that of the containing model spec, and its
    /// data() will be null-terminated and thus directly usable as a C string.
    StringRef getPreferredName(StringRef preferredName) const;
  };

  /// Describes how to fetch the model in its native binary format.
  struct ModelBinary {
    std::string format;
    std::string url;
    std::string archivedFile;
    std::string size;
    std::string sha1sum;
    std::string metadata;
    NameSubstitutions inputSubstitutions;
    NameSubstitutions outputSubstitutions;
    /// TODO: This is a temporary flag that is used as we bring up mo shape
    /// inference.
    std::optional<bool> legacyShapeInference;
  };

  ModelSpec() = default;
  ModelSpec(const ModelSpec &) = default;
  ModelSpec(ModelSpec &&) = default;
  ModelSpec &operator=(ModelSpec &&) = default;

  ~ModelSpec();

  /// Returns the model binary corresponding to the given format.
  ErrorOr<const ModelBinary *> getModelBinary(StringRef format) const;

  std::string name;
  std::string task;
  std::string description;
  SmallVector<ModelBinary, 3> model;
  SmallVector<ModelIOSpec, 5> inputs, outputs;
};
} // namespace M

#endif // GENERICML_SUPPORT_MODELSPEC_H
