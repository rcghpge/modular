//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_API_SOURCEMODEL_H
#define GENERICML_API_SOURCEMODEL_H

#include "GraphCompiler/Driver/Common/DriverInputFormat.h"
#include <filesystem>
#include <type_traits>

namespace M {

/// Opaque representation for compiling a model.
struct ModelSource {
  void *source;
  M::FileFormat fileFormat;
};
static_assert(std::is_standard_layout_v<ModelSource> &&
                  std::is_trivial_v<ModelSource>,
              "ModelSource must be C ABI compatible");

/// Represents a model that has not yet been compiled, such as a TensorFlow
/// saved model.
struct SourceModel {
  SourceModel(ModelSource source)
      : modelPath(std::nullopt), modelSource{source} {}

  SourceModel(std::filesystem::path path, FileFormat format)
      : modelPath(std::move(path)), modelSource(ModelSource{nullptr, format}) {}

  FileFormat format() const { return modelSource.fileFormat; }

  std::optional<std::filesystem::path> modelPath;
  // Always stores the FileFormat.
  // Compilation prefers to compile from `ModelSource.source` when non-null.
  ModelSource modelSource;
};

} // namespace M

#endif // GENERICML_API_SOURCEMODEL_H
