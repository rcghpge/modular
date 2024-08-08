//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares the ResultInfo and related classes, which hold
// model result information for outputs in a framework-independent way.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_SUPPORT_RESULTINFO_H
#define GENERICML_SUPPORT_RESULTINFO_H

#include "ArraySupport/Tensor.h"
#include "GenericML/Support/Value.h"
#include "Support/ML/TensorSpec.h"
#include <map>

namespace M {

/// ResultInfo base class and corresponding MappingTraits to help write output
/// information to YAML when --result-output-style=binary.
class ResultInfo {
public:
  ResultInfo() = default;
  ResultInfo(std::string name);
  virtual ~ResultInfo() = default;

  std::string name;

  virtual void map(llvm::yaml::IO &io) = 0;
};

/// TensorResultInfo class and corresponding MappingTraits to help write output
/// information to YAML when --result-output-style=binary.
class TensorResultInfo : public ResultInfo {
public:
  TensorResultInfo() = default;
  TensorResultInfo(std::string name, std::string shape);
  TensorResultInfo(const TensorResultInfo &other) = default;
  TensorResultInfo(std::string name, const Tensor &tensor);

  TensorResultInfo operator=(TensorResultInfo other);
  TensorResultInfo &operator=(TensorResultInfo &other);

  void map(llvm::yaml::IO &io) override;

  std::string shape;
};

/// VariantResultInfo class and corresponding MappingTraits to help write output
/// information to YAML when --result-output-style=binary.
class VariantResultInfo : public M::ResultInfo {
public:
  VariantResultInfo(std::string name, M::GML::Variant variant);

  VariantResultInfo(const VariantResultInfo &) = default;
  VariantResultInfo(VariantResultInfo &&) = default;
  VariantResultInfo &operator=(const VariantResultInfo &) = default;
  VariantResultInfo &operator=(VariantResultInfo &&) = default;

  void map(llvm::yaml::IO &io) override;

  M::GML::Variant variant;
};

/// ProcessedResults holds metainformation about each model output in
/// `ResultInfo` objects. Tensors contained as leaf nodes in model outputs are
/// collected as a flat map. Each tensor either has the same name as the model
/// output (if the output is a tensor) or is assigned a name (if the tensor is
/// contained as a leaf node in a composed type).
/// `ProcessedResults` is populated as part of an "unzipping" process, where
/// the initial model result is a nested structure (Dict, List, Tuple) with
/// tensors as leaf nodes. A flat representation of the (unique) tensors is
/// needed for export, and tensors contained in the structure are replace with
/// their metainformation (name, shape).
class ProcessedResults {
public:
  ProcessedResults();

  /// Inserts a new result information object. Currently this can be a
  /// `TensorResultInfo` if the output is a tensor, or `VariantResultInfo` if
  /// the output is a composed type.
  void insertResultInfo(std::shared_ptr<ResultInfo> info);

  /// Inserts a new named tensor into the tensor map.
  void insertTensor(const std::string &name, Tensor *tensor);

  /// Inserts a new unnamed tensor, assigns it a name and returns the name. If a
  /// tensor with the same underlying buffer already exists, its name is
  /// returned instead.
  std::string insertTensor(Tensor *tensor);

  /// Returns true if a tensor with the same underlying buffer
  /// exists in the results, and false otherwise.
  bool containsTensor(Tensor *tensor) const;

  /// Returns the tensor's name if a tensor with the same underlying buffer
  /// exists in the results, and an empty string otherwise. If a tensor is
  /// registered with multiple names, the first matching map entry is returned.
  std::string getTensorName(Tensor *) const;

  /// Returns the `ResultInfo`s.
  SmallVector<std::shared_ptr<ResultInfo>> &getResultInfos();

  /// Returns the collected named tensors.
  const std::map<std::string, Tensor *> &getTensors() const;

private:
  // Holds the ResultInfo objects for each output.
  SmallVector<std::shared_ptr<ResultInfo>> resultInfos;
  // Counter to count unnamed binary outputs.
  size_t counter;
  // Holds the collected tensors.
  std::map<std::string, Tensor *> tensors;
};

} // namespace M

#endif // GENERICML_SUPPORT_RESULTINFO_H
