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

#include "GenericML/Support/Tensor.h"
#include "Support/ML/TensorSpec.h"

namespace M {

class ResultInfo {
public:
  ResultInfo() = default;
  ResultInfo(std::string name);
  virtual ~ResultInfo() = default;

  std::string name;

  virtual void map(llvm::yaml::IO &io) = 0;
};

/// TensorResultInfo struct and corresponding MappingTraits to help write output
/// information to YAML when --result-output-style=binary.
class TensorResultInfo : public ResultInfo {
public:
  TensorResultInfo() = default;
  TensorResultInfo(std::string name, std::string shape);
  TensorResultInfo(const TensorResultInfo &other) = default;
  TensorResultInfo(std::string name, const GML::Tensor &tensor);

  TensorResultInfo operator=(TensorResultInfo other);
  TensorResultInfo &operator=(TensorResultInfo &other);

  void map(llvm::yaml::IO &io) override;

  std::string shape;
};

} // namespace M

#endif // GENERICML_SUPPORT_RESULTINFO_H
