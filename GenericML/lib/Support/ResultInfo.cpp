//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/Support/ResultInfo.h"
#include <llvm/Support/YAMLTraits.h>

using namespace M;

ResultInfo::ResultInfo(std::string name) : name(std::move(name)) {}

TensorResultInfo::TensorResultInfo(std::string name, std::string shape)
    : ResultInfo(std::move(name)), shape(std::move(shape)){};

TensorResultInfo TensorResultInfo::operator=(TensorResultInfo other) {
  name = other.name;
  shape = other.shape;
  return *this;
}

TensorResultInfo::TensorResultInfo(std::string name, const GML::Tensor &tensor)
    : ResultInfo(name) {
  shape = tensor.getSpec().getAsString();
}

TensorResultInfo &TensorResultInfo::operator=(TensorResultInfo &other) {
  name = other.name;
  shape = other.shape;
  return *this;
}

void TensorResultInfo::map(llvm::yaml::IO &io) {
  io.mapRequired("name", name);
  io.mapRequired("shape", shape);
}
