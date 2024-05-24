//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/Support/ResultInfo.h"
#include "GenericML/Support/FormatYAML.h"
#include "llvm/Support/YAMLTraits.h"

using namespace M;

ResultInfo::ResultInfo(std::string name) : name(std::move(name)) {}

TensorResultInfo::TensorResultInfo(std::string name, std::string shape)
    : ResultInfo(std::move(name)), shape(std::move(shape)) {};

TensorResultInfo TensorResultInfo::operator=(TensorResultInfo other) {
  name = other.name;
  shape = other.shape;
  return *this;
}

TensorResultInfo::TensorResultInfo(std::string name, const Tensor &tensor)
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

VariantResultInfo::VariantResultInfo(std::string name, GML::Variant variant)
    : ResultInfo(std::move(name)), variant(variant) {}

void VariantResultInfo::map(llvm::yaml::IO &io) {
  io.mapRequired("name", name);
  io.mapRequired("variant", variant);
}

ProcessedResults::ProcessedResults() : resultInfos(), counter(0), tensors() {};

void ProcessedResults::insertResultInfo(std::shared_ptr<ResultInfo> info) {
  resultInfos.push_back(info);
}

std::string ProcessedResults::insertTensor(Tensor *tensor) {
  // If this tensor exists in the results already, instead of inserting it
  // again, return the existing name.
  if (containsTensor(tensor))
    return getTensorName(tensor);
  std::string name = "__tensor_" + std::to_string(counter) + "__";
  tensors.insert({name, tensor});
  ++counter;
  return name;
}

void ProcessedResults::insertTensor(const std::string &name, Tensor *tensor) {
  tensors.insert({name, tensor});
}

bool ProcessedResults::containsTensor(Tensor *tensor) const {
  for (auto &[existingName, existingTensor] : tensors) {
    if (tensor->getBuffer() == existingTensor->getBuffer() &&
        tensor->getSizeInBytes() == existingTensor->getSizeInBytes())
      return true;
  }
  return false;
}

std::string ProcessedResults::getTensorName(Tensor *tensor) const {
  for (auto &[existingName, existingTensor] : tensors) {
    if (tensor->getBuffer() == existingTensor->getBuffer() &&
        tensor->getSizeInBytes() == existingTensor->getSizeInBytes())
      return existingName;
  }
  return "";
}

SmallVector<std::shared_ptr<ResultInfo>> &ProcessedResults::getResultInfos() {
  return resultInfos;
}

const std::map<std::string, Tensor *> &ProcessedResults::getTensors() const {
  return tensors;
}
