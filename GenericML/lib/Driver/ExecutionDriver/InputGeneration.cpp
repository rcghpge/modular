//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/Driver/DriverCLOptions.h"
#include "GenericML/Driver/DriverInternals.h"
#include "GenericML/Driver/ExecutionDriver.h"
#include "GenericML/Support/BufferRef.h"
#include "GenericML/Support/Model.h"
#include "GenericML/Support/QuantSpec.h"
#include "GenericML/Support/Tensor.h"
#include "Kernels/Misc.h"
#include "LLCL/ForwardDecls.h"
#include "Support/AlignedAlloc.h"
#include "Support/Error.h"
#include "Support/ErrorOr.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/raw_ostream.h"

#include <filesystem>
#include <memory>
#include <numeric>
#include <variant>

using namespace M;

static ErrorOrSuccess fillWithElements(GML::Tensor &tensor,
                                       InputGenerationKind inputGenKind,
                                       const QuantSpec &quantSpec) {
  uint8_t elementBuffer[DType::kMaxElementSizeInBytes];
  auto eltType = tensor.getEltType();

  switch (inputGenKind) {
  case InputGenerationKind::kRandom:
    return Kernels::fillRandom(tensor.getMutableBuffer(),
                               tensor.getShape().getNumElements(), eltType);

  case InputGenerationKind::kOnes: {
    ErrorOrSuccess result = Kernels::getScalarOne(elementBuffer, eltType);
    if (failed(result))
      return result;
    break;
  }
  case InputGenerationKind::kNegOnes: {
    ErrorOrSuccess result =
        Kernels::getScalarNegativeOne(elementBuffer, eltType);
    if (failed(result))
      return result;
    break;
  }
  case InputGenerationKind::kZeros:
    if (quantSpec.isQuantized()) {
      // FIXME: we need to handle all types, and refactor this into a kernel
      if (tensor.getEltType() != DType::ui8 &&
          tensor.getEltType() != DType::si8)
        return Error("when filling quantized tensors with zeros, only 8 bit "
                     "storage types are supported");
      auto params = quantSpec.getParams();
      if (quantSpec.getKind() == QuantSpec::Kind::uniformPerAxis &&
          params.size() != 1)
        return Error("when filling quantized tensors with zeros, only a single "
                     "scale/zeroPoint is supported");
      memset(elementBuffer, (uint8_t)params[0].zeroPoint,
             sizeof(elementBuffer));
    } else
      memset(elementBuffer, 0, sizeof(elementBuffer));
    break;
  }

  // Fill the tensor with the 0, 1, -1 value.
  return Kernels::fillHomogeneous(tensor.getMutableBuffer(),
                                  tensor.getShape().getNumElements(), eltType,
                                  elementBuffer);
}

static ErrorOr<GML::Tensor> allocateInput(const QuantTensorSpec &modelSpec,
                                          Runtime &runtime, size_t alignment) {
  return GML::Tensor::createOwned({modelSpec, modelSpec.getEltType()}, runtime,
                                  alignment);
}

static ErrorOrSuccess fillInput(InputGenerationKind inputGenKind,
                                const QuantTensorSpec &modelSpec,
                                GML::Tensor &tensor) {
  if (auto err =
          fillWithElements(tensor, inputGenKind, modelSpec.getQuantSpec()))
    return err.takeError();
  return success();
}

static ErrorOr<GML::Tensor>
allocateAndFillInput(InputGenerationKind inputGenKind,
                     const QuantTensorSpec &modelSpec, Runtime &runtime,
                     size_t alignment) {
  auto resultOr = allocateInput(modelSpec, runtime, alignment);
  if (resultOr.isError())
    return resultOr.takeError();
  auto result = std::move(*resultOr);
  if (auto err = fillInput(inputGenKind, modelSpec, result))
    return err.takeError();
  return result;
}

// Tensor that has been allocated and filled with an InputGenerationKind.
class FilledTensorGenerator : public ITensorGenerator {
public:
  static ErrorOr<FilledTensorGenerator> create(InputGenerationKind genKind,
                                               const QuantTensorSpec &modelSpec,
                                               Runtime &runtime,
                                               size_t alignment) {
    auto tensorOr =
        allocateAndFillInput(genKind, modelSpec, runtime, alignment);
    if (tensorOr.isError())
      return tensorOr.takeError();
    return FilledTensorGenerator(std::move(*tensorOr));
  }

  const GML::Tensor &getNextTensor() final { return tensor; }

  const GML::Tensor &getTensor(size_t sampleIdx) const final { return tensor; }

  size_t getSampleCount() const final { return 1; }

  bool isCycled() const final { return true; }

private:
  FilledTensorGenerator(GML::Tensor &&tensor) : tensor(std::move(tensor)) {}
  GML::Tensor tensor;
};

// Represents Tensor backed by a file.
// File is loaded into memory during construction and then calls to
// getNextTensor() will stride through the memory.
class FileTensorGenerator : public ITensorGenerator {
public:
  static ErrorOr<FileTensorGenerator>
  create(const std::filesystem::path &filename, const QuantTensorSpec &spec,
         Runtime &runtime, size_t alignment, TensorGeneratorFlag flags) {
    std::vector<GML::Tensor> tensors;
    std::string errorMsg;
    auto data = mlir::openInputFile(filename.string(), llvm::Align(alignment),
                                    &errorMsg);
    if (!data) {
      return Error("Could not open input file.");
    }
    auto stride = spec.getEltType().getSizeInBytes(spec.getNumElements());
    if (!isSet(flags, TensorGeneratorFlag::kTruncate) &&
        data->getBufferSize() % stride != 0) {
      return Error("--input-data file size not a multiple of tensor size. "
                   "Add Truncate attribute to --input-data flag to "
                   "allow this behaviour.");
    }
    auto dataStart = data->getBufferStart();
    for (size_t i = 0, n = data->getBufferSize() / stride; i < n; i++) {
      auto tensor = GML::Tensor::createOwned({spec, spec.getEltType()}, runtime,
                                             alignment);
      auto tensorBuf = static_cast<char *>(tensor.getMutableBuffer());
      std::memcpy(tensorBuf, dataStart + i * stride, tensor.getSizeInBytes());
      tensors.push_back(std::move(tensor));
    }

    if (tensors.empty())
      return Error("input file contains 0 tensors");

    return FileTensorGenerator(std::move(tensors), flags);
  }
  const GML::Tensor &getNextTensor() final {
    assert((isSet(flags, TensorGeneratorFlag::kCycle) || !repeated) &&
           "All tensors in FileTensor have already been generated. "
           "Add Cycle attribute to --input-data to loop over "
           "available tensors.");
    if (pos + 1 >= tensors.size())
      repeated = true;
    auto curr = pos;
    pos = (pos + 1) % tensors.size();
    return tensors[curr];
  }

  const GML::Tensor &getTensor(size_t sampleIdx) const final {
    if (!isCycled())
      assert(sampleIdx < tensors.size() && "Requested sampleIdx out of range");
    return tensors[sampleIdx % tensors.size()];
  }

  size_t getSampleCount() const final { return tensors.size(); }

  bool isCycled() const final {
    return isSet(flags, TensorGeneratorFlag::kCycle);
  }

private:
  FileTensorGenerator(std::vector<GML::Tensor> &&tensors,
                      TensorGeneratorFlag flags)
      : tensors(std::move(tensors)), flags(flags) {}

  size_t pos{0};
  std::vector<GML::Tensor>
      tensors; // each tensor holds a reference to a position in
               // data so this shouldn't be too big
  TensorGeneratorFlag flags;
  bool repeated{false};
};

ErrorOr<ITensorGeneratorPtr>
M::createTensorGenerator(size_t tensorIdx, InputGenerationKind defaultGen,
                         ArrayRef<InputDataSpec> inputDataSpecs,
                         const QuantTensorSpec &modelSpec,
                         size_t minimumCpuAlignment, Runtime &runtime) {
  std::optional<InputDataSpec> spec;
  for (const auto &inputDataSpec : inputDataSpecs) {
    if (inputDataSpec.tensorIdx == tensorIdx)
      spec = inputDataSpec;
  }

  // if tensor spec contains a generation function, override the default
  if (spec && std::holds_alternative<InputGenerationKind>(spec->inputGen))
    defaultGen = std::get<InputGenerationKind>(spec->inputGen);

  if (!spec || std::holds_alternative<InputGenerationKind>(spec->inputGen)) {
    auto filledGeneratorOr = FilledTensorGenerator::create(
        defaultGen, modelSpec, runtime, minimumCpuAlignment);
    if (filledGeneratorOr.isError())
      return filledGeneratorOr.takeError();
    return std::make_unique<FilledTensorGenerator>(
        std::move(*filledGeneratorOr));
  }

  auto filename = std::get<std::filesystem::path>(spec->inputGen);
  // pack vector of flags into single int
  auto fileGeneratorOr = FileTensorGenerator::create(
      filename, modelSpec, runtime, minimumCpuAlignment, spec->genFlags);
  if (fileGeneratorOr.isError())
    return fileGeneratorOr.takeError();
  return std::make_unique<FileTensorGenerator>(std::move(*fileGeneratorOr));
}

ErrorOr<SmallVector<ITensorGeneratorPtr, 3>>
M::createTensorGenerators(InputGenerationKind inputGenKind,
                          ArrayRef<QuantTensorSpec> inputShapeSpecs,
                          ArrayRef<InputDataSpec> inputDataSpecs,
                          size_t minimumCpuAlignment, Runtime &runtime) {
  auto numInputs = inputShapeSpecs.size();

  for (const auto &inputData : inputDataSpecs) {
    if (inputData.tensorIdx >= numInputs) {
      return Error(
          Twine("Tensor index (") + Twine(inputData.tensorIdx) +
          ") passed to --input-data larger than number of inputs in model (" +
          Twine(numInputs) + ").");
    }
  }

  SmallVector<ITensorGeneratorPtr, 3> inputTensorGenerators;
  for (size_t i = 0; i < numInputs; i++) {
    const auto &modelSpec = inputShapeSpecs[i];
    auto generatorOrError =
        createTensorGenerator(i, inputGenKind, inputDataSpecs, modelSpec,
                              minimumCpuAlignment, runtime);
    if (generatorOrError) {
      return Error("Input " + llvm::Twine(i) + ": " +
                   generatorOrError.getError());
    }
    inputTensorGenerators.push_back(std::move(*generatorOrError));
  }
  return inputTensorGenerators;
}
