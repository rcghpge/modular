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

static ErrorOrSuccess fillWithElements(Tensor &tensor,
                                       InputGenerationKind inputGenKind,
                                       const QuantSpec &quantSpec) {
  uint8_t elementBuffer[DType::kMaxElementSizeInBytes];
  auto eltType = tensor.getEltType();

  switch (inputGenKind) {
  case InputGenerationKind::kRandom:
    return Kernel::fillRandom(tensor.getMutableBuffer(),
                              tensor.getShape().getNumElements(), eltType);

  case InputGenerationKind::kOnes: {
    ErrorOrSuccess result = Kernel::getScalarOne(elementBuffer, eltType);
    if (failed(result))
      return result;
    break;
  }
  case InputGenerationKind::kNegOnes: {
    ErrorOrSuccess result =
        Kernel::getScalarNegativeOne(elementBuffer, eltType);
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
  return Kernel::fillHomogeneous(tensor.getMutableBuffer(),
                                 tensor.getShape().getNumElements(), eltType,
                                 elementBuffer);
}

static ErrorOr<Tensor> allocateInput(const TensorShape &userShape,
                                     const QuantTensorSpec &modelSpec,
                                     Runtime &runtime, size_t alignment) {
  if (userShape.getRank() != modelSpec.getRank())
    return Error(Twine("Specified shape ") + userShape.getAsString() +
                 " doesn't match the rank of model input shape " +
                 modelSpec.getAsString());
  for (const auto &[userDim, modelDim] :
       llvm::zip(userShape.getDims(), modelSpec.getDims())) {
    if (mlir::ShapedType::isDynamic(modelDim)) {
      if (mlir::ShapedType::isDynamic(userDim))
        return Error(
            Twine("Must specify a fixed input shape for model input shape ") +
            modelSpec.getAsString());
    } else if (userDim != modelDim) {
      return Error(Twine("Specified dimension ") + Twine(userDim) +
                   " doesn't match model input dimension " + Twine(modelDim) +
                   " for shape " + modelSpec.getAsString());
    }
  }

  return Tensor::createOwned({userShape, modelSpec.getEltType()}, runtime,
                             alignment);
}

static ErrorOrSuccess fillInput(InputGenerationKind inputGenKind,
                                const QuantTensorSpec &modelSpec,
                                Tensor &tensor) {
  RETURN_ERROR(
      fillWithElements(tensor, inputGenKind, modelSpec.getQuantSpec()));
  return success();
}

static ErrorOr<Tensor> allocateAndFillInput(InputGenerationKind inputGenKind,
                                            const TensorShape &userShape,
                                            const QuantTensorSpec &modelSpec,
                                            Runtime &runtime,
                                            size_t alignment) {
  UNWRAP_ERROR(result, allocateInput(userShape, modelSpec, runtime, alignment));
  RETURN_ERROR(fillInput(inputGenKind, modelSpec, result));
  return result;
}

// Tensor that has been allocated and filled with an InputGenerationKind.
class FilledTensorGenerator : public ITensorGenerator {
public:
  FilledTensorGenerator(InputGenerationKind genKind,
                        const TensorShape &userShape,
                        const QuantTensorSpec &modelSpec, Runtime &runtime,
                        size_t alignment) {
    auto tensorOr =
        allocateAndFillInput(genKind, userShape, modelSpec, runtime, alignment);
    assert(!tensorOr.isError() && "could not initialize FilledTensorGenerator");
    tensor = std::make_unique<Tensor>(std::move(*tensorOr));
  }

  const Tensor &getNextTensor() final { return *tensor; }

  const Tensor &getTensor(size_t sampleIdx) const final { return *tensor; }

  size_t getSampleCount() const final { return 1; }

private:
  // needs to be ptr since there is no default constructor
  std::unique_ptr<Tensor> tensor;
};

// Represents Tensor backed by a file.
// File is loaded into memory during construction and then calls to
// getNextTensor() will stride through the memory.
class FileTensorGenerator : public ITensorGenerator {
public:
  FileTensorGenerator(const std::filesystem::path &filename,
                      const TensorShape &userShape, const QuantTensorSpec &spec,
                      Runtime &runtime, size_t alignment,
                      TensorGeneratorFlag flags)
      : flags(flags) {
    std::string errorMsg;
    auto data = mlir::openInputFile(filename.string(), llvm::Align(alignment),
                                    &errorMsg);
    if (!data) {
      llvm::errs() << errorMsg << "\n";
      assert(false && "Could not open input file.");
    }
    auto stride = spec.getEltType().getSizeInBytes(userShape.getNumElements());
    if (!isSet(flags, TensorGeneratorFlag::kTruncate) &&
        data->getBufferSize() % stride != 0) {
      llvm::errs() << "--input-data file size not a multiple of tensor size. "
                      "Add Truncate attribute to --input-data flag to "
                      "allow this behaviour.\n";
      assert(false);
    }
    auto dataStart = data->getBufferStart();
    for (size_t i = 0, n = data->getBufferSize() / stride; i < n; i++) {
      auto tensor = Tensor::createOwned({userShape, spec.getEltType()}, runtime,
                                        alignment);
      auto tensorBuf = static_cast<char *>(tensor.getMutableBuffer());
      std::memcpy(tensorBuf, dataStart + i * stride, tensor.getSizeInBytes());
      tensors.push_back(std::move(tensor));
    }
    assert(!tensors.empty() && "input file contains 0 tensors");
  }

  const Tensor &getNextTensor() final {
    if (!isSet(flags, TensorGeneratorFlag::kCycle) && repeated) {
      llvm::errs() << "All tensors in FileTensor have already been generated. "
                      "Add Cycle attribute to --input-data to loop over "
                      "available tensors.\n";
      assert(false);
    }
    if (pos + 1 >= tensors.size())
      repeated = true;
    auto curr = pos;
    pos = (pos + 1) % tensors.size();
    return tensors[curr];
  }

  const Tensor &getTensor(size_t sampleIdx) const final {
    assert(sampleIdx < tensors.size() && "Requested sampleIdx out of range");
    return tensors[sampleIdx];
  }

  size_t getSampleCount() const final { return tensors.size(); }

private:
  size_t pos{0};
  std::vector<Tensor> tensors; // each tensor holds a reference to a position in
                               // data so this shouldn't be too big
  TensorGeneratorFlag flags{TensorGeneratorFlag::kComplete};
  bool repeated{false};
};

ErrorOr<ITensorGeneratorPtr>
M::createTensorGenerator(size_t tensorIdx, InputGenerationKind defaultGen,
                         ArrayRef<InputDataSpec> inputDataSpecs,
                         const TensorShape &userShape,
                         const QuantTensorSpec &modelSpec, Runtime &runtime) {
  std::optional<InputDataSpec> spec;
  for (const auto &inputDataSpec : inputDataSpecs) {
    if (inputDataSpec.tensorIdx == tensorIdx)
      spec = inputDataSpec;
  }

  // if tensor spec contains a generation function, override the default
  if (spec && std::holds_alternative<InputGenerationKind>(spec->inputGen))
    defaultGen = std::get<InputGenerationKind>(spec->inputGen);

  auto alignment = kPreferredMemoryAlignment;
  if (!spec || std::holds_alternative<InputGenerationKind>(spec->inputGen))
    return std::make_unique<FilledTensorGenerator>(
        defaultGen, userShape, modelSpec, runtime, alignment);

  auto filename = std::get<std::filesystem::path>(spec->inputGen);
  // pack vector of flags into single int
  auto flagsPacked = std::accumulate(
      spec->genFlags.begin(), spec->genFlags.end(),
      TensorGeneratorFlag::kComplete, [](auto acc, auto x) { return acc | x; });
  return std::make_unique<FileTensorGenerator>(filename, userShape, modelSpec,
                                               runtime, alignment, flagsPacked);
}

ErrorOr<SmallVector<ITensorGeneratorPtr, 3>>
M::createTensorGenerators(InputGenerationKind inputGenKind,
                          ArrayRef<TensorShape> userInputShapes,
                          ArrayRef<InputDataSpec> inputDataSpecs,
                          const Model *model, Runtime &runtime) {
  SmallVector<TensorShape> inputShapes(userInputShapes.begin(),
                                       userInputShapes.end());
  auto numInputs = model->getInputSpecs().size();
  if (!inputShapes.empty()) {
    if (inputShapes.size() != numInputs) {
      return Error(Twine("Number of specified input shapes (") +
                   Twine(inputShapes.size()) +
                   ") doesn't match number of model inputs (" +
                   Twine(numInputs) + ")");
    }
  } else {
    auto modelSpecs = model->getInputSpecs();
    inputShapes.append(modelSpecs.begin(), modelSpecs.end());
  }

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
    const auto &userShape = inputShapes[i];
    const auto &modelSpec = model->getInputSpecs()[i];
    UNWRAP_ERROR(result, createTensorGenerator(i, inputGenKind, inputDataSpecs,
                                               userShape, modelSpec, runtime));
    inputTensorGenerators.push_back(std::move(result));
  }
  return inputTensorGenerators;
}
