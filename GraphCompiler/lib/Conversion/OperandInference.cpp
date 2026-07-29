//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/Conversion/OperandInference.h"
#include "GraphCompiler/MODialect/MODevices.h"
#include "GraphCompiler/MODialect/MOOps.h"

namespace M::MOGG {

static bool opHasExtraOutputShapeArgument(Operation *op) {
  return isa<MO::Static::ReshapeOp, MO::Static::BroadcastToOp>(op);
}

static StringRef moOperationName(Operation *op) {
  if (auto customOp = dyn_cast<MO::CustomOp>(op))
    return customOp.getSymbol();

  return op->getName().getStringRef();
}

static std::optional<KernelDeclOperandVariant>
MOTypeToKernelDeclOperand(Type type) {
  auto iface = dyn_cast<MO::MOKernelOperandTypeInterface>(type);
  if (!iface)
    return std::nullopt;
  switch (iface.getKernelOperandKind()) {
  case MO::KernelOperandKind::Tensor:
    return KernelDeclTensorAdaptor{MOGG::IOSpec::InputTensor};
  case MO::KernelOperandKind::Buffer:
    return KernelDeclTensorAdaptor{MOGG::IOSpec::MutableInputTensor};
  case MO::KernelOperandKind::Scalar:
    return KernelDeclScalarAdaptor{};
  case MO::KernelOperandKind::Opaque:
    return KernelDeclOpaqueAdaptor{iface.getKernelOperandOpaqueName()};
  }
  return std::nullopt;
}

static ErrorOrSuccess
matchConvertedMOTypeToKernelDeclType(const KernelDeclOperandVariant &moType,
                                     const KernelDeclOperandVariant &kernelType,
                                     Type originalMOType) {
  if (moType == kernelType)
    return success();

  if (std::holds_alternative<KernelDeclVariadicTensorAdaptor>(kernelType)) {
    const auto *tensorType = std::get_if<KernelDeclTensorAdaptor>(&moType);
    if (!tensorType)
      return Error("MO type is not a tensor type");

    const auto &kernelTensor =
        std::get<KernelDeclVariadicTensorAdaptor>(kernelType);
    if (kernelTensor.mut != tensorType->mut)
      return Error("Tensor type mismatch");

    return success();
  }

  if (std::holds_alternative<KernelDeclScalarAdaptor>(kernelType)) {
    const auto *tensorType = std::get_if<KernelDeclTensorAdaptor>(&moType);
    if (!tensorType)
      return Error("MO type is not a tensor type");

    if (tensorType->mut)
      return Error("Expected an immutable tensor type");

    // Validate rank-0 / on-host only when the operand type carries that
    // information. MO tensors do; opaque codegen handles (`!mogg.tensor_`) do
    // not, and were already validated upstream, so they are trusted here.
    if (auto shaped = dyn_cast<MO::MOShapedTypeInterface>(originalMOType))
      if (shaped.getRank() != 0)
        return Error("MO tensor type should have a rank of 0");

    if (isa<MO::MOHasDeviceTypeInterface>(originalMOType) &&
        !MO::isOnHost(originalMOType))
      return Error("Scalars should always be on the host CPU");

    return success();
  }

  return Error("Mismatched types");
}

template <bool matchOperands>
static ErrorOr<KernelDeclIndexToOperandAdaptorMapping>
matchOperandsOrResultsKernelDecl(ValueRange values,
                                 const KernelDeclFunctionAdaptor &adaptor,
                                 StringRef opName,
                                 bool hasSameVariadicOperandSize,
                                 bool hasExtraOutputShapeArgument) {
  static const char *argumentKind = matchOperands ? "input" : "output";

  auto skipContextType =
      [](const llvm::SmallVector<KernelDeclOperandAdaptor> &operands,
         uint64_t &idx) -> std::optional<KernelDeclOperandAdaptor> {
    while (idx < operands.size()) {
      if (!operands[idx].isContextType())
        return operands[idx];
      ++idx;
    }
    return std::nullopt;
  };

  auto skipChainType = [](auto range, uint64_t &idx) -> std::optional<Type> {
    while (idx < range.size()) {
      if (!isa<MO::ChainType>(range[idx]))
        return range[idx];
      ++idx;
    }
    return std::nullopt;
  };

  const llvm::SmallVector<KernelDeclOperandAdaptor> &kernelOperands =
      matchOperands ? adaptor.inputArguments : adaptor.outputArguments;

  uint64_t mojoVariadicsCount = 0;
  uint64_t mojoNonVariadicsCount = 0;
  uint64_t moOperandsCount = 0;
  uint64_t moOpIndex = 0, kernelIndex = 0;

  while (true) {
    auto moType = skipChainType(values.getTypes(), moOpIndex);
    if (!moType.has_value())
      break;
    ++moOpIndex;
    ++moOperandsCount;
  }

  while (true) {
    auto kernelType = skipContextType(kernelOperands, kernelIndex);
    if (!kernelType.has_value())
      break;
    if (kernelType->isVariadicTensorType())
      ++mojoVariadicsCount;
    else
      ++mojoNonVariadicsCount;
    ++kernelIndex;
  }

  uint64_t variadicSize = 0;
  if (mojoVariadicsCount > 0) {
    if (mojoVariadicsCount > 1 && !hasSameVariadicOperandSize) {
      return Error(
          llvm::formatv("Op with multiple variadic operands must implement the "
                        "SameVariadicOperandSize interface. Custom ops are not "
                        "supported yet."));
    }
    uint64_t totalVariadicOperands =
        moOperandsCount < mojoNonVariadicsCount
            ? 0
            : moOperandsCount - mojoNonVariadicsCount;
    if (totalVariadicOperands % mojoVariadicsCount != 0) {
      return Error(llvm::formatv(
          "Op '{}' has a total {} variadic operands divided "
          "between {} variadic kernel operands. They are expected "
          "to divide evenly",
          opName, totalVariadicOperands, mojoVariadicsCount));
    }
    variadicSize = totalVariadicOperands / mojoVariadicsCount;
  }

  KernelDeclIndexToOperandAdaptorMapping mapping;
  moOpIndex = 0, kernelIndex = 0;
  uint64_t currVariadicSize = 0;
  bool matchingVariadicTensor = false;

  while (true) {
    auto moType = skipChainType(values.getTypes(), moOpIndex);
    auto kernelType = skipContextType(kernelOperands, kernelIndex);

    if (!moType.has_value() || !kernelType.has_value())
      break;

    auto moTypeToKernelDeclOperand = MOTypeToKernelDeclOperand(*moType);
    if (!moTypeToKernelDeclOperand.has_value()) {
      return Error(
          llvm::formatv("Unsupported op argument type at index {}", moOpIndex));
    }

    auto errorOnMatch = matchConvertedMOTypeToKernelDeclType(
        *moTypeToKernelDeclOperand, kernelType->underlyingType, *moType);

    if (errorOnMatch.isError()) {
      return Error(llvm::formatv(
          "Op '{}' {} at position {} does not match registered kernel operand "
          "at position {}: {} vs {}. Detail: {}",
          opName, argumentKind, moOpIndex,
          kernelType->sourceDescriptor->position, *moTypeToKernelDeclOperand,
          *kernelType, errorOnMatch.getError()));
    }

    mapping[moOpIndex] = *kernelType;

    if (kernelType->isVariadicTensorType()) {
      matchingVariadicTensor = true;
      if (++currVariadicSize == variadicSize) {
        matchingVariadicTensor = false;
        currVariadicSize = 0;
      }
    }

    ++moOpIndex;
    kernelIndex += !matchingVariadicTensor;
  }

  if (matchingVariadicTensor)
    skipContextType(kernelOperands, ++kernelIndex);

  bool moOpEnd = moOpIndex == values.size();
  bool kernelEnd = kernelIndex == kernelOperands.size();

  if (hasExtraOutputShapeArgument)
    return mapping;

  if (!kernelEnd) {
    return Error(llvm::formatv(
        "Could not provide all {} arguments to op '{}': missing operands "
        "starting from argument named '{}'",
        argumentKind, opName,
        kernelOperands[kernelIndex].sourceDescriptor->sourceName.str()));
  }

  if constexpr (matchOperands) {
    if (!moOpEnd) {
      return Error(
          llvm::formatv("Op '{}' has too many operands: operands starting "
                        "at position {} has no equivalent in kernel",
                        opName, moOpIndex));
    }
    return mapping;
  }

  auto moType = skipChainType(values.getTypes(), moOpIndex);

  // A by-ref result with an unrecognized underlying type
  // (KernelDeclUnsupported) is treated as absent, matching the old
  // MojoKernelFunctionAdaptor behavior which skipped null argTypeName entries.
  bool hasValidOutputResult =
      adaptor.outputResult.has_value() &&
      !std::holds_alternative<KernelDeclUnsupportedAdaptor>(
          adaptor.outputResult->underlyingType);

  if (!moType.has_value() && !hasValidOutputResult)
    return mapping;

  if (moType.has_value() && !hasValidOutputResult) {
    return Error(llvm::formatv(
        "Op '{}' has too many results: results starting at position {} have no "
        "equivalent output argument or function result in kernel",
        opName, moOpIndex));
  }

  if (!moType.has_value() && hasValidOutputResult) {
    return Error(llvm::formatv(
        "Op '{}' is missing a result for kernel function result of type {}",
        opName, *adaptor.outputResult));
  }

  auto moTypeToKernelDeclOperand = MOTypeToKernelDeclOperand(*moType);
  if (!moTypeToKernelDeclOperand.has_value()) {
    return Error(
        llvm::formatv("Op '{}': Unsupported op result type at position {}",
                      opName, moOpIndex));
  }

  auto errorOnMatch = matchConvertedMOTypeToKernelDeclType(
      *moTypeToKernelDeclOperand, adaptor.outputResult->underlyingType,
      *moType);
  if (errorOnMatch.isError()) {
    return Error(llvm::formatv(
        "Op '{}' output at position {} does not match kernel function result; "
        "op has {} but kernel has {}. Detail: {}",
        opName, moOpIndex, *moTypeToKernelDeclOperand, *adaptor.outputResult,
        errorOnMatch.getError()));
  }
  mapping[moOpIndex] = *adaptor.outputResult;

  return mapping;
}

ErrorOr<KernelDeclOperandMapping>
mapToKernelImplementation(ValueRange operands, ValueRange results,
                          const KernelDeclFunctionAdaptor &adaptor,
                          StringRef opName, bool hasSameVariadicOperandSize,
                          bool hasExtraOutputShapeArgument) {
  auto matchedInputsOrError = matchOperandsOrResultsKernelDecl<true>(
      operands, adaptor, opName, hasSameVariadicOperandSize,
      hasExtraOutputShapeArgument);
  if (matchedInputsOrError.isError())
    return matchedInputsOrError.takeError();

  auto matchedOutputsOrError = matchOperandsOrResultsKernelDecl<false>(
      results, adaptor, opName, hasSameVariadicOperandSize,
      hasExtraOutputShapeArgument);
  if (matchedOutputsOrError.isError())
    return matchedOutputsOrError.takeError();

  return KernelDeclOperandMapping{matchedInputsOrError.takeValue(),
                                  matchedOutputsOrError.takeValue()};
}

ErrorOr<KernelDeclOperandMapping>
mapMOOpToKernelImplementation(Operation *op,
                              const KernelDeclFunctionAdaptor &adaptor) {
  return mapToKernelImplementation(
      op->getOperands(), op->getResults(), adaptor, moOperationName(op),
      isa<MO::SameVariadicOperandSizeInterface>(op),
      opHasExtraOutputShapeArgument(op));
}

} // namespace M::MOGG
