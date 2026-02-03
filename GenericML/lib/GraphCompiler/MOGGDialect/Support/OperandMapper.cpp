//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/MOGGDialect/Support/OperandMapper.h"
#include "GenericML/GraphCompiler/MODialect/MODevices.h"
#include "GenericML/GraphCompiler/MODialect/MOOps.h"

namespace M::MOGG {

static bool opHasExtraOutputShapeArgument(Operation *op) {
  return isa<MO::Static::ReshapeOp, MO::Static::BroadcastToOp>(op);
}

static std::optional<KGEN::MOGGPreElab::MojoKernelOperandVariant>
MOTypeToMojoKernelOperand(Type type) {
  using namespace KGEN::MOGGPreElab;
  if (isa<MO::TensorType>(type))
    return TensorOperandAdaptor{IOSpec::InputTensor};
  if (isa<MO::BufferType>(type))
    return TensorOperandAdaptor{IOSpec::MutableInputTensor};
  if (isa<MO::ScalarType>(type))
    return ScalarOperandAdaptor{};
  if (auto opaqueType = dyn_cast<MO::OpaqueType>(type))
    return OpaqueOperandAdaptor{opaqueType.getSymbol()};

  return std::nullopt;
}

static ErrorOrSuccess matchConvertedMOTypeToKernelType(
    const KGEN::MOGGPreElab::MojoKernelOperandVariant &moType,
    const KGEN::MOGGPreElab::MojoKernelOperandVariant &kernelType,
    Type originalMOType) {
  using namespace KGEN::MOGGPreElab;

  if (moType == kernelType)
    return success();

  if (std::holds_alternative<VariadicTensorOperandAdaptor>(kernelType)) {
    auto tensorType = std::get_if<TensorOperandAdaptor>(&moType);
    if (!tensorType)
      return Error("MO type is not a tensor type");

    if (!(*reinterpret_cast<const TensorOperandAdaptor *>(&kernelType) ==
          *tensorType)) {
      return Error("Tensor type mismatch");
    }

    return success();
  }

  if (std::holds_alternative<ScalarOperandAdaptor>(kernelType)) {
    auto tensorType = std::get_if<TensorOperandAdaptor>(&moType);
    if (!tensorType)
      return Error("MO type is not a tensor type");

    if (tensorType->mut)
      return Error("Expected an immutable tensor type");

    auto moTensorType = cast<MO::TensorType>(originalMOType);
    if (moTensorType.getRank() != 0)
      return Error("MO tensor type should have a rank of 0");

    if (!isOnHost(moTensorType))
      return Error("Scalars should always be on the host CPU");

    return success();
  }

  return Error("Mismatched types");
}

static StringRef moOperationName(Operation *op) {
  if (auto customOp = dyn_cast<MO::CustomOp>(op))
    return customOp.getSymbol();

  return op->getName().getStringRef();
}

// If matchOperands is set to true, this function will not attempt to validate
// results. Conversely, if set to false, it will do extra checks beyond DPS
// arguments with result types.
template <bool matchOperands>
static ErrorOr<IndexToOperandAdaptorMapping> matchOperandsOrResults(
    Operation *op,
    const KGEN::MOGGPreElab::MojoKernelFunctionAdaptor &adaptor) {
  using namespace KGEN::MOGGPreElab;
  static const char *argumentKind = matchOperands ? "input" : "output";

  // Iterates through `operands` until it finds a non context type.
  auto skipContextType =
      [](const SmallVector<MojoKernelOperandAdaptor> &operands,
         uint64_t &idx) -> std::optional<MojoKernelOperandAdaptor> {
    while (idx < operands.size()) {
      if (!operands[idx].isContextType())
        return operands[idx];
      ++idx;
    }

    return std::nullopt;
  };

  // Iterates through the range until it finds a non chain type.
  /// TODO(GEX-2066):
  auto skipChainType = [](auto range, uint64_t &idx) -> std::optional<Type> {
    while (idx < range.size()) {
      if (!isa<MO::ChainType>(range[idx]))
        return range[idx];
      ++idx;
    }
    return std::nullopt;
  };

  const SmallVector<MojoKernelOperandAdaptor> &kernelOperands =
      matchOperands ? adaptor.inputArguments : adaptor.outputArguments;

  // When the kernel have multiple variadic operands, we need to figure out
  // which operand is associated to which variadic. Today we only support one
  // mode: SameVariadicOperandSize: All variadic inputs are divided evenly
  // between the variadics. We first count the expected size of a variadic
  // before checking the operands.
  uint64_t mojoVariadicsCount = 0;
  uint64_t mojoNonVariadicsCount = 0;
  uint64_t moOperandsCount = 0;
  uint64_t moOpIndex = 0, kernelIndex = 0;
  // Go through all operands and count them beforehand.
  while (true) {
    auto moType = matchOperands
                      ? skipChainType(op->getOperandTypes(), moOpIndex)
                      : skipChainType(op->getResultTypes(), moOpIndex);
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

  // Currently variadics are expected to have the same size.
  uint64_t variadicSize = 0;
  if (mojoVariadicsCount > 0) {
    if (mojoVariadicsCount > 1 &&
        !isa<MO::SameVariadicOperandSizeInterface>(op)) {
      return Error(
          llvm::formatv("Op with multiple variadic operands must implement the "
                        "SameVariadicOperandSize interface. Custom ops are not "
                        "supported yet."));
    }
    // We'll check that we don't get empty variadic / missing operands later.
    uint64_t totalVariadicOperands =
        moOperandsCount < mojoNonVariadicsCount
            ? 0
            : moOperandsCount - mojoNonVariadicsCount;
    if (totalVariadicOperands % mojoVariadicsCount != 0) {
      return Error(llvm::formatv("Op '{}' has a total {} variadic operands "
                                 "divided between {} variadic kernel operands"
                                 "They are expected to divide evenly",
                                 moOperationName(op), totalVariadicOperands,
                                 mojoVariadicsCount));
    }
    variadicSize = totalVariadicOperands / mojoVariadicsCount;
  }

  IndexToOperandAdaptorMapping mapping;
  moOpIndex = 0, kernelIndex = 0;
  uint64_t currVariadicSize = 0;
  bool matchingVariadicTensor = false;
  while (true) {
    auto moType = matchOperands
                      ? skipChainType(op->getOperandTypes(), moOpIndex)
                      : skipChainType(op->getResultTypes(), moOpIndex);
    auto kernelType = skipContextType(kernelOperands, kernelIndex);

    // If we no longer have types to look at, we get out of the loop.
    if (!moType.has_value() || !kernelType.has_value())
      break;

    // Convert the mlir::Type to the equivalent mojo kernel type.
    auto moTypeToMojoKernelOperand = MOTypeToMojoKernelOperand(*moType);
    if (!moTypeToMojoKernelOperand.has_value()) {
      return Error(
          llvm::formatv("Unsupported op argument type at index {}", moOpIndex));
    }

    auto errorOnMatch = matchConvertedMOTypeToKernelType(
        *moTypeToMojoKernelOperand, kernelType->underlyingType, *moType);

    if (errorOnMatch.isError()) {
      return Error(llvm::formatv(
          "Op '{}' {} at position {} does not match registered kernel "
          "operand at "
          "position {}: {} vs {}. Detail: {}",
          moOperationName(op), argumentKind, moOpIndex,
          kernelType->sourceDescriptor->position, *moTypeToMojoKernelOperand,
          *kernelType, errorOnMatch.getError()));
    }

    mapping[moOpIndex] = *kernelType;

    // Keep iterating over the same variadic until we mapped the expected number
    // of operands per variadic.
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

  // At the end of the iteration, if we were matching a variadic, we must
  // increment once to ensure we reach the end (assuming a trailing context
  // type). Note that if we were on a variadic type but we never match (first
  // attempt was a mismatch), the flag will not be set and thus we will not
  // mistakenly increment.
  if (matchingVariadicTensor)
    skipContextType(kernelOperands, ++kernelIndex);

  bool moOpEnd =
      moOpIndex == (matchOperands ? op->getNumOperands() : op->getNumResults());
  bool kernelEnd = kernelIndex == kernelOperands.size();

  // Not attempting to do more error checking on certain ops. External users
  // don't implement these and they are core MO operations so there is no added
  // value, really.
  if (opHasExtraOutputShapeArgument(op))
    return mapping;

  // If we have not consumed all kernel types, this is always an error. It means
  // there are missing operands or results form the MO op.
  if (!kernelEnd) {
    return Error(llvm::formatv(
        "Could not provide all {} arguments to op '{}': missing "
        "operands "
        "starting from argument named '{}'",
        argumentKind, moOperationName(op),
        kernelOperands[kernelIndex].sourceDescriptor->sourceName.str()));
  }

  // If we are matching inputs, we must ensure that we consumed all the MO op's
  // operands or else, we are missing inputs on the kernel.
  if constexpr (matchOperands) {
    if (!moOpEnd) {
      return Error(
          llvm::formatv("Op '{}' has too many operands: operands starting "
                        "at position {} has no equivalent in kernel",
                        moOperationName(op), moOpIndex));
    }
    return mapping;
  }

  // If there are no more op results and return values from the kernel, we are
  // done.
  auto moType = skipChainType(op->getResultTypes(), moOpIndex);
  if (!moType.has_value() && !adaptor.outputResult.has_value())
    return mapping;

  // Otherwise, there are three failure cases:
  // 1. We have remaining results on the op but not on the kernel.
  if (moType.has_value() && !adaptor.outputResult.has_value()) {
    return Error(llvm::formatv(
        "Op '{}' has too many results: results starting at position {} "
        "have no "
        "equivalent output argument or function result in kernel",
        moOperationName(op), moOpIndex));
  }

  // 2. We have a remaining result on the kernel but nothing on the op.
  if (!moType.has_value() && adaptor.outputResult.has_value()) {
    return Error(
        llvm::formatv("Op '{}' is missing a result for kernel function "
                      "result of type {}",
                      moOperationName(op), *adaptor.outputResult));
  }

  auto moTypeToMojoKernelOperand = MOTypeToMojoKernelOperand(*moType);
  if (!moTypeToMojoKernelOperand.has_value()) {
    return Error(
        llvm::formatv("Op '{}': Unsupported op result type at position {}",
                      moOperationName(op), moOpIndex));
  }

  // 3. We have results for both but they don't match.
  auto errorOnMatch = matchConvertedMOTypeToKernelType(
      *moTypeToMojoKernelOperand, adaptor.outputResult->underlyingType,
      *moType);
  if (errorOnMatch.isError()) {
    return Error(llvm::formatv(
        "Op '{}' output at position {} does not match kernel "
        "function result; op has {} but kernel has {}. Detail: {}",
        moOperationName(op), moOpIndex, *moTypeToMojoKernelOperand,
        *adaptor.outputResult, errorOnMatch.getError()));
  }
  mapping[moOpIndex] = *adaptor.outputResult;

  return mapping;
}

ErrorOr<OperandMapping> mapMOOpToKernelImplementation(
    Operation *op,
    const KGEN::MOGGPreElab::MojoKernelFunctionAdaptor &adaptor) {
  auto matchedInputsOrError = matchOperandsOrResults<true>(op, adaptor);
  if (matchedInputsOrError.isError())
    return matchedInputsOrError.takeError();

  auto matchedOutputsOrError = matchOperandsOrResults<false>(op, adaptor);
  if (matchedOutputsOrError.isError())
    return matchedOutputsOrError.takeError();

  return OperandMapping{matchedInputsOrError.takeValue(),
                        matchedOutputsOrError.takeValue()};
}

InverseOperandMapping invertMapping(const OperandMapping &mapping) {
  InverseOperandMapping ret;
  for (auto &&[idx, operand] : mapping.operandMapping) {
    ret.inputArgumentsMapping[operand.sourceDescriptor->position].push_back(
        idx);
  }

  for (auto &&[idx, operand] : mapping.resultMapping) {
    if (operand.sourceDescriptor.has_value()) {
      if (operand.sourceDescriptor->isByRefResult) {
        ret.resultMapping = idx;
      } else {
        ret.outputArgumentsMapping[operand.sourceDescriptor->position]
            .push_back(idx);
      }
    } else {
      ret.resultMapping = idx;
    }
  }

  return ret;
}

} // namespace M::MOGG
