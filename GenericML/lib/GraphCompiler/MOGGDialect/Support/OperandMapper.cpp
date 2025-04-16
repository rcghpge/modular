//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/MOGGDialect/Support/OperandMapper.h"
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
  if (isa<MO::ListType>(type))
    return ListOfTensorOperandAdaptor{IOSpec::FusedInputTensor};
  if (isa<MO::ScalarType>(type))
    return ScalarOperandAdaptor{};
  if (auto opaqueType = dyn_cast<MO::OpaqueType>(type))
    return OpaqueOperandAdaptor{opaqueType.getSymbol()};

  return std::nullopt;
}

static bool matchConvertedMOTypeToKernelType(
    const KGEN::MOGGPreElab::MojoKernelOperandVariant &moType,
    const KGEN::MOGGPreElab::MojoKernelOperandVariant &kernelType) {
  using namespace KGEN::MOGGPreElab;

  if (moType == kernelType)
    return true;

  if (std::holds_alternative<VariadicTensorOperandAdaptor>(kernelType)) {
    auto tensorType = std::get_if<TensorOperandAdaptor>(&moType);
    if (!tensorType)
      return false;

    return *reinterpret_cast<const TensorOperandAdaptor *>(&kernelType) ==
           *tensorType;
  }

  if (std::holds_alternative<ScalarOperandAdaptor>(kernelType)) {
    auto tensorType = std::get_if<TensorOperandAdaptor>(&moType);
    if (!tensorType)
      return false;

    return !tensorType->mut;
  }

  return false;
}

// If matchOperands is set to true, this function will not attempt to validate
// results. Conversely, if set to false, it will do extra checks beyond DPS
// arguments with result types.
template <bool matchOperands>
static ErrorOr<IndexToOperandAdaptorMapping> matchOperandsOrResults(
    Operation *op,
    const KGEN::MOGGPreElab::MojoKernelFunctionAdaptor<KGEN::GeneratorOp>
        &adaptor) {
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

  IndexToOperandAdaptorMapping mapping;
  uint64_t moOpIndex = 0, kernelIndex = 0;
  bool matchingVariadicTensor = false;
  const SmallVector<MojoKernelOperandAdaptor> &kernelOperands =
      matchOperands ? adaptor.inputArguments : adaptor.outputArguments;
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

    bool matched = matchConvertedMOTypeToKernelType(*moTypeToMojoKernelOperand,
                                                    kernelType->underlyingType);

    // If there is a mismatch but we were previously matching on a variadic, go
    // to the next kernel argument and retry.
    if (!matched && matchingVariadicTensor) {
      matchingVariadicTensor = false;
      ++kernelIndex;
      continue;
    }

    if (!matched) {
      return Error(llvm::formatv(
          "Op {} at position {} does not match registered kernel operand at "
          "position {}: {} vs {}",
          argumentKind, moOpIndex, kernelType->sourceDescriptor->position,
          *moTypeToMojoKernelOperand, *kernelType));
    }

    mapping[moOpIndex] = *kernelType;

    // If the current kernel argument is a variadic tensor type, then we set a
    // flag such that we don't iterate and stay here until there is a mismatch.
    if (kernelType->isVariadicTensorType())
      matchingVariadicTensor = true;

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
        "Could not provide all {} arguments to kernel: missing operands "
        "starting from argument named '{}'",
        argumentKind,
        kernelOperands[kernelIndex].sourceDescriptor->sourceName.str()));
  }

  // If we are matching inputs, we must ensure that we consumed all the MO op's
  // operands or else, we are missing inputs on the kernel.
  if constexpr (matchOperands) {
    if (!moOpEnd) {
      return Error(llvm::formatv("Op has too many operands: operands starting "
                                 "at position {} has no equivalent in kernel",
                                 moOpIndex));
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
        "Op has too many results: results starting at position {} have no "
        "equivalent output argument or function result in kernel",
        moOpIndex));
  }

  // 2. We have a remaining result on the kernel but nothing on the op.
  if (!moType.has_value() && adaptor.outputResult.has_value()) {
    return Error(llvm::formatv(
        "Op is missing a result for kernel function result of type {}",
        *adaptor.outputResult));
  }

  auto moTypeToMojoKernelOperand = MOTypeToMojoKernelOperand(*moType);
  if (!moTypeToMojoKernelOperand.has_value()) {
    return Error(
        llvm::formatv("Unsupported op result type at position {}", moOpIndex));
  }

  // 3. We have results for both but they don't match.
  if (!matchConvertedMOTypeToKernelType(*moTypeToMojoKernelOperand,
                                        adaptor.outputResult->underlyingType)) {
    return Error(llvm::formatv("Op output at position {} does not match kernel "
                               "function result; op has {} but kernel has {}",
                               moOpIndex, *moTypeToMojoKernelOperand,
                               *adaptor.outputResult));
  }
  mapping[moOpIndex] = *adaptor.outputResult;

  return mapping;
}

ErrorOr<OperandMapping> mapMOOpToKernelImplementation(
    Operation *op,
    const KGEN::MOGGPreElab::MojoKernelFunctionAdaptor<KGEN::GeneratorOp>
        &adaptor) {
  auto matchedInputsOrError = matchOperandsOrResults<true>(op, adaptor);
  if (matchedInputsOrError.isError())
    return matchedInputsOrError.takeError();

  auto matchedOutputsOrError = matchOperandsOrResults<false>(op, adaptor);
  if (matchedOutputsOrError.isError())
    return matchedOutputsOrError.takeError();

  return OperandMapping{matchedInputsOrError.takeValue(),
                        matchedOutputsOrError.takeValue()};
}

} // namespace M::MOGG
