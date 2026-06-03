//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/GraphCompiler/MOGGDialect/Analysis/KernelDeclAdaptor.h"
#include "GraphCompiler/GraphCompiler/MOGGDialect/MOGGPasses.h"

using namespace M;
using namespace KGEN;
using namespace M::MOGG;

namespace M::MOGG {

#define GEN_PASS_DEF_VERIFYKERNELS
#include "GraphCompiler/GraphCompiler/MOGGDialect/MOGGPasses.h.inc"

namespace {

template <typename Func, typename... Args>
[[nodiscard]] LogicalResult setFlagIfFailed(bool &flag, Func &&func,
                                            Args &&...args) {
  LogicalResult ret = func(std::forward<Args>(args)...);
  if (ret.failed())
    flag |= true;

  return ret;
}

//===----------------------------------------------------------------------===//
// KernelDeclAdaptor-based verification (mogg.kernel.decl path)
//===----------------------------------------------------------------------===//

LogicalResult
declVerifyShapeFunctionHasNoDPSOutputs(KernelDeclOp decl,
                                       const KernelDeclAdaptor &adaptor) {
  if (!adaptor.shapeFunction.has_value())
    return success();
  if (!adaptor.shapeFunction->outputArguments.empty()) {
    decl->emitError() << "Shape functions must not have any output tensor "
                         "arguments";
    return failure();
  }
  if (!adaptor.shapeFunction->outputResult.has_value()) {
    decl->emitError() << "Shape functions must return a shape";
    return failure();
  }
  return success();
}

// Return `s` with its first ASCII letter uppercased. Used so error messages
// that start with a role label like "execute" or "build_graph" begin with a
// capital, while mid-sentence uses keep the original lowercase.
static std::string capitalize(llvm::StringRef s) {
  if (s.empty())
    return "";
  std::string out = s.str();
  out[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(out[0])));
  return out;
}

/// Walk the input argument lists of `lhs` and `rhs` in parallel, skipping
/// launch args (DeviceContext / DeviceGraphBuilder) from either side, and
/// require pairwise compatibility. `lhsKind` / `rhsKind` are lowercase role
/// labels used in error messages (e.g. "execute" / "shape").
///
/// A pair is compatible when types match exactly, or when the lhs side has a
/// `MutableInputTensor` and the rhs side has a non-mutable tensor (shape /
/// build_graph cannot mutate their inputs, so a relaxed match is allowed).
LogicalResult verifyInputArgumentsAlign(KernelDeclOp decl,
                                        const KernelDeclFunctionAdaptor &lhs,
                                        llvm::StringRef lhsKind,
                                        const KernelDeclFunctionAdaptor &rhs,
                                        llvm::StringRef rhsKind) {
  auto skipLaunchArg =
      [](const llvm::SmallVector<KernelDeclOperandAdaptor> &inputs,
         uint64_t &idx) -> std::optional<KernelDeclOperandAdaptor> {
    while (idx < inputs.size()) {
      const auto &arg = inputs[idx];
      if (!arg.isLaunchArgument())
        return arg;
      ++idx;
    }
    return std::nullopt;
  };

  uint64_t lhsIndex = 0, rhsIndex = 0;
  while (true) {
    const auto lhsArgOpt = skipLaunchArg(lhs.inputArguments, lhsIndex);
    const auto rhsArgOpt = skipLaunchArg(rhs.inputArguments, rhsIndex);
    if (!lhsArgOpt.has_value() || !rhsArgOpt.has_value())
      break;

    auto isCompatibleArgs = [&]() {
      if (*lhsArgOpt == *rhsArgOpt)
        return true;
      return lhsArgOpt->isMutableTensorType() &&
             (rhsArgOpt->isTensorType() && !rhsArgOpt->isMutableTensorType());
    };

    if (!isCompatibleArgs()) {
      return decl->emitError()
             << capitalize(rhsKind) << " function and " << lhsKind
             << " function input argument types do not match. "
             << capitalize(lhsKind) << " function input at position "
             << lhsArgOpt->sourceDescriptor->position << " is a "
             << lhsArgOpt->underlyingType << " but corresponding " << rhsKind
             << " function argument at position " << rhsIndex << " is a "
             << rhsArgOpt->underlyingType;
    }
    ++lhsIndex;
    ++rhsIndex;
  }

  bool lhsEnd = lhsIndex == lhs.inputArguments.size();
  bool rhsEnd = rhsIndex == rhs.inputArguments.size();

  if (!lhsEnd || !rhsEnd) {
    std::stringstream errorMessage;
    errorMessage << capitalize(lhsKind) << " and " << rhsKind.str()
                 << " functions do not have the same input tensors. ";
    auto appendUnmatchedInputs =
        [&](const llvm::SmallVector<KernelDeclOperandAdaptor> &inputs,
            uint64_t idx) {
          while (skipLaunchArg(inputs, idx).has_value()) {
            errorMessage << inputs[idx].sourceDescriptor->position << ",";
            ++idx;
          }
        };
    if (lhsEnd) {
      errorMessage << capitalize(rhsKind) << " function inputs at positions (";
      appendUnmatchedInputs(rhs.inputArguments, rhsIndex);
      errorMessage << ") do not appear in the " << lhsKind.str() << " function";
    } else {
      errorMessage << capitalize(lhsKind) << " function inputs at positions (";
      appendUnmatchedInputs(lhs.inputArguments, lhsIndex);
      errorMessage << ") do not appear in the " << rhsKind.str() << " function";
    }
    return decl->emitError(errorMessage.str());
  }

  return success();
}

LogicalResult declVerifyShapeFunctionInputsMatchExecuteFunction(
    KernelDeclOp decl, const KernelDeclAdaptor &adaptor) {
  if (!adaptor.shapeFunction.has_value())
    return success();

  if (adaptor.executeFunction.has_value() &&
      verifyInputArgumentsAlign(decl, *adaptor.executeFunction, "execute",
                                *adaptor.shapeFunction, "shape")
          .failed())
    return failure();

  if (adaptor.buildGraphFunction.has_value() &&
      verifyInputArgumentsAlign(decl, *adaptor.buildGraphFunction,
                                "build_graph", *adaptor.shapeFunction, "shape")
          .failed())
    return failure();

  return success();
}

/// build_graph must align with execute modulo their launch arguments
/// (DeviceContext vs DeviceGraphBuilder). Same tensor signature is required.
LogicalResult
declVerifyExecuteMatchesBuildGraph(KernelDeclOp decl,
                                   const KernelDeclAdaptor &adaptor) {
  if (!adaptor.executeFunction.has_value() ||
      !adaptor.buildGraphFunction.has_value())
    return success();
  return verifyInputArgumentsAlign(decl, *adaptor.executeFunction, "execute",
                                   *adaptor.buildGraphFunction, "build_graph");
}

/// A build_graph method records nodes against a DeviceGraphBuilder, so the
/// signature must explicitly accept one. There is no meaningful default for
/// this argument.
LogicalResult declVerifyBuildGraphHasBuilder(KernelDeclOp decl,
                                             const KernelDeclAdaptor &adaptor) {
  if (!adaptor.buildGraphFunction.has_value())
    return success();

  bool hasBuilder = llvm::any_of(
      adaptor.buildGraphFunction->inputArguments,
      [](const KernelDeclOperandAdaptor &arg) { return arg.isBuilderType(); });

  if (!hasBuilder) {
    decl->emitError()
        << "build_graph method must accept a DeviceGraphBuilder argument";
    return failure();
  }
  return success();
}

LogicalResult declVerifyExecuteFunctionDoesNotReturnATensor(
    KernelDeclOp decl, const KernelDeclAdaptor &adaptor) {
  auto doesNotReturnTensor =
      [&decl](const std::optional<KernelDeclFunctionAdaptor> &func)
      -> LogicalResult {
    if (!func.has_value() || !func->outputResult.has_value())
      return success();
    if (func->outputResult->isTensorType()) {
      decl->emitError()
          << "Value semantics kernels (return a tensor) are not yet supported";
      return failure();
    }
    return success();
  };

  if (doesNotReturnTensor(adaptor.executeFunction).failed() ||
      doesNotReturnTensor(adaptor.buildGraphFunction).failed())
    return failure();
  return success();
}

LogicalResult
declVerifyOutputsAppearBeforeInputs(KernelDeclOp decl,
                                    const KernelDeclAdaptor &adaptor) {
  auto outputsBeforeInputs =
      [&decl](const std::optional<KernelDeclFunctionAdaptor> &func)
      -> LogicalResult {
    if (!func.has_value())
      return success();
    int64_t indexOfFirstInput = std::numeric_limits<int64_t>::max();
    for (const auto &arg : func->inputArguments) {
      indexOfFirstInput =
          std::min<int64_t>(indexOfFirstInput, arg.sourceDescriptor->position);
    }
    int64_t indexOfLastOutput = std::numeric_limits<int64_t>::min();
    for (const auto &arg : func->outputArguments) {
      indexOfLastOutput =
          std::max<int64_t>(indexOfLastOutput, arg.sourceDescriptor->position);
    }
    if (indexOfFirstInput < indexOfLastOutput) {
      decl->emitError()
          << "Outputs must appear before inputs in the function signature";
      return failure();
    }
    return success();
  };

  if (outputsBeforeInputs(adaptor.executeFunction).failed() ||
      outputsBeforeInputs(adaptor.buildGraphFunction).failed())
    return failure();
  return success();
}

/// Each kernel function variant has at most one launch arg
/// (DeviceContext / DeviceContextList on execute / shape, DeviceGraphBuilder
/// on build_graph) and the launch arg must come last in the signature.
LogicalResult
declVerifyContextsAppearLastInSignature(KernelDeclOp decl,
                                        const KernelDeclAdaptor &adaptor) {
  const auto launchArgsAreLast =
      [&decl](const std::optional<KernelDeclFunctionAdaptor> &func,
              llvm::StringRef argLower,
              llvm::StringRef argTitle) -> LogicalResult {
    if (!func.has_value())
      return success();
    if (llvm::count_if(func->inputArguments,
                       [](const KernelDeclOperandAdaptor &arg) {
                         return arg.isLaunchArgument();
                       }) > 1) {
      decl->emitError() << "There can be at most one " << argLower;
      return failure();
    }
    int64_t indexOfLastArgument = std::numeric_limits<int64_t>::min();
    int64_t indexOfLaunchArgument = std::numeric_limits<int64_t>::max();
    for (const auto &arg : func->inputArguments) {
      if (arg.isLaunchArgument())
        indexOfLaunchArgument = arg.sourceDescriptor->position;
      indexOfLastArgument = std::max<int64_t>(indexOfLastArgument,
                                              arg.sourceDescriptor->position);
    }
    if (indexOfLaunchArgument < indexOfLastArgument) {
      decl->emitError() << argTitle
                        << " argument must be the last argument "
                           "in the function signature";
      return failure();
    }
    return success();
  };

  if (launchArgsAreLast(adaptor.executeFunction, "context", "Context")
          .failed() ||
      launchArgsAreLast(adaptor.shapeFunction, "context", "Context").failed() ||
      launchArgsAreLast(adaptor.buildGraphFunction, "graph builder",
                        "Graph builder")
          .failed())
    return failure();
  return success();
}

LogicalResult
declVerifyOpaqueArgumentsArePassedByReference(KernelDeclOp decl,
                                              KernelDeclAdaptor &adaptor) {
  auto opaquesByReference =
      [&decl](std::optional<KernelDeclFunctionAdaptor> &func) -> LogicalResult {
    if (!func.has_value())
      return success();
    auto allArgs = llvm::concat<const KernelDeclOperandAdaptor>(
        func->outputArguments, func->inputArguments);
    for (const auto &arg : allArgs) {
      // Copy sig to call the non-const getArgConvention.
      auto sig = func->attr.getSignature();
      auto convention = sig.getArgConvention(arg.sourceDescriptor->position);
      if (arg.isOpaqueType() && (convention == KGEN::ArgConvention::OwnedMem ||
                                 convention == KGEN::ArgConvention::DeinitMem ||
                                 convention == KGEN::ArgConvention::OwnedReg)) {
        decl->emitError()
            << "Argument convention for argument at position "
            << arg.sourceDescriptor->position
            << " is invalid: Owned arguments not supported for custom types "
               "in kernels";
        return failure();
      }
      // Tensors must be passed by const reference (`read`/`read_mem`):
      // their mutability is encoded in the type itself via IOSpec, not in
      // the calling convention. Reject `mut`, `var`, owned, and `ref`.
      if (arg.isTensorType() && convention != KGEN::ArgConvention::ReadMem &&
          convention != KGEN::ArgConvention::ReadReg) {
        decl->emitError()
            << "Only the borrowed argument (read) convention is supported "
               "for tensor arguments (argument at position "
            << arg.sourceDescriptor->position << " uses '"
            << KGEN::stringifyArgConvention(convention) << "')";
        return failure();
      }
    }
    return success();
  };

  if (opaquesByReference(adaptor.executeFunction).failed() ||
      opaquesByReference(adaptor.shapeFunction).failed() ||
      opaquesByReference(adaptor.buildGraphFunction).failed())
    return failure();
  return success();
}

LogicalResult
declVerifyAtMostOneVariadicInputAndOutput(KernelDeclOp decl,
                                          const KernelDeclAdaptor &adaptor) {
  // TODO: (GEX-1994) Remove this.
  if (adaptor.registeredOpName.strref().starts_with("mo."))
    return success();

  const auto atMostOneVariadic =
      [&decl](const std::optional<KernelDeclFunctionAdaptor> &func)
      -> LogicalResult {
    if (!func.has_value())
      return success();
    if (llvm::count_if(func->inputArguments,
                       [](const KernelDeclOperandAdaptor &arg) {
                         return arg.isVariadicTensorType();
                       }) > 1) {
      decl->emitError()
          << "Kernel function can have at most one variadic input tensors";
      return failure();
    }
    if (llvm::count_if(func->outputArguments,
                       [](const KernelDeclOperandAdaptor &arg) {
                         return arg.isVariadicTensorType();
                       }) > 1) {
      decl->emitError()
          << "Kernel function can have at most one variadic output tensors";
      return failure();
    }
    return success();
  };

  if (atMostOneVariadic(adaptor.executeFunction).failed() ||
      atMostOneVariadic(adaptor.shapeFunction).failed() ||
      atMostOneVariadic(adaptor.buildGraphFunction).failed())
    return failure();
  return success();
}

LogicalResult
declVerifyExecuteNotReturningOpaqueAndThrows(KernelDeclOp decl,
                                             KernelDeclAdaptor &adaptor) {
  auto notOpaqueAndThrows =
      [&decl](std::optional<KernelDeclFunctionAdaptor> &func) -> LogicalResult {
    if (!func.has_value() || !func->outputResult.has_value())
      return success();
    auto sig = func->attr.getSignature();
    if (func->outputResult->isOpaqueType() && sig.isThrows()) {
      decl->emitError() << "Functions that raise and return a custom type are "
                           "not yet supported";
      return failure();
    }
    return success();
  };

  if (notOpaqueAndThrows(adaptor.executeFunction).failed() ||
      notOpaqueAndThrows(adaptor.buildGraphFunction).failed())
    return failure();
  return success();
}

LogicalResult declVerifyHasEntryPoint(KernelDeclOp decl,
                                      const KernelDeclAdaptor &adaptor) {
  if (!adaptor.executeFunction.has_value() &&
      !adaptor.elementwiseFunction.has_value()) {
    decl->emitError() << "The kernel must have an entry point named execute";
    return failure();
  }
  return success();
}

LogicalResult declVerifyElementwiseFunction(KernelDeclOp decl,
                                            const KernelDeclAdaptor &adaptor) {
  if (!adaptor.elementwiseFunction.has_value())
    return success();

  const KernelDeclFunctionAdaptor &elemwiseFn = *adaptor.elementwiseFunction;

  // Copy sig to call non-const isThrows.
  auto sig = elemwiseFn.attr.getSignature();
  if (sig.isThrows())
    return decl->emitError() << "Elementwise kernels can't throw";

  if (!elemwiseFn.outputArguments.empty())
    return decl->emitError() << "Elementwise kernels must return an SIMD value";
  if (!elemwiseFn.outputResult.has_value() ||
      !elemwiseFn.outputResult->isScalarType())
    return decl->emitError() << "Elementwise kernels must return an SIMD value";

  size_t operandsCount = 0;
  for (size_t i = 0, inputsCount = elemwiseFn.inputArguments.size();
       i < inputsCount; ++i) {
    // The last argument might be an IndexList. Skip it — it is an
    // elementwise-only concept.
    if (i + 1 == inputsCount &&
        elemwiseFn.inputArguments[i].isOpaqueWithTypeName(
            MOGG::kMojoInternalDpsIndexListTypeQualifiedName)) {
      continue;
    }
    ++operandsCount;
    if (!elemwiseFn.inputArguments[i].isScalarType())
      return decl->emitError() << "Elementwise kernels must have SIMD operands";
  }

  if (!operandsCount)
    return decl->emitError()
           << "Elementwise kernels must have at least one operand";

  return success();
}

LogicalResult
declVerifyTileElementwiseFunction(KernelDeclOp decl,
                                  const KernelDeclAdaptor &adaptor) {
  if (!adaptor.tileElementwiseFunction.has_value())
    return success();

  const KernelDeclFunctionAdaptor &fn = *adaptor.tileElementwiseFunction;

  auto sig = fn.attr.getSignature();
  if (sig.isThrows())
    return decl->emitError() << "Tile elementwise kernels can't throw";

  if (!fn.outputArguments.empty())
    return decl->emitError()
           << "Tile elementwise kernels must return a TileTensor value";
  if (!fn.outputResult.has_value() || !fn.outputResult->isTileType())
    return decl->emitError()
           << "Tile elementwise kernels must return a TileTensor value";

  size_t operandsCount = 0;
  for (const auto &arg : fn.inputArguments) {
    ++operandsCount;
    if (!arg.isTileType())
      return decl->emitError()
             << "Tile elementwise kernels must have TileTensor operands";
  }

  if (!operandsCount)
    return decl->emitError()
           << "Tile elementwise kernels must have at least one operand";

  return success();
}

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct VerifyKernelsPass : public impl::VerifyKernelsBase<VerifyKernelsPass> {
  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();

    bool failed = false;
    for (KernelDeclOp decl : moduleOp.getOps<KernelDeclOp>()) {
      KernelDeclAdaptor adaptor(decl);

      if (setFlagIfFailed(failed, declVerifyHasEntryPoint, decl, adaptor)
              .failed())
        continue;

      // Elementwise-only kernels: verify elementwise function(s).
      if (!adaptor.executeFunction.has_value()) {
        if (setFlagIfFailed(failed, declVerifyElementwiseFunction, decl,
                            adaptor)
                .failed())
          continue;
        if (setFlagIfFailed(failed, declVerifyTileElementwiseFunction, decl,
                            adaptor)
                .failed())
          continue;
        continue;
      }

      if (setFlagIfFailed(failed, declVerifyShapeFunctionHasNoDPSOutputs, decl,
                          adaptor)
              .failed())
        continue;

      if (setFlagIfFailed(failed,
                          declVerifyShapeFunctionInputsMatchExecuteFunction,
                          decl, adaptor)
              .failed())
        continue;

      if (setFlagIfFailed(failed, declVerifyBuildGraphHasBuilder, decl, adaptor)
              .failed())
        continue;

      if (setFlagIfFailed(failed, declVerifyExecuteMatchesBuildGraph, decl,
                          adaptor)
              .failed())
        continue;

      if (setFlagIfFailed(failed, declVerifyExecuteFunctionDoesNotReturnATensor,
                          decl, adaptor)
              .failed())
        continue;

      if (setFlagIfFailed(failed, declVerifyOutputsAppearBeforeInputs, decl,
                          adaptor)
              .failed())
        continue;

      if (setFlagIfFailed(failed, declVerifyContextsAppearLastInSignature, decl,
                          adaptor)
              .failed())
        continue;

      if (setFlagIfFailed(failed, declVerifyOpaqueArgumentsArePassedByReference,
                          decl, adaptor)
              .failed())
        continue;

      if (setFlagIfFailed(failed, declVerifyAtMostOneVariadicInputAndOutput,
                          decl, adaptor)
              .failed())
        continue;

      if (setFlagIfFailed(failed, declVerifyExecuteNotReturningOpaqueAndThrows,
                          decl, adaptor)
              .failed())
        continue;

      if (setFlagIfFailed(failed, declVerifyElementwiseFunction, decl, adaptor)
              .failed())
        continue;
    }

    if (failed)
      signalPassFailure();
  }
};

} // namespace

} // namespace M::MOGG
