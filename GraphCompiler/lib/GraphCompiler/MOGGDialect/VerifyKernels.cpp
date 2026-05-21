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

LogicalResult declVerifyShapeFunctionInputsMatchExecuteFunction(
    KernelDeclOp decl, const KernelDeclAdaptor &adaptor) {
  if (!adaptor.shapeFunction.has_value())
    return success();

  auto skipContextType =
      [](const llvm::SmallVector<KernelDeclOperandAdaptor> &inputs,
         uint64_t &idx) -> std::optional<KernelDeclOperandAdaptor> {
    while (idx < inputs.size()) {
      const auto &arg = inputs[idx];
      if (!arg.isContextType())
        return arg;
      ++idx;
    }
    return std::nullopt;
  };

  uint64_t execIndex = 0, shapeIndex = 0;
  while (true) {
    const auto execArgOpt =
        skipContextType(adaptor.executeFunction->inputArguments, execIndex);
    const auto shapeArgOpt =
        skipContextType(adaptor.shapeFunction->inputArguments, shapeIndex);
    if (!execArgOpt.has_value() || !shapeArgOpt.has_value())
      break;

    auto isCompatibleArgs = [&]() {
      if (*execArgOpt == *shapeArgOpt)
        return true;
      return execArgOpt->isMutableTensorType() &&
             (shapeArgOpt->isTensorType() &&
              !shapeArgOpt->isMutableTensorType());
    };

    if (!isCompatibleArgs()) {
      return decl->emitError()
             << "Shape function and execute function input argument types do "
                "not match. Execute function input at position "
             << execArgOpt->sourceDescriptor->position << " is a "
             << execArgOpt->underlyingType
             << " but corresponding shape function argument at position "
             << shapeIndex << " is a " << shapeArgOpt->underlyingType;
    }
    ++execIndex;
    ++shapeIndex;
  }

  bool execEnd = execIndex == adaptor.executeFunction->inputArguments.size();
  bool shapeEnd = shapeIndex == adaptor.shapeFunction->inputArguments.size();

  if (!execEnd || !shapeEnd) {
    std::stringstream errorMessage;
    errorMessage
        << "Execute and shape functions do not have the same input tensors. ";
    auto appendUnmatchedInputs =
        [&](const llvm::SmallVector<KernelDeclOperandAdaptor> &inputs,
            uint64_t idx) {
          while (skipContextType(inputs, idx).has_value()) {
            errorMessage << inputs[idx].sourceDescriptor->position << ",";
            ++idx;
          }
        };
    if (execEnd) {
      errorMessage << "Shape function inputs at positions (";
      appendUnmatchedInputs(adaptor.shapeFunction->inputArguments, shapeIndex);
      errorMessage << ") do not appear in the execute function";
    } else {
      errorMessage << "Execute function inputs at positions (";
      appendUnmatchedInputs(adaptor.executeFunction->inputArguments, execIndex);
      errorMessage << ") do not appear in the shape function";
    }
    return decl->emitError(errorMessage.str());
  }

  return success();
}

LogicalResult declVerifyExecuteFunctionDoesNotReturnATensor(
    KernelDeclOp decl, const KernelDeclAdaptor &adaptor) {
  if (!adaptor.executeFunction->outputResult.has_value())
    return success();
  if (adaptor.executeFunction->outputResult->isTensorType()) {
    decl->emitError()
        << "Value semantics kernels (return a tensor) are not yet supported";
    return failure();
  }
  return success();
}

LogicalResult
declVerifyOutputsAppearBeforeInputs(KernelDeclOp decl,
                                    const KernelDeclAdaptor &adaptor) {
  int64_t indexOfFirstInput = std::numeric_limits<int64_t>::max();
  for (const auto &arg : adaptor.executeFunction->inputArguments) {
    indexOfFirstInput =
        std::min<int64_t>(indexOfFirstInput, arg.sourceDescriptor->position);
  }

  int64_t indexOfLastOutput = std::numeric_limits<int64_t>::min();
  for (const auto &arg : adaptor.executeFunction->outputArguments) {
    indexOfLastOutput =
        std::max<int64_t>(indexOfLastOutput, arg.sourceDescriptor->position);
  }

  if (indexOfFirstInput < indexOfLastOutput) {
    decl->emitError()
        << "Outputs must appear before inputs in the function signature";
    return failure();
  }
  return success();
}

LogicalResult
declVerifyContextsAppearLastInSignature(KernelDeclOp decl,
                                        const KernelDeclAdaptor &adaptor) {
  const auto contextsAreLast =
      [&decl](const std::optional<KernelDeclFunctionAdaptor> &func)
      -> LogicalResult {
    if (!func.has_value())
      return success();
    if (llvm::count_if(func->inputArguments,
                       [](const KernelDeclOperandAdaptor &arg) {
                         return arg.isContextType();
                       }) > 1) {
      decl->emitError() << "There can be at most one context";
      return failure();
    }
    int64_t indexOfLastArgument = std::numeric_limits<int64_t>::min();
    int64_t indexOfContextArgument = std::numeric_limits<int64_t>::max();
    for (const auto &arg : func->inputArguments) {
      if (arg.isContextType())
        indexOfContextArgument = arg.sourceDescriptor->position;
      indexOfLastArgument = std::max<int64_t>(indexOfLastArgument,
                                              arg.sourceDescriptor->position);
    }
    if (indexOfContextArgument < indexOfLastArgument) {
      decl->emitError() << "Context argument must be the last argument in the "
                           "function signature";
      return failure();
    }
    return success();
  };

  if (contextsAreLast(adaptor.executeFunction).failed() ||
      contextsAreLast(adaptor.shapeFunction).failed())
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
      opaquesByReference(adaptor.shapeFunction).failed())
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
      atMostOneVariadic(adaptor.shapeFunction).failed())
    return failure();
  return success();
}

LogicalResult
declVerifyExecuteNotReturningOpaqueAndThrows(KernelDeclOp decl,
                                             KernelDeclAdaptor &adaptor) {
  if (!adaptor.executeFunction->outputResult.has_value())
    return success();
  // Copy sig to call non-const isThrows.
  auto sig = adaptor.executeFunction->attr.getSignature();
  if (adaptor.executeFunction->outputResult->isOpaqueType() && sig.isThrows()) {
    decl->emitError() << "Functions that raise and return a custom type are "
                         "not yet supported";
    return failure();
  }
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

      // Elementwise-only kernels: verify only the elementwise function.
      if (!adaptor.executeFunction.has_value()) {
        if (setFlagIfFailed(failed, declVerifyElementwiseFunction, decl,
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
