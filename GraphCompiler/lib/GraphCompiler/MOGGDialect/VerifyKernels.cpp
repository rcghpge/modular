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

//===----------------------------------------------------------------------===//
// MojoKernelAdaptor-based verification (legacy annotated-FnOp path)
//===----------------------------------------------------------------------===//

LogicalResult verifyHasExecuteFunction(LIT::StructDeclOp structDeclOp,
                                       const MojoKernelAdaptor &adaptor) {
  if (!adaptor.executeFunction.has_value()) {
    structDeclOp->emitError()
        << "All kernels must have an execution entry point named "
           "\"execute\"";
    return failure();
  }

  return success();
}

LogicalResult
verifyShapeFunctionHasNoDPSOutputs(const MojoKernelAdaptor &adaptor) {
  if (!adaptor.shapeFunction.has_value())
    return success();

  if (!adaptor.shapeFunction->outputArguments.empty()) {
    adaptor.shapeFunction->mojoCode->emitError()
        << "Shape functions must not have any output tensor arguments";
    return failure();
  }

  if (!adaptor.shapeFunction->outputResult.has_value()) {
    adaptor.shapeFunction->mojoCode->emitError()
        << "Shape functions must return a shape";
    return failure();
  }

  return success();
}

LogicalResult verifyShapeFunctionInputsMatchExecuteFunction(
    const MojoKernelAdaptor &adaptor) {
  if (!adaptor.shapeFunction.has_value())
    return success();

  auto skipContextType =
      [](const SmallVector<MojoKernelOperandAdaptor> &inputs,
         uint64_t &idx) -> std::optional<MojoKernelOperandAdaptor> {
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
      // Arguments of the shape / exec function must have the same type.
      if (*execArgOpt == *shapeArgOpt)
        return true;

      // One exception is Mutable tensors: Since shape functions can't mutate
      // their input, the shape operand must be an InputTensorType.
      return execArgOpt->isMutableTensorType() &&
             (shapeArgOpt->isTensorType() &&
              !shapeArgOpt->isMutableTensorType());
    };

    if (!isCompatibleArgs()) {
      return adaptor.executeFunction->mojoCode
                 ->getParentOfType<LIT::StructDeclOp>()
                 .emitError()
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
        [&](const SmallVector<MojoKernelOperandAdaptor> &inputs, uint64_t idx) {
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

    return adaptor.executeFunction->mojoCode
        ->getParentOfType<LIT::StructDeclOp>()
        ->emitError(errorMessage.str());
  }

  return success();
}

LogicalResult
verifyExecuteFunctionDoesNotReturnATensor(const MojoKernelAdaptor &adaptor) {

  if (!adaptor.executeFunction->outputResult.has_value())
    return success();

  if (adaptor.executeFunction->outputResult->isTensorType()) {
    adaptor.executeFunction->mojoCode->emitError()
        << "Value semantics kernels (return a tensor) are not yet supported";
    return failure();
  }

  return success();
}

LogicalResult verifyOutputsAppearBeforeInputs(MojoKernelAdaptor &adaptor) {
  int64_t indexOfFirstInput = std::numeric_limits<int64_t>::max();
  for (auto &arg : adaptor.executeFunction->inputArguments) {
    indexOfFirstInput =
        std::min<int64_t>(indexOfFirstInput, arg.sourceDescriptor->position);
  }

  int64_t indexOfLastOutput = std::numeric_limits<int64_t>::min();
  for (auto &arg : adaptor.executeFunction->outputArguments) {
    indexOfLastOutput =
        std::max<int64_t>(indexOfLastOutput, arg.sourceDescriptor->position);
  }

  if (indexOfFirstInput < indexOfLastOutput) {
    adaptor.executeFunction->mojoCode->emitError()
        << "Outputs must appear before inputs in the function signature";
    return failure();
  }

  return success();
}

LogicalResult verifyContextsAppearLastInSignature(MojoKernelAdaptor &adaptor) {
  const auto contextsAreLast =
      [](const std::optional<MojoKernelFunctionAdaptor> &func)
      -> LogicalResult {
    if (!func.has_value())
      return success();

    if (llvm::count_if(func->inputArguments,
                       [](const MojoKernelOperandAdaptor &arg) {
                         return arg.isContextType();
                       }) > 1) {
      func->mojoCode->emitError() << "There can be at most one context";
      return failure();
    }

    int64_t indexOfLastArgument = std::numeric_limits<int64_t>::min();
    int64_t indexOfContextArgument = std::numeric_limits<int64_t>::max();
    for (auto &arg : func->inputArguments) {
      if (arg.isContextType())
        indexOfContextArgument = arg.sourceDescriptor->position;
      indexOfLastArgument = std::max<int64_t>(indexOfLastArgument,
                                              arg.sourceDescriptor->position);
    }

    if (indexOfContextArgument < indexOfLastArgument) {
      func->mojoCode->emitError()
          << "Context argument must be the last argument in the function "
             "signature";
      return failure();
    }

    return success();
  };

  LogicalResult executeResult = contextsAreLast(adaptor.executeFunction);
  LogicalResult shapeResult = contextsAreLast(adaptor.shapeFunction);

  if (executeResult.failed() || shapeResult.failed())
    return failure();

  return success();
}

LogicalResult
verifyOpaqueArgumentsArePassedByReference(MojoKernelAdaptor &adaptor) {
  auto opaquesByReference =
      [](std::optional<MojoKernelFunctionAdaptor> &func) -> LogicalResult {
    if (!func.has_value())
      return success();

    auto conventions =
        func->mojoCode.getFuncTypeGenerator().getBody().getArgConventions();
    for (const auto &arg : func->inputArguments) {
      auto convention = conventions[arg.sourceDescriptor->position];
      if (arg.isOpaqueType() && (convention == KGEN::ArgConvention::OwnedMem ||
                                 convention == KGEN::ArgConvention::DeinitMem ||
                                 convention == KGEN::ArgConvention::OwnedReg)) {
        func->mojoCode.emitError()
            << "Argument convention for argument at position "
            << arg.sourceDescriptor->position
            << " is invalid: Owned arguments not supported for custom types "
               "in kernels";
        return failure();
      }
    }

    return success();
  };

  LogicalResult executeResult = opaquesByReference(adaptor.executeFunction);
  LogicalResult shapeResult = opaquesByReference(adaptor.shapeFunction);

  if (executeResult.failed() || shapeResult.failed())
    return failure();

  return success();
}

LogicalResult
verifyAtMostOneVariadicInputAndOutput(const MojoKernelAdaptor &adaptor) {
  /// TODO: (GEX-1994) Remove this.
  if (adaptor.isCoreMOOperation())
    return success();

  auto atMostOneVariadic =
      [](const std::optional<MojoKernelFunctionAdaptor> &func)
      -> LogicalResult {
    if (!func.has_value())
      return success();

    if (llvm::count_if(func->inputArguments,
                       [&](const MojoKernelOperandAdaptor &arg) {
                         return arg.isVariadicTensorType();
                       }) > 1) {
      func->mojoCode->emitError()
          << "Kernel function can have at most one variadic input tensors";
      return failure();
    }

    if (llvm::count_if(func->outputArguments,
                       [&](const MojoKernelOperandAdaptor &arg) {
                         return arg.isVariadicTensorType();
                       }) > 1) {
      func->mojoCode->emitError()
          << "Kernel function can have at most one variadic output tensors";
      return failure();
    }

    return success();
  };

  LogicalResult executeResult = atMostOneVariadic(adaptor.executeFunction);
  LogicalResult shapeResult = atMostOneVariadic(adaptor.shapeFunction);

  if (executeResult.failed() || shapeResult.failed())
    return failure();

  return success();
}

LogicalResult
verifyExecuteNotReturningOpaqueAndThrows(MojoKernelAdaptor &adaptor) {
  if (!adaptor.executeFunction->outputResult.has_value())
    return success();

  if (adaptor.executeFunction->outputResult->isOpaqueType() &&
      adaptor.executeFunction->mojoCode.getFuncTypeGenerator()
          .getBody()
          .isThrows()) {
    adaptor.executeFunction->mojoCode.emitError()
        << "Functions that raise and return a custom type are not yet "
           "supported";
    return failure();
  }

  return success();
}

LogicalResult
verifyAllOperandTypesAreSupported(const MojoKernelAdaptor &adaptor) {
  auto noUnsupportedArgument =
      [&](const std::optional<MojoKernelFunctionAdaptor> &function)
      -> LogicalResult {
    if (!function.has_value())
      return success();

    static constexpr StringLiteral commonMessage =
        "Traits are currently unsupported";

    for (auto &operand : function->inputArguments) {
      if (operand.isUnsupportedType()) {
        function->mojoCode.emitError(
            llvm::formatv("Unsupported input operand type for `{}`: {}",
                          operand.sourceDescriptor->sourceName, commonMessage));
        return failure();
      }
    }

    for (auto result : function->outputArguments) {
      if (result.isUnsupportedType()) {
        function->mojoCode.emitError(
            llvm::formatv("Unsupported output operand type for `{}`: {}",
                          result.sourceDescriptor->sourceName, commonMessage));
        return failure();
      }
    }

    if (function->outputResult.has_value() &&
        function->outputResult->isUnsupportedType()) {
      function->mojoCode.emitError(
          llvm::formatv("Unsupported result type: {}", commonMessage));
      return failure();
    }

    return success();
  };

  if (noUnsupportedArgument(adaptor.executeFunction).failed())
    return failure();

  return noUnsupportedArgument(adaptor.shapeFunction);
}

LogicalResult verifyElementwiseFunction(const MojoKernelAdaptor &adaptor) {

  if (!adaptor.elementwiseFunction.has_value())
    return success();

  const MojoKernelFunctionAdaptor &elemwiseFn = *adaptor.elementwiseFunction;

  // The elementwise must only take SIMD in / outs and the IndexList operand.
  // TODO: Check if the scalars I/Os are properly parametrized SIMD values.

  if (elemwiseFn.mojoCode.isThrows())
    return elemwiseFn.mojoCode->emitError()
           << "Elementwise kernels can't throw";

  if (!elemwiseFn.outputArguments.empty())
    return elemwiseFn.mojoCode->emitError()
           << "Elementwise kernels must return an SIMD value";
  if (!elemwiseFn.outputResult.has_value() ||
      !elemwiseFn.outputResult->isScalarType())
    return elemwiseFn.mojoCode->emitError()
           << "Elementwise kernels must return an SIMD value";

  size_t operandsCount = 0;
  for (size_t i = 0, inputsCount = elemwiseFn.inputArguments.size();
       i < inputsCount; ++i) {
    // The last argument might be an IndexList.
    // Since only elementwise kernels use them, no need to create a special
    // type, just check the opaque typename.
    if (i + 1 == inputsCount &&
        elemwiseFn.inputArguments[i].isOpaqueWithTypeName(
            MOGG::MOJO_INTERNAL_DPS_INDEX_LIST_TYPE_NAME)) {
      continue;
    }

    // All other operands must be SIMD.
    ++operandsCount;
    if (!elemwiseFn.inputArguments[i].isScalarType())
      return elemwiseFn.mojoCode->emitError()
             << "Elementwise kernels must have SIMD operands";
  }

  if (!operandsCount)
    return elemwiseFn.mojoCode->emitError()
           << "Elementwise kernels must have at least one operand";

  return success();
}

template <typename Func, typename... Args>
[[nodiscard]] LogicalResult setFlagIfFailed(bool &flag, Func &&func,
                                            Args &&...args) {
  LogicalResult ret = func(std::forward<Args>(args)...);
  if (ret.failed())
    flag |= true;

  return ret;
}

struct KernelFuncs {
  LIT::FnOp execute;
  LIT::FnOp shape;
  LIT::FnOp elementwise;
};

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
            MOGG::MOJO_INTERNAL_DPS_INDEX_LIST_TYPE_QUALIFIED_NAME)) {
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

    // If mogg.kernel.decl ops are present, use KernelDeclAdaptor.
    if (!moduleOp.getOps<KernelDeclOp>().empty()) {
      for (KernelDeclOp decl : moduleOp.getOps<KernelDeclOp>()) {
        KernelDeclAdaptor adaptor(decl);

        if (setFlagIfFailed(failed, declVerifyHasEntryPoint, decl, adaptor)
                .failed()) {
          continue;
        }

        // Elementwise-only kernels: verify only the elementwise function.
        if (!adaptor.executeFunction.has_value()) {
          if (setFlagIfFailed(failed, declVerifyElementwiseFunction, decl,
                              adaptor)
                  .failed())
            continue;
          continue;
        }

        if (setFlagIfFailed(failed, declVerifyShapeFunctionHasNoDPSOutputs,
                            decl, adaptor)
                .failed())
          continue;

        if (setFlagIfFailed(failed,
                            declVerifyShapeFunctionInputsMatchExecuteFunction,
                            decl, adaptor)
                .failed())
          continue;

        if (setFlagIfFailed(failed,
                            declVerifyExecuteFunctionDoesNotReturnATensor, decl,
                            adaptor)
                .failed())
          continue;

        if (setFlagIfFailed(failed, declVerifyOutputsAppearBeforeInputs, decl,
                            adaptor)
                .failed())
          continue;

        if (setFlagIfFailed(failed, declVerifyContextsAppearLastInSignature,
                            decl, adaptor)
                .failed())
          continue;

        if (setFlagIfFailed(failed,
                            declVerifyOpaqueArgumentsArePassedByReference, decl,
                            adaptor)
                .failed())
          continue;

        if (setFlagIfFailed(failed, declVerifyAtMostOneVariadicInputAndOutput,
                            decl, adaptor)
                .failed())
          continue;

        if (setFlagIfFailed(failed,
                            declVerifyExecuteNotReturningOpaqueAndThrows, decl,
                            adaptor)
                .failed())
          continue;

        if (setFlagIfFailed(failed, declVerifyElementwiseFunction, decl,
                            adaptor)
                .failed())
          continue;
      }

      if (failed)
        signalPassFailure();
      return;
    }

    // Fallback: walk over annotated FnOps and use MojoKernelAdaptor.
    llvm::DenseMap<LIT::StructDeclOp, KernelFuncs> kernels;
    moduleOp.walk([&](LIT::FnOp fnOp) {
      auto structName = fnOp->getAttrOfType<StringAttr>(kMOGGKernelStructName);
      if (!structName)
        return;

      auto structDeclOp = fnOp->getParentOfType<LIT::StructDeclOp>();
      if (fnOp->hasAttr(kMOGGExecuteFunctionLabel)) {
        kernels[structDeclOp].execute = fnOp;
      } else if (fnOp->hasAttr(kMOGGShapeFunctionLabel)) {
        kernels[structDeclOp].shape = fnOp;
      } else if (fnOp->hasAttr(kMOGGElementwiseFunctionLabel)) {
        kernels[structDeclOp].elementwise = fnOp;
      }
    });

    for (auto &[structDeclOp, entry] : kernels) {
      MojoKernelAdaptor adaptor(
          structDeclOp.getSourceName().value().getName().strref(),
          entry.execute, entry.shape, entry.elementwise);

      if (setFlagIfFailed(failed, verifyAllOperandTypesAreSupported, adaptor)
              .failed())
        continue;

      if (setFlagIfFailed(failed, verifyHasExecuteFunction, structDeclOp,
                          adaptor)
              .failed())
        continue;

      if (setFlagIfFailed(failed, verifyShapeFunctionHasNoDPSOutputs, adaptor)
              .failed())
        continue;

      if (setFlagIfFailed(failed, verifyShapeFunctionInputsMatchExecuteFunction,
                          adaptor)
              .failed())
        continue;

      if (setFlagIfFailed(failed, verifyExecuteFunctionDoesNotReturnATensor,
                          adaptor)
              .failed())
        continue;

      if (setFlagIfFailed(failed, verifyOutputsAppearBeforeInputs, adaptor)
              .failed())
        continue;

      if (setFlagIfFailed(failed, verifyContextsAppearLastInSignature, adaptor)
              .failed())
        continue;

      if (setFlagIfFailed(failed, verifyOpaqueArgumentsArePassedByReference,
                          adaptor)
              .failed())
        continue;

      if (setFlagIfFailed(failed, verifyAtMostOneVariadicInputAndOutput,
                          adaptor)
              .failed())
        continue;

      if (setFlagIfFailed(failed, verifyExecuteNotReturningOpaqueAndThrows,
                          adaptor)
              .failed())
        continue;

      if (setFlagIfFailed(failed, verifyElementwiseFunction, adaptor).failed())
        continue;
    }

    if (failed)
      signalPassFailure();
  }
};

} // namespace

} // namespace M::MOGG
